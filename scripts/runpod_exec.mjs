#!/usr/bin/env node

const POD_ID = process.env.RUNPOD_POD_ID;
const JUPYTER_PASSWORD = process.env.RUNPOD_JUPYTER_PASSWORD;

if (!POD_ID || !JUPYTER_PASSWORD) {
  console.error("Error: RUNPOD_POD_ID and RUNPOD_JUPYTER_PASSWORD environment variables are required.");
  console.error("Usage: RUNPOD_POD_ID=<pod_id> RUNPOD_JUPYTER_PASSWORD=<password> node runpod_exec.mjs <shell-command>");
  process.exit(1);
}

const BASE_URL = `https://${POD_ID}-8888.proxy.runpod.net`;

const cmd = process.argv.slice(2).join(" ");
if (!cmd) {
  console.error("Usage: node runpod_exec.mjs <shell-command>");
  process.exit(1);
}

async function run() {
  const loginRes = await fetch(`${BASE_URL}/login?next=%2Flab`);
  const cookieHeader = loginRes.headers.get("set-cookie") || "";
  const xsrfMatch = cookieHeader.match(/_xsrf=([^;]+)/);
  const xsrf = xsrfMatch ? xsrfMatch[1] : "";

  const postParams = new URLSearchParams();
  postParams.append("_xsrf", xsrf);
  postParams.append("password", JUPYTER_PASSWORD);

  const authRes = await fetch(`${BASE_URL}/login?next=%2Flab`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      "Cookie": `_xsrf=${xsrf}`,
      "Referer": `${BASE_URL}/login?next=%2Flab`
    },
    body: postParams.toString(),
    redirect: "manual"
  });

  const authCookies = authRes.headers.getSetCookie ? authRes.headers.getSetCookie() : [authRes.headers.get("set-cookie")];
  const cookieMap = {};
  for (const c of [cookieHeader, ...authCookies]) {
    if (!c) continue;
    const parts = c.split(";")[0].split("=");
    if (parts.length >= 2) cookieMap[parts[0].trim()] = parts.slice(1).join("=").trim();
  }
  const fullCookie = Object.entries(cookieMap).map(([k,v]) => `${k}=${v}`).join("; ");

  const finalXsrf = cookieMap["_xsrf"] || xsrf;
  const termRes = await fetch(`${BASE_URL}/api/terminals?_xsrf=${encodeURIComponent(finalXsrf)}`, {
    method: "POST",
    headers: {
      "Cookie": fullCookie,
      "X-XSRFToken": finalXsrf
    }
  });
  if (!termRes.ok) {
    throw new Error(`Failed to create terminal: ${termRes.status} ${await termRes.text()}`);
  }
  const termData = await termRes.json();
  const termName = termData.name;

  const wsUrl = `wss://${POD_ID}-8888.proxy.runpod.net/terminals/websocket/${termName}`;
  const ws = new WebSocket(wsUrl, {
    headers: {
      "Cookie": fullCookie,
      "Origin": BASE_URL
    }
  });

  const END_MARKER = `__COMMAND_DONE_${Date.now()}_${Math.random().toString(36).slice(2)}__`;
  let exitCode = 0;

  ws.onopen = () => {
    setTimeout(() => {
      const wrappedCmd = `${cmd}\n__RC=$?; echo "${END_MARKER}:$__RC"\n`;
      ws.send(JSON.stringify(["stdin", wrappedCmd]));
    }, 1500);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg[0] === "stdout" && msg[1]) {
        const text = msg[1];
        process.stdout.write(text);
        if (text.includes(END_MARKER)) {
          const match = text.match(new RegExp(`${END_MARKER}:(\\d+)`));
          if (match) {
            exitCode = parseInt(match[1], 10);
          }
          setTimeout(() => {
            ws.close();
            fetch(`${BASE_URL}/api/terminals/${termName}`, {
              method: "DELETE",
              headers: { "Cookie": fullCookie, "X-XSRFToken": cookieMap["_xsrf"] || xsrf }
            }).finally(() => {
              process.exit(exitCode);
            });
          }, 500);
        }
      }
    } catch (e) {}
  };

  ws.onerror = (err) => {
    console.error("WS Error:", err);
    process.exit(1);
  };
}

run().catch((err) => {
  console.error("Runner Error:", err);
  process.exit(1);
});
