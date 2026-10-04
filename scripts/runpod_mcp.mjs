#!/usr/bin/env node
// Generic Runpod MCP caller (uses the cached mcp-remote OAuth session).
//   node scripts/runpod_mcp.mjs list-tools
//   node scripts/runpod_mcp.mjs <tool-name> '<json-arguments>'
// Waits for real JSON-RPC responses (no fixed sleeps) and always exits with a hard timeout.
import { spawn } from "child_process";

const [tool, argJson = "{}"] = process.argv.slice(2);
if (!tool) {
  console.error("Usage: runpod_mcp.mjs <list-tools|tool-name> [json-args]");
  process.exit(2);
}

const proc = spawn("npx", ["-y", "mcp-remote", "https://mcp.getrunpod.io/"], {
  stdio: ["pipe", "pipe", "pipe"],
});
const pending = new Map();
let buf = "";
proc.stdout.on("data", (d) => {
  buf += d.toString();
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (!line) continue;
    try {
      const msg = JSON.parse(line);
      if (msg.id !== undefined && pending.has(msg.id)) {
        pending.get(msg.id)(msg);
        pending.delete(msg.id);
      }
    } catch (_) { /* non-JSON log line */ }
  }
});
proc.stderr.on("data", () => {});

const send = (obj) => proc.stdin.write(JSON.stringify(obj) + "\n");
const rpc = (id, method, params) =>
  new Promise((resolve) => {
    pending.set(id, resolve);
    send({ jsonrpc: "2.0", id, method, params });
  });

const hard = setTimeout(() => {
  console.error("TIMEOUT: no response from Runpod MCP within 60s");
  proc.kill();
  process.exit(3);
}, 60000);

(async () => {
  await rpc(1, "initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "antigravity", version: "1.0.0" },
  });
  send({ jsonrpc: "2.0", method: "notifications/initialized" });
  const res =
    tool === "list-tools"
      ? await rpc(2, "tools/list", {})
      : await rpc(2, "tools/call", { name: tool, arguments: JSON.parse(argJson) });
  if (tool === "list-tools") {
    for (const t of res.result?.tools ?? []) console.log(t.name);
  } else {
    const content = res.result?.content ?? [];
    for (const c of content) console.log(c.text ?? JSON.stringify(c));
    if (res.error) console.log(JSON.stringify(res.error));
    if (res.result?.isError) process.exitCode = 1;
  }
  clearTimeout(hard);
  proc.kill();
  process.exit(process.exitCode ?? 0);
})();
