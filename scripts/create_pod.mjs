import { spawn } from "child_process";

const proc = spawn("npx", ["-y", "mcp-remote", "https://mcp.getrunpod.io/"], {
  stdio: ["pipe", "pipe", "pipe"],
  env: { ...process.env }
});

let out = "";
proc.stdout.on("data", d => { out += d.toString(); });
proc.stderr.on("data", d => { process.stderr.write(d); });

proc.stdin.write(JSON.stringify({
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "antigravity", version: "1.0.0" } }
}) + "\n");

setTimeout(() => {
  proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
  proc.stdin.write(JSON.stringify({
    jsonrpc: "2.0",
    id: 2,
    method: "tools/call",
    params: {
      name: "create-pod",
      arguments: {
        body: {
          name: "fedhep-4090-suite",
          cloud: "SECURE",
          templateId: "runpod-torch-v240",
          gpu: {
            id: "NVIDIA GeForce RTX 4090",
            count: 1
          },
          mounts: {
            persistent: {
              size: 20,
              path: "/workspace"
            }
          }
        }
      }
    }
  }) + "\n");
}, 2000);

setTimeout(() => {
  proc.kill();
  const lines = out.split("\n").filter(Boolean);
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed.id === 2) {
        console.log("POD_CREATION_RESULT:", JSON.stringify(parsed, null, 2));
      }
    } catch(e) {}
  }
  process.exit(0);
}, 10000);
