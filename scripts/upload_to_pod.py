#!/usr/bin/env python3
import base64
import os
import subprocess
import sys

FILES_TO_SYNC = [
    "scripts/run_scale_50clients.py",
    "scripts/run_cifar100_ablation.py",
    "scripts/run_master_pipeline.sh",
]

def upload_file(local_path: str, remote_path: str):
    print(f"Uploading {local_path} -> {remote_path}...")
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    
    remote_dir = os.path.dirname(remote_path)
    cmd = (
        f"mkdir -p {remote_dir} && "
        f"cat << 'EOF' | base64 -d > {remote_path}\n"
        f"{b64}\n"
        f"EOF\n"
        f"ls -lh {remote_path}"
    )
    res = subprocess.run(["node", "scripts/runpod_exec.mjs", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED uploading {local_path}: RC {res.returncode}")
        print(res.stderr)
        sys.exit(res.returncode)
    print(res.stdout.strip())
    print(f"Uploaded {local_path} successfully.")

if __name__ == "__main__":
    for f in FILES_TO_SYNC:
        remote = f"/workspace/Topology-aware-FDL/{f}"
        upload_file(f, remote)
    print("All files synchronized to pod successfully.")
