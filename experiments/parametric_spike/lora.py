"""Toggle llama-server LoRA adapters at runtime."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import requests

SERVE_SH = Path(__file__).resolve().parent / "serve.sh"


def list_adapters(base: str = "http://127.0.0.1:11810") -> list[dict]:
    r = requests.get(f"{base}/lora-adapters", timeout=10)
    r.raise_for_status()
    return r.json()


def set_adapters(scales: dict[str, float], base: str = "http://127.0.0.1:11810") -> list[dict]:
    """Apply scales by adapter name (file stem). Adapters not named get 0.0."""
    current = list_adapters(base)
    body = [{"id": a["id"], "scale": float(scales.get(Path(a["path"]).stem, 0.0))} for a in current]
    r = requests.post(f"{base}/lora-adapters", json=body, timeout=30)
    r.raise_for_status()
    return list_adapters(base)


def restart_server(base: str = "http://127.0.0.1:11810", timeout_s: int = 120) -> None:
    """Kill any llama-server bound to :11810 (pid-based — never pkill -f), relaunch serve.sh detached, wait for /health."""
    for pid in subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout.split():
        args = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True).stdout
        if "11810" in args:
            os.kill(int(pid), signal.SIGTERM)
    time.sleep(3)
    log = open(SERVE_SH.parent / "out" / "serve.log", "ab")
    subprocess.Popen(["bash", str(SERVE_SH)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if requests.get(f"{base}/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("spike server did not become healthy")
