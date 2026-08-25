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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal (shouldn't happen here)
    return True


def _is_gone(pid: int) -> bool:
    """True once pid has exited.

    restart_server() launches llama-server via subprocess.Popen from this same
    long-lived process (bench_facts.py calls it repeatedly across one run), so
    a killed instance is our own child: the kernel keeps it as a zombie until
    we reap it, and a plain os.kill(pid, 0) existence check never clears for a
    zombie — it can report "still alive" forever even though the process
    finished and freed all its memory. os.waitpid(pid, WNOHANG) both detects
    the exit and reaps it. For a pid that is *not* our child (e.g. a
    llama-server left running from an earlier, now-exited process — the
    common case on first use), waitpid raises ChildProcessError immediately;
    that pid's real parent already died, so it was reparented to init, which
    reaps it automatically once it actually exits, and the plain existence
    check is correct there.
    """
    try:
        reaped_pid, _status = os.waitpid(pid, os.WNOHANG)
        return reaped_pid == pid
    except ChildProcessError:
        return not _pid_alive(pid)


def _wait_for_exit(pids: list[int], timeout_s: float) -> list[int]:
    """Poll _is_gone() until each pid is gone or timeout_s elapses. Returns still-alive pids."""
    t0 = time.time()
    alive = set(pids)
    while alive and time.time() - t0 < timeout_s:
        alive = {p for p in alive if not _is_gone(p)}
        if alive:
            time.sleep(0.5)
    return sorted(alive)


def restart_server(base: str = "http://127.0.0.1:11810", timeout_s: int = 120) -> None:
    """Kill any llama-server bound to :11810 (pid-based — never pkill -f), relaunch serve.sh detached, wait for /health.

    Waits for each SIGTERM'd pid to actually exit (up to 20s) before relaunching —
    llama-server can release its listen socket well before it finishes tearing down
    its CUDA context, so a fixed sleep + "new instance is healthy" check is not
    sufficient: it lets the old process linger, holding several GB of host RAM and
    GPU memory, while a new one is already serving. If a pid survives the 20s
    SIGTERM wait it is SIGKILL'd and given up to 10s more; a pid that survives
    even that raises RuntimeError rather than silently proceeding to relaunch on
    top of a still-live process.
    """
    pids = []
    for pid in subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout.split():
        args = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True).stdout
        if "11810" in args:
            pids.append(int(pid))
            os.kill(int(pid), signal.SIGTERM)

    still_alive = _wait_for_exit(pids, timeout_s=20)
    if still_alive:
        for pid in still_alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        still_alive = _wait_for_exit(still_alive, timeout_s=10)
    if still_alive:
        raise RuntimeError(f"llama-server pid(s) {still_alive} survived SIGTERM+SIGKILL; refusing to relaunch")

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
