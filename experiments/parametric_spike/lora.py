"""Toggle llama-server LoRA adapters at runtime."""
from __future__ import annotations

from pathlib import Path

import requests


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
