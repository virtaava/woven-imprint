"""Streamed dataset download with size verification.

Sizes and URLs verified 2026-08-27 (``curl -sI -L <url>``): the
``Content-Length`` on a HEAD/redirect-followed request matches the byte
count below exactly for all four datasets when the response isn't
compressed. In practice both hosts (GitHub raw, HF's CDN) may gzip the
transfer, in which case ``Content-Length`` is the *compressed* size and
can't be checked against the (decompressed) expected size up front — the
authoritative check is always the final on-disk file size after streaming
decompresses it.
"""

from __future__ import annotations

from pathlib import Path

import requests

from .common import DATA_DIR

# name -> (url, expected size in bytes)
DATASETS: dict[str, tuple[str, int]] = {
    "locomo": (
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        2805274,
    ),
    "locomo_plus": (
        "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo_plus.json",
        305737,
    ),
    "longmemeval_s": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        277383467,
    ),
    "longmemeval_oracle": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
        15388478,
    ),
}


def fetch(name: str, force: bool = False) -> Path:
    """Download ``DATASETS[name]`` to ``DATA_DIR/<name>.json``, streamed.

    Skips the download when the target file already exists with the
    expected size and ``force`` is False. Verifies the server's
    ``Content-Length`` against the expected size up front when the response
    isn't content-encoded, and always verifies the final on-disk size;
    raises ``ValueError`` on a mismatch (the partial file is removed).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choices: {sorted(DATASETS)}")
    url, expected_size = DATASETS[name]
    dest = DATA_DIR / f"{name}.json"

    if not force and dest.exists() and dest.stat().st_size == expected_size:
        return dest

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        # requests/urllib3 transparently decompresses gzip/deflate while
        # streaming, so Content-Length (the *transfer* size) only equals the
        # decompressed byte count when the response isn't content-encoded.
        # GitHub raw and HF's CDN both gzip these JSON files, so skip this
        # pre-check in that case and rely on the final on-disk size below.
        content_encoding = (resp.headers.get("Content-Encoding") or "identity").lower()
        content_length = resp.headers.get("Content-Length")
        if (
            content_encoding == "identity"
            and content_length is not None
            and int(content_length) != expected_size
        ):
            raise ValueError(
                f"{name}: server Content-Length {content_length} != expected {expected_size}"
            )
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{name}: downloaded size {actual_size} != expected {expected_size}")
    tmp.replace(dest)
    return dest
