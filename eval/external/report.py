"""Merge-write ``external_<run_id>.json``, ``external_latest.json``, and the judge sample.

``external_latest.json`` holds a dict keyed by ``f"{bench}:{mode}"`` so multiple benches/modes
coexist across separate runs; every write here loads the existing file, updates one key, and
writes the whole thing back so an unrelated run's results are never dropped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .common import RESULTS_DIR


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def write_results(results: dict, results_dir: Path | str | None = None) -> None:
    """Write ``external_<run_id>.json`` and merge ``results`` into ``external_latest.json``."""
    rdir = Path(results_dir) if results_dir else RESULTS_DIR
    run_path = rdir / f"external_{results['run_id']}.json"
    _write(run_path, results)

    latest_path = rdir / "external_latest.json"
    latest = _load(latest_path)
    key = f"{results['bench']}:{results['mode']}"
    latest[key] = results
    _write(latest_path, latest)


def _stratified_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin sample across (category, label) groups, seeded per-group shuffle."""
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        key = (str(it.get("category")), str(it.get("label")))
        groups.setdefault(key, []).append(it)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    keys = sorted(groups)
    out: list[dict] = []
    i = 0
    while len(out) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            out.append(groups[k].pop(0))
        i += 1
    return out


def write_judge_sample(
    items: list[dict], n: int = 60, seed: int = 7, results_dir: Path | str | None = None
) -> None:
    """Stratified sample (by category, verdict) of judged items for human calibration review.

    Merge-written into ``external_judge_sample.json`` keyed by ``f"{bench}:{mode}"`` (read from
    the items themselves), same convention as :func:`write_results`.
    """
    rdir = Path(results_dir) if results_dir else RESULTS_DIR
    judged = [it for it in items if it.get("label") is not None]
    if not judged:
        return
    sample = _stratified_sample(judged, n, seed)

    bench = judged[0].get("bench", "unknown")
    mode = judged[0].get("mode", "unknown")
    key = f"{bench}:{mode}"

    path = rdir / "external_judge_sample.json"
    data = _load(path)
    data[key] = sample
    _write(path, data)
