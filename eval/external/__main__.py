"""CLI: ``python -m eval.external fetch <names>`` / ``run --bench ... --mode ...``."""

from __future__ import annotations

import argparse
import sys

from .fetch import DATASETS, fetch
from .runner import RunConfig, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.external")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="Download benchmark datasets")
    fetch_parser.add_argument("names", nargs="+", choices=sorted(DATASETS))
    fetch_parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file already exists"
    )

    run_parser = sub.add_parser("run", help="Run a benchmark (ingest, answer, judge)")
    run_parser.add_argument("--bench", required=True, choices=["locomo", "longmemeval_s"])
    run_parser.add_argument("--mode", required=True, choices=["memory", "fullcontext"])
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--limit", type=int, default=None, help="Max conversations")
    run_parser.add_argument(
        "--sample", type=int, default=None, help="Stratified sample size (longmemeval_s)"
    )
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--k", type=int, default=20, help="Retrieval limit per question")
    run_parser.add_argument(
        "--interval", type=int, default=1, help="memory.fact_extraction_interval during ingest"
    )
    run_parser.add_argument(
        "--max-sessions", type=int, default=None, help="Cap sessions ingested per conversation"
    )
    run_parser.add_argument(
        "--max-questions", type=int, default=None, help="Cap questions answered per conversation"
    )
    run_parser.add_argument(
        "--reuse-run", default=None, help="Reuse another run's ingested DBs (Task 4; not yet wired)"
    )

    args = parser.parse_args(argv)

    if args.command == "fetch":
        for name in args.names:
            path = fetch(name, force=args.force)
            print(f"{name}: {path} ({path.stat().st_size} bytes)")
        return 0

    if args.command == "run":
        cfg = RunConfig(
            bench=args.bench,
            mode=args.mode,
            run_id=args.run_id,
            k=args.k,
            limit_conversations=args.limit,
            sample=args.sample,
            seed=args.seed,
            fact_extraction_interval=args.interval,
            max_sessions=args.max_sessions,
            max_questions=args.max_questions,
            reuse_run=args.reuse_run,
        )
        results = run(cfg)
        summary = results["summary"]
        print(
            f"\n{cfg.bench}:{cfg.mode} run {cfg.run_id} done — "
            f"overall_j={summary['overall_j']:.3f} overall_f1={summary['overall_f1']:.3f} "
            f"n={summary['n_questions']} adversarial_acc={summary['adversarial_accuracy']} "
            f"abstain_acc={summary['abstain_accuracy']}"
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
