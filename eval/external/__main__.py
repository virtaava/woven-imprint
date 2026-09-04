"""CLI: ``python -m eval.external fetch <names>`` / ``run --bench ... --mode ...``."""

from __future__ import annotations

import argparse
import sys

from .fetch import DATASETS, fetch
from .runner import RunConfig, rejudge, run, run_plus


def _parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    index, count = value.split("/")
    return int(index), int(count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.external")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="Download benchmark datasets")
    fetch_parser.add_argument("names", nargs="+", choices=sorted(DATASETS))
    fetch_parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file already exists"
    )

    run_parser = sub.add_parser("run", help="Run a benchmark (ingest, answer, judge)")
    run_parser.add_argument(
        "--bench", required=True, choices=["locomo", "longmemeval_s", "locomo_plus"]
    )
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
        "--reuse-run",
        default=None,
        help="locomo_plus memory mode (required): run-id of a prior "
        "'--bench locomo --mode memory' run to copy ingested DBs from",
    )
    run_parser.add_argument(
        "--reuse-ingest",
        default=None,
        metavar="RUN_ID",
        help="Before ingesting a conversation, if this run's own <conv>.db/.ingest.json are "
        "missing and RUN_ID (a sibling run under the same root) has both, copy them over so "
        "this run answers/judges without re-ingesting — e.g. re-running with different "
        "--set config overrides against an already-ingested run's DBs",
    )
    run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a woven-imprint config field for this run (repeatable)",
    )
    run_parser.add_argument(
        "--shard",
        default=None,
        help="i/n: process only conversations with index %% n == i (parallel workers)",
    )
    run_parser.add_argument(
        "--no-results",
        action="store_true",
        help="Skip writing eval/results files (use for shard workers; aggregate with a final run)",
    )
    run_parser.add_argument(
        "--timeout", type=int, default=900, help="Per-request LLM timeout in seconds"
    )
    run_parser.add_argument(
        "--force-aggregate",
        action="store_true",
        help="Skip the unsharded-aggregation safety check (only if you know shard workers are done)",
    )
    pair_group = run_parser.add_mutually_exclusive_group()
    pair_group.add_argument(
        "--pair-turns",
        dest="pair_turns",
        action="store_true",
        default=None,
        help="Ingest consecutive (user, assistant) turns as one Character.ingest_exchange() "
        "call (default: on for --bench longmemeval_s, off otherwise)",
    )
    pair_group.add_argument(
        "--no-pair-turns",
        dest="pair_turns",
        action="store_false",
        default=None,
        help="Ingest every turn individually via Character.ingest() (see --pair-turns)",
    )
    run_parser.add_argument(
        "--delete-db-after-answer",
        action="store_true",
        help="Delete a conversation's .db (+ -wal/-shm) once its answers are complete and "
        "judged — keeps the ingest.json/answers.json checkpoints",
    )
    run_parser.add_argument(
        "--agg-stage",
        action="store_true",
        help="Tier 3n: answer aggregation-shaped questions (how many/in total/what order/...) "
        "via enumerate-then-answer — per-question memory.query_expansion=3 retrieval + a "
        "larger answer budget (default off: byte-identical to a run without this flag)",
    )

    rejudge_parser = sub.add_parser(
        "rejudge", help="Re-run the judge over an existing run's answer checkpoints, in place"
    )
    rejudge_parser.add_argument("--bench", required=True, choices=["locomo", "longmemeval_s"])
    rejudge_parser.add_argument("--mode", required=True, choices=["memory", "fullcontext"])
    rejudge_parser.add_argument("--run-id", required=True)
    rejudge_parser.add_argument(
        "--only-unjudged",
        action="store_true",
        help="Skip records already carrying judge_version 2 (resume a killed rejudge pass)",
    )
    rejudge_parser.add_argument(
        "--timeout", type=int, default=900, help="Per-request LLM timeout in seconds"
    )
    rejudge_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Stratified sample size (longmemeval_s) — must match the original run's --sample "
        "so the reload doesn't pull in conversations that were never run",
    )
    rejudge_parser.add_argument("--seed", type=int, default=7)
    rejudge_parser.add_argument(
        "--interval", type=int, default=1, help="Recorded into config when no stored results exist"
    )
    rejudge_parser.add_argument(
        "--k", type=int, default=20, help="Recorded into config when no stored results exist"
    )
    rejudge_parser.add_argument(
        "--agg-stage",
        action="store_true",
        help="Recorded into config when no stored results exist (see `run --agg-stage`)",
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
            shard=_parse_shard(args.shard),
            overrides=tuple(args.overrides),
            write_results=not args.no_results,
            sample=args.sample,
            seed=args.seed,
            fact_extraction_interval=args.interval,
            max_sessions=args.max_sessions,
            max_questions=args.max_questions,
            reuse_run=args.reuse_run,
            reuse_ingest=args.reuse_ingest,
            timeout=args.timeout,
            force_aggregate=args.force_aggregate,
            pair_turns=args.pair_turns,
            delete_db_after_answer=args.delete_db_after_answer,
            agg_stage=args.agg_stage,
        )
        if cfg.bench == "locomo_plus":
            results = run_plus(cfg)
            summary = results["summary"]
            print(
                f"\n{cfg.bench}:{cfg.mode} run {cfg.run_id} done — "
                f"cognitive_accuracy={summary['cognitive_accuracy']:.3f} "
                f"n={summary['n_probes']} mean_tokens={summary['mean_prompt_tokens_est']:.0f}"
            )
        else:
            results = run(cfg)
            summary = results["summary"]
            print(
                f"\n{cfg.bench}:{cfg.mode} run {cfg.run_id} done — "
                f"overall_j={summary['overall_j']:.3f} overall_f1={summary['overall_f1']:.3f} "
                f"n={summary['n_questions']} adversarial_acc={summary['adversarial_accuracy']} "
                f"abstain_acc={summary['abstain_accuracy']}"
            )
        return 0

    if args.command == "rejudge":
        cfg = RunConfig(
            bench=args.bench,
            mode=args.mode,
            run_id=args.run_id,
            timeout=args.timeout,
            sample=args.sample,
            seed=args.seed,
            fact_extraction_interval=args.interval,
            k=args.k,
            agg_stage=args.agg_stage,
        )
        results = rejudge(cfg, only_unjudged=args.only_unjudged)
        summary = results["summary"]
        print(
            f"\n{cfg.bench}:{cfg.mode} run {cfg.run_id} rejudged — "
            f"overall_j={summary['overall_j']:.3f} overall_f1={summary['overall_f1']:.3f} "
            f"n={summary['n_questions']} adversarial_acc={summary['adversarial_accuracy']} "
            f"abstain_acc={summary['abstain_accuracy']}"
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
