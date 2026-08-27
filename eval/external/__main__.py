"""CLI: ``python -m eval.external fetch <names>``.

Task 3 adds the ``run`` subcommand alongside this one.
"""

from __future__ import annotations

import argparse
import sys

from .fetch import DATASETS, fetch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.external")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="Download benchmark datasets")
    fetch_parser.add_argument("names", nargs="+", choices=sorted(DATASETS))
    fetch_parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file already exists"
    )

    args = parser.parse_args(argv)

    if args.command == "fetch":
        for name in args.names:
            path = fetch(name, force=args.force)
            print(f"{name}: {path} ({path.stat().st_size} bytes)")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
