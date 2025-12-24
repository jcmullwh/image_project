from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image_project", add_help=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Run a generation pipeline")
    sub.add_parser("list-stages", help="List available stages")
    sub.add_parser("list-plans", help="List available prompt plans")

    run_review = sub.add_parser("run-review", help="Generate an HTML/JSON run report")
    run_review.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.command == "generate":
        from .app.generate import main as generate_main

        generate_main()
        return 0

    if args.command == "list-stages":
        from .impl.current.prompting import list_stages

        list_stages()
        return 0

    if args.command == "list-plans":
        from .impl.current.plans import list_plans

        list_plans()
        return 0

    if args.command == "run-review":
        from .run_review.cli import main as run_review_main

        return int(run_review_main(args.args))

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
