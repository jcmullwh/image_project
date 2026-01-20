from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image_project", add_help=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Run a generation pipeline")
    sub.add_parser("list-stages", help="List available stages")
    sub.add_parser("list-plans", help="List available prompt plans")
    experiments = sub.add_parser("experiments", help="List/describe/run registered experiments")
    experiments.add_argument("args", nargs=argparse.REMAINDER)
    index_artifacts = sub.add_parser(
        "index-artifacts",
        help="Build an index of experiment plans and images under _artifacts/",
    )
    index_artifacts.add_argument(
        "--artifacts-root",
        action="append",
        default=[],
        help="Artifacts root directory to scan. Repeat to build a combined index across multiple roots.",
    )
    index_artifacts.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write index files (default: <artifacts-root>/index, or ./_artifacts/index_combined for multi-root).",
    )

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
        from .stages.registry import get_stage_registry

        for entry in get_stage_registry().describe():
            stage_id = entry.get("stage_id")
            if not isinstance(stage_id, str) or not stage_id.strip():
                continue
            doc = entry.get("doc") if isinstance(entry, dict) else None
            doc_str = (doc or "").strip() if isinstance(doc, str) else ""
            print(f"{stage_id}\t{doc_str}".rstrip())
        return 0

    if args.command == "list-plans":
        from .impl.current.plans import list_plans

        list_plans()
        return 0

    if args.command == "run-review":
        from .run_review.cli import main as run_review_main

        return int(run_review_main(args.args))

    if args.command == "experiments":
        from .app.experiments_cli import main as experiments_main

        return int(experiments_main(getattr(args, "args", None)))

    if args.command == "index-artifacts":
        from .framework.artifacts import update_artifacts_index, update_artifacts_index_combined

        roots = list(getattr(args, "artifacts_root", []) or [])
        if not roots:
            roots = ["_artifacts"]

        if len(roots) == 1:
            payload = update_artifacts_index(
                artifacts_root=roots[0],
                output_dir=args.output_dir,
            )
        else:
            payload = update_artifacts_index_combined(
                artifacts_roots=roots,
                output_dir=args.output_dir or "_artifacts/index_combined",
            )
        counts = payload.get("counts") if isinstance(payload, dict) else None
        registries = payload.get("registries") if isinstance(payload, dict) else None

        if isinstance(counts, dict):
            print(json.dumps(counts, ensure_ascii=False, indent=2))
        if isinstance(registries, dict):
            print(json.dumps(registries, ensure_ascii=False, indent=2))
        warnings = payload.get("warnings") if isinstance(payload, dict) else None
        if isinstance(warnings, dict):
            nonempty = {k: v for k, v in warnings.items() if v}
            if nonempty:
                sys.stderr.write("index-artifacts warnings:\n")
                sys.stderr.write(json.dumps(nonempty, ensure_ascii=False, indent=2))
                sys.stderr.write("\n")
                return 1
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
