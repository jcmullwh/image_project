from __future__ import annotations

"""CLI for experiment discovery and execution.

This is invoked via the top-level CLI:

  python -m image_project experiments list
  python -m image_project experiments describe <name>
  python -m image_project experiments run <name> [runner opts...] [experiment opts...]
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from image_project.app.experiment_runner import run_experiment
from image_project.impl.current.experiments import ExperimentManager


def _print_err(message: str) -> None:
    """Write a message to stderr with a trailing newline."""

    sys.stderr.write(message.rstrip() + "\n")


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level `image_project experiments` argument parser."""

    parser = argparse.ArgumentParser(prog="image_project experiments", add_help=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available experiments")

    describe = sub.add_parser("describe", help="Describe an experiment")
    describe.add_argument("name", type=str, help="Experiment name (plugin id)")

    run = sub.add_parser("run", help="Run an experiment")
    run.add_argument("name", type=str, help="Experiment name (plugin id)")
    run.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional explicit pipeline config path (otherwise uses config/config.yaml + config/config.local.yaml, or IMAGE_PROJECT_CONFIG).",
    )
    run.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for experiment artifacts (logs/generated/upscaled). Defaults under ./_artifacts/experiments/.",
    )
    run.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment id written into transcript metadata (defaults to <name>_<timestamp>).",
    )
    run.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=("full", "prompt_only"),
        help="Run mode override (otherwise uses the loaded config's run.mode).",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and write plan artifacts without executing any runs.",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing experiment directory by skipping runs already marked success in logs/runs_index.jsonl.",
    )
    run.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep executing remaining runs if a run fails.",
    )
    run.add_argument(
        "--enable-upscale",
        action="store_true",
        help="Enable upscaling (uses the base config's upscale settings).",
    )
    run.add_argument(
        "--enable-upload",
        action="store_true",
        help="Enable rclone upload (uses the base config's rclone settings).",
    )
    run.add_argument(
        "--no-fail-on-config-error",
        action="store_true",
        help="Do not abort when experiment_plan_full.json contains config_error entries.",
    )

    return parser


def _build_plugin_parser(*, name: str) -> argparse.ArgumentParser:
    """Build an argparse parser for experiment-specific arguments."""

    exp = ExperimentManager.get(name)
    parser = argparse.ArgumentParser(
        prog=f"image_project experiments run {name}",
        add_help=True,
        description=getattr(exp, "summary", "") or None,
    )
    exp.add_cli_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the experiments CLI (returns a process exit code)."""

    args_list = list(argv) if argv is not None else None
    parser = _build_parser()

    try:
        args, unknown = parser.parse_known_args(args_list)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.command == "list":
        for name in ExperimentManager.available():
            exp = ExperimentManager.get(name)
            summary = getattr(exp, "summary", "") if isinstance(getattr(exp, "summary", None), str) else ""
            print(f"{name}\t{summary}".rstrip())
        return 0

    if args.command == "describe":
        name = str(args.name)
        exp = ExperimentManager.get(name)
        print(f"name: {getattr(exp, 'name', name)}")
        print(f"summary: {getattr(exp, 'summary', '').strip()}")
        print(f"default_run_mode: {getattr(exp, 'default_run_mode', '').strip()}")
        ref = f"{exp.__class__.__module__}.{exp.__class__.__name__}"
        print(f"plugin: {ref}")
        plugin_parser = _build_plugin_parser(name=name)
        print("")
        print(plugin_parser.format_help().rstrip())
        return 0

    if args.command == "run":
        name = str(args.name)
        exp = ExperimentManager.get(name)
        plugin_parser = _build_plugin_parser(name=name)
        try:
            plugin_ns = plugin_parser.parse_args(list(unknown))
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 2

        plugin_args: dict[str, Any] = vars(plugin_ns)
        if not isinstance(plugin_args, Mapping):
            plugin_args = dict(plugin_args)

        run_mode = str(args.mode) if getattr(args, "mode", None) is not None else getattr(exp, "default_run_mode", "")
        if args.mode is None:
            _print_err(
                f"experiments run: --mode not provided; defaulting to {run_mode} (override with --mode)"
            )

        try:
            return int(
                run_experiment(
                    experiment_name=name,
                    config_path=getattr(args, "config_path", None),
                    output_root=getattr(args, "output_root", None),
                    experiment_id=getattr(args, "experiment_id", None),
                    run_mode=run_mode,
                    dry_run=bool(getattr(args, "dry_run", False)),
                    resume=bool(getattr(args, "resume", False)),
                    continue_on_error=bool(getattr(args, "continue_on_error", False)),
                    enable_upscale=bool(getattr(args, "enable_upscale", False)),
                    enable_upload=bool(getattr(args, "enable_upload", False)),
                    fail_on_config_error=not bool(getattr(args, "no_fail_on_config_error", False)),
                    plugin_args=plugin_args,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _print_err(f"experiments run failed: {exc.__class__.__name__}: {exc}")
            return 2

    _print_err(f"Unhandled command: {args.command}")
    return 2
