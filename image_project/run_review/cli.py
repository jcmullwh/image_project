from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any, Optional

import yaml

from image_project.foundation.config_io import find_repo_root

from .evolution import thresholds_from_overrides
from .render_html import render_compare_html, render_html
from .compare_experiment import PairSelection, compare_experiment_pairs
from .report_builder import RunLoadError, build_report, report_to_dict, diff_reports
from .report_model import RunInputs, RunReport


def _discover_artifact(logs_dir: str, generation_id: str, suffix: str) -> Optional[str]:
    candidate = os.path.join(logs_dir, f"{generation_id}{suffix}")
    return candidate if os.path.exists(candidate) else None


def _pipeline_config_candidates(explicit_path: str | None) -> list[str]:
    if explicit_path:
        expanded = os.path.expandvars(os.path.expanduser(explicit_path.strip()))
        return [os.path.abspath(expanded)]

    repo_root: str | None = None
    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        repo_root = None

    candidates = [
        os.path.join(repo_root, "config", "config.yaml") if repo_root else None,
        os.path.join(os.getcwd(), "config", "config.yaml"),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _load_pipeline_paths(config_path: str) -> tuple[str | None, list[str]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid pipeline config (expected mapping): {config_path}")

    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(config_dir)

    def normalize_path(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"Invalid config value type for path (expected string): {value!r}")
        expanded = os.path.expandvars(os.path.expanduser(value.strip()))
        if not expanded:
            return None
        if not os.path.isabs(expanded):
            expanded = os.path.join(project_root, expanded)
        return os.path.abspath(expanded)

    image_cfg = payload.get("image")
    if not isinstance(image_cfg, Mapping):
        return None, []

    logs_dir = normalize_path(image_cfg.get("log_path"))
    generation_dir = normalize_path(image_cfg.get("generation_path") or image_cfg.get("save_path"))
    upscale_dir = normalize_path(image_cfg.get("upscale_path"))

    image_dirs = [d for d in (generation_dir, upscale_dir) if d]
    return logs_dir, image_dirs


def _resolve_default_locations(args, *, needs_logs_dir: bool, needs_image_dirs: bool) -> tuple[str | None, list[str]]:
    logs_dir = args.logs_dir if hasattr(args, "logs_dir") else None
    image_dirs = list(getattr(args, "images_dir", None) or [])

    if (needs_logs_dir and not logs_dir) or (needs_image_dirs and not image_dirs):
        missing: list[str] = []
        if needs_logs_dir and not logs_dir:
            missing.append("--logs-dir")
        if needs_image_dirs and not image_dirs:
            missing.append("--images-dir")

        candidates = _pipeline_config_candidates(getattr(args, "config_path", None))
        if getattr(args, "config_path", None):
            print(f"run_review: {', '.join(missing)} not set; loading pipeline config from {candidates[0]}")
        else:
            print(
                f"run_review: {', '.join(missing)} not set; searching for pipeline config at {', '.join(candidates)}"
            )

        config_path = next((p for p in candidates if os.path.exists(p)), None)
        if not config_path:
            if needs_logs_dir and not logs_dir:
                print("run_review: No pipeline config found; defaulting logs_dir='.' (set --logs-dir to override)")
                logs_dir = "."
            return logs_dir, image_dirs

        print(f"run_review: Using pipeline config {config_path}")
        cfg_logs_dir, cfg_image_dirs = _load_pipeline_paths(config_path)

        if needs_logs_dir and not logs_dir:
            if cfg_logs_dir:
                logs_dir = cfg_logs_dir
                print(f"run_review: Defaulted logs_dir from config image.log_path: {logs_dir}")
            else:
                logs_dir = "."
                print("run_review: Config missing image.log_path; defaulting logs_dir='.' (set --logs-dir to override)")

        if needs_image_dirs and not image_dirs and cfg_image_dirs:
            image_dirs = cfg_image_dirs
            print(f"run_review: Defaulted image dirs from config: {', '.join(image_dirs)}")

    return logs_dir, image_dirs


def _most_recent_generation_id(logs_dir: str, *, require_both: bool) -> tuple[str, str]:
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"logs_dir does not exist: {logs_dir}")

    suffixes = {
        "_transcript.json": "transcript",
        "_oplog.log": "oplog",
    }
    seen: dict[str, dict[str, float]] = {}

    with os.scandir(logs_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            for suffix, kind in suffixes.items():
                if not name.endswith(suffix):
                    continue
                generation_id = name[: -len(suffix)]
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                seen.setdefault(generation_id, {})[kind] = mtime
                break

    candidates: list[tuple[float, str, str]] = []
    for generation_id, kinds in seen.items():
        if require_both and not ("oplog" in kinds and "transcript" in kinds):
            continue
        most_recent_kind = max(kinds.items(), key=lambda item: item[1])[0]
        most_recent_ts = kinds[most_recent_kind]
        candidates.append((most_recent_ts, generation_id, most_recent_kind))

    if not candidates:
        requirement = "both oplog+transcript" if require_both else "any artifacts"
        raise FileNotFoundError(f"No runs found with {requirement} in logs_dir={logs_dir}")

    _, generation_id, kind = max(candidates, key=lambda item: item[0])
    basis = os.path.join(
        logs_dir,
        f"{generation_id}{'_transcript.json' if kind == 'transcript' else '_oplog.log'}",
    )
    return generation_id, basis


def _infer_generation_id(*, oplog_path: str | None, transcript_path: str | None) -> str:
    for path, suffix in (
        (transcript_path, "_transcript.json"),
        (oplog_path, "_oplog.log"),
    ):
        if not path:
            continue
        base = os.path.basename(path)
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return "unknown"


def resolve_inputs(args) -> RunInputs:
    if args.oplog or args.transcript:
        generation_id = args.generation_id or _infer_generation_id(
            oplog_path=args.oplog, transcript_path=args.transcript
        )
        if args.oplog and not os.path.exists(args.oplog):
            raise FileNotFoundError(args.oplog)
        if args.transcript and not os.path.exists(args.transcript):
            raise FileNotFoundError(args.transcript)
        if not args.best_effort and (not args.oplog or not args.transcript):
            missing = []
            if not args.oplog:
                missing.append("--oplog")
            if not args.transcript:
                missing.append("--transcript")
            raise ValueError(f"Missing {', '.join(missing)} (set --best-effort to continue)")
        return RunInputs(generation_id, oplog_path=args.oplog, transcript_path=args.transcript)

    if not args.generation_id:
        raise ValueError("generation_id is required when discovery is used")
    logs_dir = args.logs_dir or "."
    oplog = _discover_artifact(logs_dir, args.generation_id, "_oplog.log")
    transcript = _discover_artifact(logs_dir, args.generation_id, "_transcript.json")
    if not args.best_effort:
        if oplog is None:
            raise FileNotFoundError(f"Could not find oplog for {args.generation_id} in {logs_dir}")
        if transcript is None:
            raise FileNotFoundError(f"Could not find transcript for {args.generation_id} in {logs_dir}")
    return RunInputs(args.generation_id, oplog_path=oplog, transcript_path=transcript)


def _print_oplog_summary(report: RunReport) -> None:
    stats = report.metadata.oplog_stats or {}
    total = stats.get("total_lines")
    parsed = stats.get("parsed_lines")
    coverage = stats.get("coverage")
    unknown = stats.get("unknown_event_count")
    events = stats.get("event_count")
    fmt = stats.get("detected_format")
    parse_failed = any(i.code == "oplog_parse_failed" for i in report.issues)

    bits = [f"generation_id={report.metadata.generation_id}"]
    if fmt:
        bits.append(f"oplog_format={fmt}")
    if total is not None and parsed is not None and coverage is not None:
        bits.append(f"oplog_coverage={coverage:.1%} ({parsed}/{total})")
    if events is not None:
        bits.append(f"oplog_events={events}")
    if unknown is not None:
        bits.append(f"unknown_lines={unknown}")
    bits.append(f"oplog_parse_failed={str(parse_failed).lower()}")
    print("run_review:", " ".join(bits))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline run analyzer (defaults to --most-recent using pipeline config when no run is specified)"
    )
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        help="Pipeline config.yaml used for default log/image locations when flags are omitted (default: config/config.yaml).",
    )
    parser.add_argument("--generation-id", dest="generation_id")
    parser.add_argument(
        "--most-recent",
        "--most_recent",
        action="store_true",
        dest="most_recent",
        help="Select the most recently modified run in logs_dir (requires discovery mode).",
    )
    parser.add_argument(
        "--logs-dir",
        dest="logs_dir",
        default=None,
        help="Directory containing <id>_oplog.log and <id>_transcript.json. Defaults to image.log_path from pipeline config.",
    )
    parser.add_argument(
        "--images-dir",
        dest="images_dir",
        action="append",
        help="Optional directory to search for images when transcript metadata lacks image_path (repeatable). Defaults to image.generation_path + image.upscale_path from pipeline config.",
    )
    parser.add_argument("--oplog", dest="oplog")
    parser.add_argument("--transcript", dest="transcript")
    parser.add_argument("--best-effort", action="store_true", dest="best_effort")
    parser.add_argument("--no-evolution", action="store_true", dest="no_evolution")
    parser.add_argument("--evolution-thresholds", dest="evolution_thresholds", help="JSON overrides for evolution thresholds")
    parser.add_argument("--compare", nargs=2, dest="compare")
    parser.add_argument(
        "--compare-experiment",
        dest="compare_experiment",
        help="Compare A/B experiment pairs from <experiment_dir>/pairs.json",
    )
    parser.add_argument("--pair", dest="pair", type=int, help="Compare a single run_index from pairs.json")
    parser.add_argument("--all", action="store_true", dest="all_pairs", help="Compare all pairs in pairs.json")
    parser.add_argument("--output-dir", dest="output_dir", default=".")
    args = parser.parse_args(argv)

    evolution_thresholds = None
    if args.evolution_thresholds:
        try:
            overrides = json.loads(args.evolution_thresholds)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --evolution-thresholds JSON: {exc}") from exc
        evolution_thresholds = thresholds_from_overrides(overrides)

    if args.compare and args.most_recent:
        raise ValueError("--most-recent cannot be combined with --compare")
    if args.compare and (args.oplog or args.transcript):
        raise ValueError("--compare cannot be combined with --oplog/--transcript")
    if args.compare and args.compare_experiment:
        raise ValueError("--compare cannot be combined with --compare-experiment")
    if args.compare_experiment and args.most_recent:
        raise ValueError("--most-recent cannot be combined with --compare-experiment")
    if args.compare_experiment and (args.generation_id or args.oplog or args.transcript):
        raise ValueError(
            "--compare-experiment cannot be combined with --generation-id or --oplog/--transcript"
        )

    if args.compare_experiment:
        selection = PairSelection(all_pairs=bool(args.all_pairs), run_index=args.pair)
        summary = compare_experiment_pairs(
            experiment_dir=args.compare_experiment,
            output_dir=args.output_dir,
            selection=selection,
            logs_dir=args.logs_dir,
            best_effort=bool(args.best_effort),
            enable_evolution=not args.no_evolution,
            evolution_thresholds=evolution_thresholds,
            print_fn=print,
        )
        failed = int(((summary.get("counts") or {}).get("failed") or 0))
        return 0 if failed == 0 else 1

    if not args.compare:
        has_artifact_paths = bool(args.oplog or args.transcript)
        if args.most_recent and (args.generation_id or has_artifact_paths):
            raise ValueError("--most-recent cannot be combined with --generation-id or --oplog/--transcript")

        if not args.most_recent and not args.generation_id and not has_artifact_paths:
            args.most_recent = True
            print("run_review: No run selector provided; defaulting to --most-recent.")

    needs_logs_dir = bool(args.compare or args.most_recent or (not args.oplog and not args.transcript))
    needs_image_dirs = bool(needs_logs_dir and not args.logs_dir and not getattr(args, "images_dir", None))
    args.logs_dir, args.images_dir = _resolve_default_locations(
        args, needs_logs_dir=needs_logs_dir, needs_image_dirs=needs_image_dirs
    )

    if args.most_recent:
        assert args.logs_dir is not None
        generation_id, basis = _most_recent_generation_id(args.logs_dir, require_both=not args.best_effort)
        args.generation_id = generation_id
        print(f"run_review: Selected most recent generation_id={generation_id} (basis={basis})")

    if args.compare:
        base_id, other_id = args.compare
        args.generation_id = base_id
        base_inputs = resolve_inputs(args)
        args.generation_id = other_id
        other_inputs = resolve_inputs(args)
        base_report = build_report(
            base_inputs,
            best_effort=args.best_effort,
            enable_evolution=not args.no_evolution,
            evolution_thresholds=evolution_thresholds,
        )
        other_report = build_report(
            other_inputs,
            best_effort=args.best_effort,
            enable_evolution=not args.no_evolution,
            evolution_thresholds=evolution_thresholds,
        )
        _print_oplog_summary(base_report)
        _print_oplog_summary(other_report)
        diff = diff_reports(base_report, other_report)
        html_out = os.path.join(args.output_dir, f"{base_id}_vs_{other_id}_run_compare.html")
        json_out = os.path.join(args.output_dir, f"{base_id}_vs_{other_id}_run_compare.json")
        with open(html_out, "w", encoding="utf-8") as handle:
            handle.write(render_compare_html(diff))
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "added_steps": diff.added_steps,
                    "removed_steps": diff.removed_steps,
                    "metadata_changes": diff.metadata_changes,
                    "injector_diffs": diff.injector_diffs,
                    "post_processing_diffs": diff.post_processing_diffs,
                },
                handle,
                indent=2,
            )
        print(f"run_review: Wrote {html_out}")
        print(f"run_review: Wrote {json_out}")
        return 0

    inputs = resolve_inputs(args)
    report = build_report(
        inputs,
        best_effort=args.best_effort,
        enable_evolution=not args.no_evolution,
        evolution_thresholds=evolution_thresholds,
    )
    if report.metadata.image_path is None and not getattr(args, "images_dir", None):
        _, args.images_dir = _resolve_default_locations(args, needs_logs_dir=False, needs_image_dirs=True)

    if report.metadata.image_path is None and getattr(args, "images_dir", None):
        for dir_path in args.images_dir:
            for suffix in ("_image_4k.jpg", "_image.jpg"):
                candidate = os.path.join(dir_path, f"{inputs.generation_id}{suffix}")
                if os.path.exists(candidate):
                    report.metadata.image_path = candidate
                    report.metadata.artifact_paths.setdefault("image", candidate)
                    print(f"run_review: Inferred image_path from images_dir: {candidate}")
                    break
            if report.metadata.image_path is not None:
                break
    _print_oplog_summary(report)
    json_path = os.path.join(args.output_dir, f"{inputs.generation_id}_run_report.json")
    html_path = os.path.join(args.output_dir, f"{inputs.generation_id}_run_report.html")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report_to_dict(report), handle, indent=2)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_html(report))
    print(f"run_review: Wrote {html_path}")
    print(f"run_review: Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
