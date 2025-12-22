from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .render_html import render_compare_html, render_html
from .report_builder import RunLoadError, build_report, report_to_dict, diff_reports
from .report_model import RunInputs, RunReport


def _discover_artifact(logs_dir: str, generation_id: str, suffix: str) -> Optional[str]:
    candidate = os.path.join(logs_dir, f"{generation_id}{suffix}")
    return candidate if os.path.exists(candidate) else None


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
    parser = argparse.ArgumentParser(description="Offline run analyzer")
    parser.add_argument("--generation-id", dest="generation_id")
    parser.add_argument("--logs-dir", dest="logs_dir", default=".")
    parser.add_argument("--oplog", dest="oplog")
    parser.add_argument("--transcript", dest="transcript")
    parser.add_argument("--best-effort", action="store_true", dest="best_effort")
    parser.add_argument("--compare", nargs=2, dest="compare")
    parser.add_argument("--output-dir", dest="output_dir", default=".")
    args = parser.parse_args(argv)

    if args.compare:
        base_id, other_id = args.compare
        args.generation_id = base_id
        base_inputs = resolve_inputs(args)
        args.generation_id = other_id
        other_inputs = resolve_inputs(args)
        base_report = build_report(base_inputs, best_effort=args.best_effort)
        other_report = build_report(other_inputs, best_effort=args.best_effort)
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
        return 0

    inputs = resolve_inputs(args)
    report = build_report(inputs, best_effort=args.best_effort)
    _print_oplog_summary(report)
    json_path = os.path.join(args.output_dir, f"{inputs.generation_id}_run_report.json")
    html_path = os.path.join(args.output_dir, f"{inputs.generation_id}_run_report.html")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report_to_dict(report), handle, indent=2)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_html(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
