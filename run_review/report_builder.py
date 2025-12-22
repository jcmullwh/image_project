from __future__ import annotations

from collections import defaultdict
import dataclasses
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import TOOL_VERSION
from .parse_oplog import parse_oplog
from .parse_transcript import TranscriptParseError, parse_transcript
from .report_model import (
    CompareResult,
    Issue,
    OplogEvent,
    RunInputs,
    RunMetadata,
    RunReport,
    SideEffect,
    StepReport,
    StepTiming,
    TranscriptStep,
)

PARSER_VERSION = "1.0.0"
LARGE_CONTEXT_CHARS_THRESHOLD = 5_000
LARGE_INPUT_CHARS_THRESHOLD = 25_000


class RunLoadError(Exception):
    pass


def _step_name_from_path(path: str) -> str:
    if not path:
        return "unknown"
    return path.split("/")[-1]


def _merge_step_data(
    transcript_steps: List[TranscriptStep],
    oplog_events: List[OplogEvent],
    issues: List[Issue],
) -> Tuple[List[StepReport], List[str]]:
    start_by_path: Dict[str, List[OplogEvent]] = defaultdict(list)
    end_by_path: Dict[str, List[OplogEvent]] = defaultdict(list)

    first_event_index_by_occurrence: Dict[tuple[str, int], int] = {}
    start_occurrence_counter: Dict[str, int] = defaultdict(int)
    end_occurrence_counter: Dict[str, int] = defaultdict(int)

    for event_index, event in enumerate(oplog_events):
        if event.type == "step_start" and event.path:
            start_by_path[event.path].append(event)
            occurrence_index = start_occurrence_counter[event.path]
            start_occurrence_counter[event.path] += 1
            first_event_index_by_occurrence.setdefault((event.path, occurrence_index), event_index)
        if event.type == "step_end" and event.path:
            end_by_path[event.path].append(event)
            occurrence_index = end_occurrence_counter[event.path]
            end_occurrence_counter[event.path] += 1
            first_event_index_by_occurrence.setdefault((event.path, occurrence_index), event_index)

    transcript_paths = {s.path for s in transcript_steps if s.path}
    unmatched_oplog_paths: List[str] = []

    mismatch_paths: set[str] = set()
    for path in set(start_by_path) | set(end_by_path):
        if len(start_by_path.get(path, ())) != len(end_by_path.get(path, ())):
            mismatch_paths.add(path)

    def attach_oplog_timing(step: StepReport, start: OplogEvent | None, end: OplogEvent | None) -> None:
        timing = StepTiming()
        if start:
            timing.start_ts = start.timestamp
            step.oplog_prompt_chars = start.data.get("prompt_chars")
            step.oplog_input_chars = start.data.get("input_chars")
            step.oplog_context_chars = start.data.get("context_chars")
        if end:
            timing.end_ts = end.timestamp
            step.oplog_response_chars = end.data.get("response_chars") or end.data.get("chars")
            if end.data.get("duration_ms") is not None:
                timing.duration_ms = float(end.data["duration_ms"])
        if start and end and timing.duration_ms is None:
            timing.duration_ms = (end.timestamp - start.timestamp).total_seconds() * 1000
        step.timing = timing

        if start and not end:
            step.issues.append(Issue("warn", "step_missing_end", f"No end event for {step.path}", path=step.path))
        if end and not start:
            step.issues.append(Issue("warn", "step_missing_start", f"No start event for {step.path}", path=step.path))

    def add_common_step_issues(step: StepReport) -> None:
        if step.response is not None and not str(step.response).strip():
            step.issues.append(Issue("warn", "empty_response", f"Empty response for {step.path}", path=step.path))
        if (step.context_chars or 0) > LARGE_CONTEXT_CHARS_THRESHOLD:
            step.issues.append(
                Issue("info", "large_context", f"Large context_chars={step.context_chars}", path=step.path)
            )
        if (step.input_chars or 0) > LARGE_INPUT_CHARS_THRESHOLD:
            step.issues.append(Issue("info", "large_input", f"Large input_chars={step.input_chars}", path=step.path))

    # Transcript steps are authoritative for ordering.
    step_reports: List[StepReport] = []
    transcript_occurrence_counter: Dict[str, int] = defaultdict(int)
    transcript_steps_by_path: Dict[str, List[StepReport]] = defaultdict(list)

    for idx, t_step in enumerate(transcript_steps):
        path = t_step.path or "unknown"
        occurrence_index = transcript_occurrence_counter[path]
        transcript_occurrence_counter[path] += 1

        report = StepReport(
            path=path,
            name=t_step.name,
            step_index=idx,
            prompt=t_step.prompt,
            response=t_step.response,
            prompt_chars=t_step.prompt_chars,
            input_chars=t_step.input_chars,
            context_chars=t_step.context_chars,
            response_chars=t_step.response_chars,
        )

        start_list = start_by_path.get(path, [])
        end_list = end_by_path.get(path, [])
        start_event = start_list[occurrence_index] if occurrence_index < len(start_list) else None
        end_event = end_list[occurrence_index] if occurrence_index < len(end_list) else None
        attach_oplog_timing(report, start_event, end_event)

        if path in mismatch_paths:
            starts = len(start_list)
            ends = len(end_list)
            report.issues.append(
                Issue(
                    "warn",
                    "step_event_count_mismatch",
                    f"Oplog has {starts} step_start and {ends} step_end events for {path}",
                    path=path,
                )
            )

        if start_event is None and end_event is None and path in transcript_paths:
            report.issues.append(
                Issue("warn", "unmatched_transcript_step", f"Step present only in transcript: {path}", path=path)
            )
            issues.append(
                Issue("warn", "unmatched_transcript_step", f"Step present only in transcript: {path}", path=path)
            )

        add_common_step_issues(report)
        step_reports.append(report)
        transcript_steps_by_path[path].append(report)

    # Oplog-only steps (paths not present in transcript) are appended in oplog order.
    oplog_only_paths = sorted((set(start_by_path) | set(end_by_path)) - transcript_paths)
    occurrence_tuples: List[tuple[int, str, int]] = []
    for path in oplog_only_paths:
        starts = start_by_path.get(path, [])
        ends = end_by_path.get(path, [])
        occurrence_count = max(len(starts), len(ends))
        for occurrence_index in range(occurrence_count):
            first_index = first_event_index_by_occurrence.get((path, occurrence_index), 1_000_000_000)
            occurrence_tuples.append((first_index, path, occurrence_index))

    occurrence_tuples.sort(key=lambda t: (t[0], t[1], t[2]))

    next_index = len(step_reports)
    for _, path, occurrence_index in occurrence_tuples:
        start_list = start_by_path.get(path, [])
        end_list = end_by_path.get(path, [])
        start_event = start_list[occurrence_index] if occurrence_index < len(start_list) else None
        end_event = end_list[occurrence_index] if occurrence_index < len(end_list) else None

        report = StepReport(path=path, name=_step_name_from_path(path), step_index=next_index)
        next_index += 1

        attach_oplog_timing(report, start_event, end_event)

        if path in mismatch_paths:
            report.issues.append(
                Issue(
                    "warn",
                    "step_event_count_mismatch",
                    f"Oplog has {len(start_list)} step_start and {len(end_list)} step_end events for {path}",
                    path=path,
                )
            )

        report.issues.append(
            Issue("warn", "unmatched_oplog_step", f"Step appears in oplog but not transcript: {path}", path=path)
        )
        add_common_step_issues(report)
        step_reports.append(report)
        if path not in unmatched_oplog_paths:
            unmatched_oplog_paths.append(path)

    for path in sorted(mismatch_paths):
        issues.append(
            Issue(
                "warn",
                "step_event_count_mismatch",
                f"Oplog has {len(start_by_path.get(path, []))} step_start and {len(end_by_path.get(path, []))} step_end events for {path}",
                path=path,
            )
        )

    return step_reports, unmatched_oplog_paths


def _compute_oplog_header_stats(oplog_path: str) -> Dict[str, Any]:
    from .parse_oplog import TIMESTAMP_PIPE_RE, TIMESTAMP_SPACE_RE, _parse_timestamp

    total_lines = 0
    parsed_lines = 0
    pipe_lines = 0
    space_lines = 0

    with open(oplog_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            total_lines += 1
            ts, level, _message = _parse_timestamp(raw_line)
            if ts is None or level is None:
                continue
            parsed_lines += 1
            raw = raw_line.rstrip("\n")
            if TIMESTAMP_PIPE_RE.match(raw):
                pipe_lines += 1
            elif TIMESTAMP_SPACE_RE.match(raw):
                space_lines += 1

    unparsed_lines = total_lines - parsed_lines
    coverage = (parsed_lines / total_lines) if total_lines else 0.0
    if pipe_lines and not space_lines:
        detected_format = "pipe"
    elif space_lines and not pipe_lines:
        detected_format = "space"
    elif pipe_lines and space_lines:
        detected_format = "mixed"
    else:
        detected_format = "unknown"

    return {
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "unparsed_lines": unparsed_lines,
        "coverage": coverage,
        "detected_format": detected_format,
    }


def build_report(inputs: RunInputs, *, best_effort: bool = False) -> RunReport:
    issues: List[Issue] = []
    metadata: Optional[RunMetadata] = None
    transcript_steps: List[TranscriptStep] = []
    oplog_events: List[OplogEvent] = []
    unknown_events: List[str] = []
    side_effects: List[SideEffect] = []

    if inputs.transcript_path:
        try:
            metadata, transcript_steps = parse_transcript(inputs.transcript_path)
        except TranscriptParseError as exc:
            raise RunLoadError(str(exc)) from exc
    else:
        if not best_effort:
            raise RunLoadError("Transcript missing (set --best-effort to continue)")
        issues.append(Issue("warn", "missing_transcript", "Transcript missing", artifact_path=inputs.transcript_path))

    if inputs.oplog_path:
        oplog_events, unknown_events, side_effects = parse_oplog(inputs.oplog_path)
    else:
        if not best_effort:
            raise RunLoadError("Oplog missing (set --best-effort to continue)")
        issues.append(Issue("warn", "missing_oplog", "Oplog missing", artifact_path=inputs.oplog_path))

    if metadata is None:
        metadata = RunMetadata(inputs.generation_id or "unknown")
    metadata.artifact_paths.setdefault("oplog", inputs.oplog_path)
    metadata.artifact_paths.setdefault("transcript", inputs.transcript_path)

    if inputs.oplog_path:
        stats = _compute_oplog_header_stats(inputs.oplog_path)
        stats.update(
            {
                "event_count": len(oplog_events),
                "unknown_event_count": len(unknown_events),
                "side_effect_count": len(side_effects),
            }
        )
        metadata.oplog_stats = stats

        total = int(stats.get("total_lines", 0) or 0)
        parsed = int(stats.get("parsed_lines", 0) or 0)
        coverage = float(stats.get("coverage", 0.0) or 0.0)
        unparsed_ratio = (1.0 - coverage) if total else 0.0

        parse_failed = False
        if total and unparsed_ratio >= 0.95:
            parse_failed = True
            issues.append(
                Issue(
                    "error",
                    "oplog_parse_failed",
                    f"Oplog header parse failed: parsed {parsed}/{total} lines (coverage {coverage:.1%}).",
                    artifact_path=inputs.oplog_path,
                )
            )
        if total and not parse_failed and parsed and len(oplog_events) == 0:
            parse_failed = True
            issues.append(
                Issue(
                    "warn",
                    "oplog_parse_failed",
                    f"Oplog semantic parse produced 0 events ({parsed}/{total} headers parsed). Log formats may have changed.",
                    artifact_path=inputs.oplog_path,
                )
            )
        if total and not parse_failed and coverage < 0.9:
            issues.append(
                Issue(
                    "warn",
                    "oplog_low_coverage",
                    f"Oplog header parse coverage low: parsed {parsed}/{total} lines (coverage {coverage:.1%}).",
                    artifact_path=inputs.oplog_path,
                )
            )

    if not transcript_steps and inputs.transcript_path:
        issues.append(Issue("warn", "empty_transcript_steps", "Transcript contains no steps", artifact_path=inputs.transcript_path))

    seed_event = next((e for e in oplog_events if e.type == "seed_selection"), None)
    if metadata.seed is None and seed_event:
        metadata.seed = seed_event.data.get("seed")

    config_defaults = [e for e in oplog_events if e.type == "config_default"]
    for event in config_defaults:
        severity = "warn" if event.level.upper() in {"WARNING", "WARN"} else "info"
        issues.append(Issue(severity, "config_default", event.message, artifact_path=inputs.oplog_path))

    context_injectors = [e for e in oplog_events if e.type == "context_injector"]
    for event in context_injectors:
        detail = event.data.get("detail")
        if isinstance(event.data.get("injectors"), list):
            detail = f"enabled={event.data.get('injectors')}"
        issues.append(Issue("info", "context_injector", detail or event.message, artifact_path=inputs.oplog_path))

    if metadata.concept_filter_log:
        filt = metadata.concept_filter_log
        if isinstance(filt, dict) and filt.get("input") == filt.get("output"):
            issues.append(Issue("info", "concept_filter_noop", "Concept filter output matches input", artifact_path=inputs.transcript_path))

    steps, unmatched_oplog_paths = _merge_step_data(transcript_steps, oplog_events, issues)

    run_start_events = [e for e in oplog_events if e.type == "run_start"]
    run_end_events = [e for e in oplog_events if e.type == "run_end"]
    run_start_ts = min((e.timestamp for e in run_start_events), default=None)
    run_end_ts = max((e.timestamp for e in run_end_events), default=None)
    runtime_ms: float | None = None
    if run_start_ts and run_end_ts:
        runtime_ms = (run_end_ts - run_start_ts).total_seconds() * 1000

    if unknown_events and inputs.oplog_path:
        issues.append(
            Issue(
                "warn",
                "unknown_oplog_lines",
                f"Unrecognized oplog lines: {len(unknown_events)} (see unknown_events)",
                artifact_path=inputs.oplog_path,
            )
        )

    run_report = RunReport(
        metadata=metadata,
        steps=steps,
        side_effects=side_effects,
        issues=issues,
        parser_version=PARSER_VERSION,
        tool_version=TOOL_VERSION,
        run_start_ts=run_start_ts,
        run_end_ts=run_end_ts,
        runtime_ms=runtime_ms,
        unknown_events=unknown_events,
    )
    return run_report


def report_to_dict(report: RunReport) -> Dict:
    def serialize_issue(issue: Issue) -> Dict:
        return {
            "severity": issue.severity,
            "code": issue.code,
            "message": issue.message,
            "path": issue.path,
            "artifact_path": issue.artifact_path,
        }

    def serialize_step(step: StepReport) -> Dict:
        return {
            "step_index": step.step_index,
            "path": step.path,
            "name": step.name,
            "prompt": step.prompt,
            "response": step.response,
            "prompt_chars": step.prompt_chars,
            "input_chars": step.input_chars,
            "context_chars": step.context_chars,
            "response_chars": step.response_chars,
            "oplog_prompt_chars": step.oplog_prompt_chars,
            "oplog_input_chars": step.oplog_input_chars,
            "oplog_context_chars": step.oplog_context_chars,
            "oplog_response_chars": step.oplog_response_chars,
            "timing": {
                "start_ts": step.timing.start_ts.isoformat() if step.timing.start_ts else None,
                "end_ts": step.timing.end_ts.isoformat() if step.timing.end_ts else None,
                "duration_ms": step.timing.duration_ms,
            },
            "issues": [serialize_issue(i) for i in step.issues],
        }

    def serialize_side_effect(effect: SideEffect) -> Dict:
        return {
            "type": effect.type,
            "timestamp": effect.timestamp.isoformat(),
            "data": effect.data,
            "raw": effect.raw,
        }

    return {
        "metadata": dataclasses.asdict(report.metadata),
        "steps": [serialize_step(s) for s in report.steps],
        "side_effects": [serialize_side_effect(s) for s in report.side_effects],
        "issues": [serialize_issue(i) for i in report.issues],
        "parser_version": report.parser_version,
        "tool_version": report.tool_version,
        "run_timing": {
            "start_ts": report.run_start_ts.isoformat() if report.run_start_ts else None,
            "end_ts": report.run_end_ts.isoformat() if report.run_end_ts else None,
            "runtime_ms": report.runtime_ms,
        },
        "unknown_events": report.unknown_events,
    }


def diff_reports(run_a: RunReport, run_b: RunReport) -> CompareResult:
    paths_a = {s.path for s in run_a.steps}
    paths_b = {s.path for s in run_b.steps}
    added = sorted(paths_b - paths_a)
    removed = sorted(paths_a - paths_b)

    metadata_changes: Dict[str, Dict[str, Optional[str]]] = {}
    interesting_keys = ["context", "title_generation", "concept_filter_log"]
    for key in interesting_keys:
        val_a = getattr(run_a.metadata, key)
        val_b = getattr(run_b.metadata, key)
        if bool(val_a) != bool(val_b):
            metadata_changes[key] = {"run_a": val_a, "run_b": val_b}

    injector_diffs: List[str] = []

    injector_msgs_a = sorted({i.message for i in run_a.issues if i.code == "context_injector"})
    injector_msgs_b = sorted({i.message for i in run_b.issues if i.code == "context_injector"})

    if injector_msgs_a != injector_msgs_b:
        added_msgs = sorted(set(injector_msgs_b) - set(injector_msgs_a))
        removed_msgs = sorted(set(injector_msgs_a) - set(injector_msgs_b))
        if added_msgs:
            injector_diffs.append(f"Injector logs added: {added_msgs[0]}" + (" (and more)" if len(added_msgs) > 1 else ""))
        if removed_msgs:
            injector_diffs.append(f"Injector logs removed: {removed_msgs[0]}" + (" (and more)" if len(removed_msgs) > 1 else ""))

    def adoption_strength(messages: List[str]) -> str | None:
        joined = "\n".join(messages).lower()
        if "must adopt" in joined:
            return "must"
        if "should adopt" in joined:
            return "should"
        return None

    strength_a = adoption_strength(injector_msgs_a)
    strength_b = adoption_strength(injector_msgs_b)
    if strength_a != strength_b and (strength_a or strength_b):
        injector_diffs.append(f"Injector adoption strength changed: {strength_a or 'none'} -> {strength_b or 'none'}")

    post_processing_diffs: List[str] = []
    upscale_formats_a = sorted({se.data.get("format") for se in run_a.side_effects if se.type == "upscale"} - {None})
    upscale_formats_b = sorted({se.data.get("format") for se in run_b.side_effects if se.type == "upscale"} - {None})
    if upscale_formats_a != upscale_formats_b:
        post_processing_diffs.append(
            f"Upscale log format changed: {upscale_formats_a or ['none']} -> {upscale_formats_b or ['none']}"
        )

    return CompareResult(
        run_a,
        run_b,
        added,
        removed,
        metadata_changes,
        injector_diffs,
        post_processing_diffs=post_processing_diffs,
    )
from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from . import TOOL_VERSION
from .parse_oplog import parse_oplog
from .parse_transcript import TranscriptParseError, parse_transcript
from .report_model import (
    CompareResult,
    Issue,
    OplogEvent,
    RunInputs,
    RunMetadata,
    RunReport,
    SideEffect,
    StepReport,
    StepTiming,
    TranscriptStep,
)

PARSER_VERSION = "1.0.0"
LARGE_CONTEXT_CHARS_THRESHOLD = 5_000
LARGE_INPUT_CHARS_THRESHOLD = 25_000


class RunLoadError(Exception):
    pass


def _step_name_from_path(path: str) -> str:
    if not path:
        return "unknown"
    return path.split("/")[-1]


def _merge_step_data(
    transcript_steps: List[TranscriptStep],
    oplog_events: List[OplogEvent],
    issues: List[Issue],
) -> Tuple[List[StepReport], List[str]]:
    steps_by_path: Dict[str, StepReport] = {}
    unmatched_oplog_paths: List[str] = []
    transcript_paths = {s.path for s in transcript_steps}

    for step in transcript_steps:
        steps_by_path[step.path] = StepReport(
            path=step.path,
            name=step.name,
            prompt=step.prompt,
            response=step.response,
            prompt_chars=step.prompt_chars,
            input_chars=step.input_chars,
            context_chars=step.context_chars,
            response_chars=step.response_chars,
        )

    start_events: Dict[str, OplogEvent] = {}
    end_events: Dict[str, OplogEvent] = {}

    for event in oplog_events:
        if event.type == "step_start" and event.path:
            start_events[event.path] = event
            if event.path not in steps_by_path:
                steps_by_path[event.path] = StepReport(path=event.path, name=_step_name_from_path(event.path))
        if event.type == "step_end" and event.path:
            end_events[event.path] = event
            if event.path not in steps_by_path:
                steps_by_path[event.path] = StepReport(path=event.path, name=_step_name_from_path(event.path))

    for path, step in steps_by_path.items():
        start = start_events.get(path)
        end = end_events.get(path)
        timing = StepTiming()
        if start:
            timing.start_ts = start.timestamp
            step.oplog_prompt_chars = start.data.get("prompt_chars")
            step.oplog_input_chars = start.data.get("input_chars")
            step.oplog_context_chars = start.data.get("context_chars")
        if end:
            timing.end_ts = end.timestamp
            step.oplog_response_chars = end.data.get("response_chars")
            if end.data.get("duration_ms") is not None:
                timing.duration_ms = end.data["duration_ms"]
        if start and end and timing.duration_ms is None:
            timing.duration_ms = (end.timestamp - start.timestamp).total_seconds() * 1000
        step.timing = timing

        if start and not end:
            step.issues.append(Issue("warn", "step_missing_end", f"No end event for {path}", path=path))
        if end and not start:
            step.issues.append(Issue("warn", "step_missing_start", f"No start event for {path}", path=path))
        if step.response is not None and not str(step.response).strip():
            step.issues.append(Issue("warn", "empty_response", f"Empty response for {path}", path=path))
        if (step.context_chars or 0) > LARGE_CONTEXT_CHARS_THRESHOLD:
            step.issues.append(
                Issue("info", "large_context", f"Large context_chars={step.context_chars}", path=path)
            )
        if (step.input_chars or 0) > LARGE_INPUT_CHARS_THRESHOLD:
            step.issues.append(Issue("info", "large_input", f"Large input_chars={step.input_chars}", path=path))

        if path not in transcript_paths:
            step.issues.append(Issue("warn", "unmatched_oplog_step", f"Step appears in oplog but not transcript: {path}", path=path))
            unmatched_oplog_paths.append(path)

    for path in transcript_paths:
        if path not in start_events and path not in end_events:
            steps_by_path[path].issues.append(
                Issue("warn", "unmatched_transcript_step", f"Step present only in transcript: {path}", path=path)
            )
            issues.append(
                Issue("warn", "unmatched_transcript_step", f"Step present only in transcript: {path}", path=path)
            )

    ordered_steps = sorted(steps_by_path.values(), key=lambda s: (s.timing.start_ts or datetime.min, s.path))
    return ordered_steps, unmatched_oplog_paths


def build_report(inputs: RunInputs, *, best_effort: bool = False) -> RunReport:
    issues: List[Issue] = []
    metadata: Optional[RunMetadata] = None
    transcript_steps: List[TranscriptStep] = []
    oplog_events: List[OplogEvent] = []
    unknown_events: List[str] = []
    side_effects: List[SideEffect] = []

    if inputs.transcript_path:
        try:
            metadata, transcript_steps = parse_transcript(inputs.transcript_path)
        except TranscriptParseError as exc:
            raise RunLoadError(str(exc)) from exc
    else:
        if not best_effort:
            raise RunLoadError("Transcript missing (set --best-effort to continue)")
        issues.append(Issue("warn", "missing_transcript", "Transcript missing", artifact_path=inputs.transcript_path))

    if inputs.oplog_path:
        oplog_events, unknown_events, side_effects = parse_oplog(inputs.oplog_path)
    else:
        if not best_effort:
            raise RunLoadError("Oplog missing (set --best-effort to continue)")
        issues.append(Issue("warn", "missing_oplog", "Oplog missing", artifact_path=inputs.oplog_path))

    if metadata is None:
        metadata = RunMetadata(inputs.generation_id or "unknown")
    metadata.artifact_paths.setdefault("oplog", inputs.oplog_path)
    metadata.artifact_paths.setdefault("transcript", inputs.transcript_path)

    if not transcript_steps and inputs.transcript_path:
        issues.append(Issue("warn", "empty_transcript_steps", "Transcript contains no steps", artifact_path=inputs.transcript_path))

    seed_event = next((e for e in oplog_events if e.type == "seed_selection"), None)
    if metadata.seed is None and seed_event:
        metadata.seed = seed_event.data.get("seed")

    config_defaults = [e for e in oplog_events if e.type == "config_default"]
    for event in config_defaults:
        severity = "warn" if event.level.upper() in {"WARNING", "WARN"} else "info"
        issues.append(Issue(severity, "config_default", event.message, artifact_path=inputs.oplog_path))

    context_injectors = [e for e in oplog_events if e.type == "context_injector"]
    for event in context_injectors:
        detail = event.data.get("detail")
        if isinstance(event.data.get("injectors"), list):
            detail = f"enabled={event.data.get('injectors')}"
        issues.append(Issue("info", "context_injector", detail or event.message, artifact_path=inputs.oplog_path))

    if metadata.concept_filter_log:
        filt = metadata.concept_filter_log
        if isinstance(filt, dict) and filt.get("input") == filt.get("output"):
            issues.append(Issue("info", "concept_filter_noop", "Concept filter output matches input", artifact_path=inputs.transcript_path))

    steps, unmatched_oplog_paths = _merge_step_data(transcript_steps, oplog_events, issues)

    run_start_events = [e for e in oplog_events if e.type == "run_start"]
    run_end_events = [e for e in oplog_events if e.type == "run_end"]
    run_start_ts = min((e.timestamp for e in run_start_events), default=None)
    run_end_ts = max((e.timestamp for e in run_end_events), default=None)
    runtime_ms: float | None = None
    if run_start_ts and run_end_ts:
        runtime_ms = (run_end_ts - run_start_ts).total_seconds() * 1000

    if unknown_events and inputs.oplog_path:
        issues.append(
            Issue(
                "warn",
                "unknown_oplog_lines",
                f"Unrecognized oplog lines: {len(unknown_events)} (see unknown_events)",
                artifact_path=inputs.oplog_path,
            )
        )

    run_report = RunReport(
        metadata=metadata,
        steps=steps,
        side_effects=side_effects,
        issues=issues,
        parser_version=PARSER_VERSION,
        tool_version=TOOL_VERSION,
        run_start_ts=run_start_ts,
        run_end_ts=run_end_ts,
        runtime_ms=runtime_ms,
        unknown_events=unknown_events,
    )
    return run_report


def report_to_dict(report: RunReport) -> Dict:
    def serialize_issue(issue: Issue) -> Dict:
        return {
            "severity": issue.severity,
            "code": issue.code,
            "message": issue.message,
            "path": issue.path,
            "artifact_path": issue.artifact_path,
        }

    def serialize_step(step: StepReport) -> Dict:
        return {
            "path": step.path,
            "name": step.name,
            "prompt": step.prompt,
            "response": step.response,
            "prompt_chars": step.prompt_chars,
            "input_chars": step.input_chars,
            "context_chars": step.context_chars,
            "response_chars": step.response_chars,
            "oplog_prompt_chars": step.oplog_prompt_chars,
            "oplog_input_chars": step.oplog_input_chars,
            "oplog_context_chars": step.oplog_context_chars,
            "oplog_response_chars": step.oplog_response_chars,
            "timing": {
                "start_ts": step.timing.start_ts.isoformat() if step.timing.start_ts else None,
                "end_ts": step.timing.end_ts.isoformat() if step.timing.end_ts else None,
                "duration_ms": step.timing.duration_ms,
            },
            "issues": [serialize_issue(i) for i in step.issues],
        }

    def serialize_side_effect(effect: SideEffect) -> Dict:
        return {
            "type": effect.type,
            "timestamp": effect.timestamp.isoformat(),
            "data": effect.data,
            "raw": effect.raw,
        }

    return {
        "metadata": dataclasses.asdict(report.metadata),
        "steps": [serialize_step(s) for s in report.steps],
        "side_effects": [serialize_side_effect(s) for s in report.side_effects],
        "issues": [serialize_issue(i) for i in report.issues],
        "parser_version": report.parser_version,
        "tool_version": report.tool_version,
        "run_timing": {
            "start_ts": report.run_start_ts.isoformat() if report.run_start_ts else None,
            "end_ts": report.run_end_ts.isoformat() if report.run_end_ts else None,
            "runtime_ms": report.runtime_ms,
        },
        "unknown_events": report.unknown_events,
    }


def diff_reports(run_a: RunReport, run_b: RunReport) -> CompareResult:
    paths_a = {s.path for s in run_a.steps}
    paths_b = {s.path for s in run_b.steps}
    added = sorted(paths_b - paths_a)
    removed = sorted(paths_a - paths_b)

    metadata_changes: Dict[str, Dict[str, Optional[str]]] = {}
    interesting_keys = ["context", "title_generation", "concept_filter_log"]
    for key in interesting_keys:
        val_a = getattr(run_a.metadata, key)
        val_b = getattr(run_b.metadata, key)
        if bool(val_a) != bool(val_b):
            metadata_changes[key] = {"run_a": val_a, "run_b": val_b}

    injector_diffs: List[str] = []

    injector_msgs_a = sorted({i.message for i in run_a.issues if i.code == "context_injector"})
    injector_msgs_b = sorted({i.message for i in run_b.issues if i.code == "context_injector"})

    if injector_msgs_a != injector_msgs_b:
        added_msgs = sorted(set(injector_msgs_b) - set(injector_msgs_a))
        removed_msgs = sorted(set(injector_msgs_a) - set(injector_msgs_b))
        if added_msgs:
            injector_diffs.append(f"Injector logs added: {added_msgs[0]}" + (" (and more)" if len(added_msgs) > 1 else ""))
        if removed_msgs:
            injector_diffs.append(f"Injector logs removed: {removed_msgs[0]}" + (" (and more)" if len(removed_msgs) > 1 else ""))

    def adoption_strength(messages: List[str]) -> str | None:
        joined = "\n".join(messages).lower()
        if "must adopt" in joined:
            return "must"
        if "should adopt" in joined:
            return "should"
        return None

    strength_a = adoption_strength(injector_msgs_a)
    strength_b = adoption_strength(injector_msgs_b)
    if strength_a != strength_b and (strength_a or strength_b):
        injector_diffs.append(f"Injector adoption strength changed: {strength_a or 'none'} -> {strength_b or 'none'}")

    post_processing_diffs: List[str] = []
    upscale_formats_a = sorted({se.data.get("format") for se in run_a.side_effects if se.type == "upscale"} - {None})
    upscale_formats_b = sorted({se.data.get("format") for se in run_b.side_effects if se.type == "upscale"} - {None})
    if upscale_formats_a != upscale_formats_b:
        post_processing_diffs.append(
            f"Upscale log format changed: {upscale_formats_a or ['none']} -> {upscale_formats_b or ['none']}"
        )

    return CompareResult(
        run_a,
        run_b,
        added,
        removed,
        metadata_changes,
        injector_diffs,
        post_processing_diffs=post_processing_diffs,
    )
