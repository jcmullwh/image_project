from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .report_model import OplogEvent, SideEffect

TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+(?P<level>\w+)\s+(?P<msg>.*)$"
)

RUN_START_FOR_GENERATION_RE = re.compile(r"^Run started for generation\s+(?P<id>\S+)", re.IGNORECASE)
RUN_START_KV_RE = re.compile(r"^Run started.*generation_id=(?P<id>\S+)", re.IGNORECASE)
RUN_END_SUCCESS_RE = re.compile(r"^Run completed successfully for generation\s+(?P<id>\S+)", re.IGNORECASE)
RUN_END_GENERIC_RE = re.compile(r"^Run (completed|ended)\b", re.IGNORECASE)

SEED_GENERATED_RE = re.compile(r"generated seed=(?P<seed>\d+)", re.IGNORECASE)
SEED_CONFIGURED_RE = re.compile(r"Using configured prompt\.random_seed=(?P<seed>\d+)", re.IGNORECASE)
SEED_SELECTED_RE = re.compile(r"Seed (selected|chosen):\s*(?P<seed>\d+)", re.IGNORECASE)

CONFIG_DEFAULT_RE = re.compile(r"\bConfig\b.*\bdefault", re.IGNORECASE)

CONTEXT_INJECTORS_ENABLED_RE = re.compile(r"^Context injectors enabled:\s*(?P<injectors>.*)$", re.IGNORECASE)
INJECTOR_LINE_RE = re.compile(r"^(?P<name>[\w-]+)\s+injector:\s*(?P<detail>.*)$", re.IGNORECASE)

STEP_START_PARENS_RE = re.compile(r"^Step:\s+(?P<path>\S+)\s*\((?P<metrics>[^)]*)\)\s*$", re.IGNORECASE)
STEP_START_INLINE_RE = re.compile(r"^Step start:?\s+(?P<path>\S+)(?P<rest>.*)$", re.IGNORECASE)

STEP_END_PARENS_RE = re.compile(
    r"^Received response for\s+(?P<path>\S+)\s*\((?P<metrics>[^)]*)\)\s*$", re.IGNORECASE
)
STEP_END_INLINE_RE = re.compile(
    r"^(?:Step end:?|Received response for)\s+(?P<path>\S+)(?P<rest>.*)$",
    re.IGNORECASE,
)
STEP_ERROR_RE = re.compile(r"^Step failed:\s+(?P<path>\S+)\b", re.IGNORECASE)

IMAGE_REQUEST_SENT_RE = re.compile(r"^Image generation request sent\s*\((?P<metrics>.*)\)\s*$", re.IGNORECASE)
IMAGE_REQUEST_INLINE_RE = re.compile(r"^image generation request(?P<detail>.*)$", re.IGNORECASE)
IMAGE_PAYLOAD_LEN_RE = re.compile(r"^Received image payload length:\s*(?P<len>\d+)\s*$", re.IGNORECASE)

UPSCALE_OLD_RE = re.compile(r"upscaling enabled .*target_long_edge_px=(?P<len>\d+)", re.IGNORECASE)
UPSCALE_NEW_RE = re.compile(r"upscaling .*target=.*?(?P<len>\d+)", re.IGNORECASE)
UPSCALE_ENABLED_RE = re.compile(r"^Upscaling enabled:\s*(?P<metrics>.*)$", re.IGNORECASE)

MANIFEST_APPEND_RE = re.compile(
    r"^Appended manifest row to\s+(?P<path>.+?)(?:\s*\(seq=(?P<seq>\d+)\))?\s*$",
    re.IGNORECASE,
)
GENERATION_APPEND_RE = re.compile(r"^Appended generation row to\s+(?P<path>.+)\s*$", re.IGNORECASE)

UPLOAD_BEGIN_RE = re.compile(r"^Uploading\s+(?P<file>.+?)\s+to\s+(?P<dest>.+?)\s+via rclone\s*$", re.IGNORECASE)
UPLOAD_COMPLETE_RE = re.compile(r"^Uploaded image via rclone to\s+(?P<dest>.+)\s*$", re.IGNORECASE)
UPLOAD_SKIPPED_RE = re.compile(r"^Upload skipped:\s+(?P<detail>.*)$", re.IGNORECASE)
UPLOAD_FAILED_RE = re.compile(r"^Upload failed\b(?P<detail>.*)$", re.IGNORECASE)
RCLONE_FAILED_RE = re.compile(r"^Rclone upload failed\b(?P<detail>.*)$", re.IGNORECASE)

FILE_WRITE_TO_RE = re.compile(r"^(?P<verb>Saved|Wrote)\s+(?P<what>.+?)\s+to\s+(?P<path>.+)\s*$", re.IGNORECASE)
OPLOG_STORED_RE = re.compile(r"^Operational log stored at\s+(?P<path>.+)\s*$", re.IGNORECASE)

KV_TOKEN_RE = re.compile(r"(?P<key>[\w.]+)=(?P<value>[^\s,]+)")


def _parse_timestamp(line: str) -> tuple[datetime | None, str | None, str]:
    match = TIMESTAMP_RE.match(line.rstrip("\n"))
    if not match:
        return None, None, line.strip()
    ts_raw = match.group("ts")
    level = match.group("level")
    message = match.group("msg").strip()
    try:
        ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None, level, line.strip()
    return ts, level, message


def _parse_kv_metrics(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for match in KV_TOKEN_RE.finditer(text):
        key = match.group("key")
        raw_value = match.group("value")
        if re.fullmatch(r"[-+]?\d+", raw_value):
            value: Any = int(raw_value)
        elif re.fullmatch(r"[-+]?\d+\.\d+", raw_value):
            value = float(raw_value)
        else:
            value = raw_value
        metrics[key] = value
    return metrics


def parse_oplog(path: str) -> Tuple[List[OplogEvent], List[str], List[SideEffect]]:
    events: List[OplogEvent] = []
    unknown: List[str] = []
    side_effects: List[SideEffect] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            ts, level, message = _parse_timestamp(raw_line)
            if ts is None or level is None:
                unknown.append(raw_line.rstrip("\n"))
                continue

            raw = raw_line.rstrip("\n")

            if m := RUN_START_FOR_GENERATION_RE.match(message):
                generation_id = m.group("id")
                events.append(OplogEvent(ts, level, "run_start", message, data={"generation_id": generation_id}, raw=raw))
                continue
            if m := RUN_START_KV_RE.match(message):
                generation_id = m.group("id")
                events.append(OplogEvent(ts, level, "run_start", message, data={"generation_id": generation_id}, raw=raw))
                continue
            if m := RUN_END_SUCCESS_RE.match(message):
                generation_id = m.group("id")
                events.append(OplogEvent(ts, level, "run_end", message, data={"generation_id": generation_id}, raw=raw))
                continue
            if RUN_END_GENERIC_RE.match(message):
                events.append(OplogEvent(ts, level, "run_end", message, raw=raw))
                continue

            if m := SEED_GENERATED_RE.search(message):
                events.append(
                    OplogEvent(
                        ts,
                        level,
                        "seed_selection",
                        message,
                        data={"seed": int(m.group("seed")), "source": "generated"},
                        raw=raw,
                    )
                )
                continue
            if m := SEED_CONFIGURED_RE.search(message):
                events.append(
                    OplogEvent(
                        ts,
                        level,
                        "seed_selection",
                        message,
                        data={"seed": int(m.group("seed")), "source": "configured"},
                        raw=raw,
                    )
                )
                continue
            if m := SEED_SELECTED_RE.search(message):
                events.append(
                    OplogEvent(ts, level, "seed_selection", message, data={"seed": int(m.group("seed"))}, raw=raw)
                )
                continue

            if CONFIG_DEFAULT_RE.search(message):
                events.append(OplogEvent(ts, level, "config_default", message, raw=raw))
                continue

            if m := CONTEXT_INJECTORS_ENABLED_RE.match(message):
                injectors_raw = m.group("injectors").strip()
                injectors = [s.strip() for s in injectors_raw.split(",") if s.strip()] if injectors_raw else []
                events.append(
                    OplogEvent(
                        ts,
                        level,
                        "context_injector",
                        message,
                        data={"injectors": injectors},
                        raw=raw,
                    )
                )
                continue
            if m := INJECTOR_LINE_RE.match(message):
                injector = m.group("name").strip().lower()
                detail = m.group("detail").strip()
                events.append(
                    OplogEvent(
                        ts,
                        level,
                        "context_injector",
                        message,
                        data={"injector": injector, "detail": detail},
                        raw=raw,
                    )
                )
                continue

            if m := STEP_START_PARENS_RE.match(message):
                path_val = m.group("path")
                metrics = _parse_kv_metrics(m.group("metrics"))
                events.append(OplogEvent(ts, level, "step_start", message, path=path_val, data=metrics, raw=raw))
                continue
            if m := STEP_START_INLINE_RE.match(message):
                path_val = m.group("path")
                metrics = _parse_kv_metrics(m.group("rest") or "")
                events.append(OplogEvent(ts, level, "step_start", message, path=path_val, data=metrics, raw=raw))
                continue

            if m := STEP_END_PARENS_RE.match(message):
                path_val = m.group("path")
                metrics = _parse_kv_metrics(m.group("metrics"))
                if "chars" in metrics and "response_chars" not in metrics:
                    metrics["response_chars"] = metrics["chars"]
                events.append(OplogEvent(ts, level, "step_end", message, path=path_val, data=metrics, raw=raw))
                continue
            if m := STEP_END_INLINE_RE.match(message):
                path_val = m.group("path")
                metrics = _parse_kv_metrics(m.group("rest") or "")
                if "chars" in metrics and "response_chars" not in metrics:
                    metrics["response_chars"] = metrics["chars"]
                events.append(OplogEvent(ts, level, "step_end", message, path=path_val, data=metrics, raw=raw))
                continue
            if m := STEP_ERROR_RE.match(message):
                path_val = m.group("path")
                events.append(OplogEvent(ts, level, "step_error", message, path=path_val, raw=raw))
                continue

            if m := IMAGE_REQUEST_SENT_RE.match(message):
                metrics = _parse_kv_metrics(m.group("metrics"))
                events.append(OplogEvent(ts, level, "image_generation_request", message, data=metrics, raw=raw))
                side_effects.append(SideEffect("image_generation_request", ts, metrics, raw=raw))
                continue
            if m := IMAGE_REQUEST_INLINE_RE.match(message):
                detail = m.group("detail").strip()
                event = OplogEvent(ts, level, "image_generation_request", message, data={"detail": detail}, raw=raw)
                events.append(event)
                side_effects.append(SideEffect("image_generation_request", ts, {"detail": detail}, raw=raw))
                continue
            if m := IMAGE_PAYLOAD_LEN_RE.match(message):
                data = {"payload_length": int(m.group("len"))}
                events.append(OplogEvent(ts, level, "image_generation_payload", message, data=data, raw=raw))
                side_effects.append(SideEffect("image_generation_payload", ts, data, raw=raw))
                continue

            if m := UPSCALE_ENABLED_RE.match(message):
                metrics = _parse_kv_metrics(m.group("metrics"))
                target_desc = m.group("metrics")
                length_match = UPSCALE_OLD_RE.search(message) or UPSCALE_NEW_RE.search(message)
                length_val = int(length_match.group("len")) if length_match and length_match.group("len") else None
                metrics.setdefault("target_long_edge_px", length_val)
                metrics["format"] = "new" if "target=" in target_desc else "unknown"
                events.append(OplogEvent(ts, level, "upscale", message, data=metrics, raw=raw))
                side_effects.append(SideEffect("upscale", ts, metrics, raw=raw))
                continue
            if UPSCALE_OLD_RE.search(message) or UPSCALE_NEW_RE.search(message):
                length_match = UPSCALE_OLD_RE.search(message) or UPSCALE_NEW_RE.search(message)
                length_val = int(length_match.group("len")) if length_match and length_match.group("len") else None
                data = {"target_long_edge_px": length_val, "format": "old" if UPSCALE_OLD_RE.search(message) else "new"}
                events.append(OplogEvent(ts, level, "upscale", message, data=data, raw=raw))
                side_effects.append(SideEffect("upscale", ts, data, raw=raw))
                continue

            if m := MANIFEST_APPEND_RE.match(message):
                data: Dict[str, Any] = {"path": m.group("path").strip()}
                if m.group("seq"):
                    data["seq"] = int(m.group("seq"))
                events.append(OplogEvent(ts, level, "manifest_append", message, data=data, raw=raw))
                side_effects.append(SideEffect("manifest_append", ts, data, raw=raw))
                continue
            if m := GENERATION_APPEND_RE.match(message):
                data = {"path": m.group("path").strip()}
                events.append(OplogEvent(ts, level, "generation_append", message, data=data, raw=raw))
                side_effects.append(SideEffect("generation_append", ts, data, raw=raw))
                continue

            if m := UPLOAD_BEGIN_RE.match(message):
                data = {"state": "begin", "file": m.group("file").strip(), "dest": m.group("dest").strip()}
                events.append(OplogEvent(ts, level, "upload", message, data=data, raw=raw))
                side_effects.append(SideEffect("upload", ts, data, raw=raw))
                continue
            if m := UPLOAD_COMPLETE_RE.match(message):
                data = {"state": "complete", "dest": m.group("dest").strip()}
                events.append(OplogEvent(ts, level, "upload", message, data=data, raw=raw))
                side_effects.append(SideEffect("upload", ts, data, raw=raw))
                continue
            if m := UPLOAD_SKIPPED_RE.match(message):
                data = {"state": "skipped", "detail": m.group("detail").strip()}
                events.append(OplogEvent(ts, level, "upload", message, data=data, raw=raw))
                side_effects.append(SideEffect("upload", ts, data, raw=raw))
                continue
            if m := UPLOAD_FAILED_RE.match(message) or (m := RCLONE_FAILED_RE.match(message)):
                data = {"state": "error", "detail": (m.group("detail") or "").strip()}
                events.append(OplogEvent(ts, level, "upload", message, data=data, raw=raw))
                side_effects.append(SideEffect("upload", ts, data, raw=raw))
                continue

            if m := OPLOG_STORED_RE.match(message):
                data = {"path": m.group("path").strip(), "kind": "oplog"}
                events.append(OplogEvent(ts, level, "file_write", message, data=data, raw=raw))
                side_effects.append(SideEffect("file_write", ts, data, raw=raw))
                continue
            if m := FILE_WRITE_TO_RE.match(message):
                data = {"path": m.group("path").strip(), "what": m.group("what").strip(), "verb": m.group("verb").strip().lower()}
                events.append(OplogEvent(ts, level, "file_write", message, data=data, raw=raw))
                side_effects.append(SideEffect("file_write", ts, data, raw=raw))
                continue

            unknown.append(raw)

    events.sort(key=lambda e: (e.timestamp, e.path or "", e.type, e.message))
    side_effects.sort(key=lambda e: (e.timestamp, e.type))
    return events, unknown, side_effects
