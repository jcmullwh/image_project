from __future__ import annotations

import csv
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

TITLE_SOURCE_LLM = "llm"
TITLE_SOURCE_FALLBACK = "fallback"

MANIFEST_FIELDNAMES: list[str] = [
    "seq",
    "title",
    "generation_id",
    "image_prompt",
    "image_path",
    "created_at",
    "model",
    "size",
    "quality",
    "seed",
    "title_source",
    "title_raw",
]

REQUIRED_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {"seq", "title", "generation_id", "image_prompt", "image_path"}
)

_QUOTE_CHARS = "\"'“”‘’"
_ROMAN_NUMERALS: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
)

_TITLE_ALLOWED_RE = re.compile(r"^[A-Za-z -]+$")
_TITLE_WORD_RE = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)*$")
_TITLE_CASE_SEGMENT_RE = re.compile(r"^[A-Z][a-z]+$")


def utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_unique_id() -> str:
    unique_id = uuid.uuid4()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{unique_id}"


def generate_file_location(file_path: str, id: str, file_type: str) -> str:
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")
    if not id or not isinstance(id, str):
        raise ValueError("id must be a non-empty string")
    if not file_type or not isinstance(file_type, str):
        raise ValueError("file_type must be a non-empty string")

    return os.path.join(file_path, id + file_type)


def read_manifest(manifest_path: str) -> list[dict[str, str]]:
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        return []

    with open(manifest_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            if None in row:
                row.pop(None, None)
            cleaned = {str(k): ("" if v is None else str(v)) for k, v in row.items()}
            rows.append(cleaned)
        return rows


def get_next_seq(manifest_path: str) -> int:
    rows = read_manifest(manifest_path)
    max_seq = 0
    for row in rows:
        raw = (row.get("seq") or "").strip()
        try:
            value = int(raw)
        except Exception:
            continue
        if value > max_seq:
            max_seq = value
    return max_seq + 1 if max_seq > 0 else 1


def append_manifest_row(
    manifest_path: str,
    row: Mapping[str, Any],
    *,
    fieldnames: Sequence[str] = MANIFEST_FIELDNAMES,
) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(row.keys()))
    if missing:
        raise KeyError(f"Manifest row missing required fields: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    file_exists = os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0

    output_row: dict[str, Any] = {name: row.get(name, "") for name in fieldnames}
    if not output_row.get("created_at"):
        output_row["created_at"] = utc_now_iso8601()

    with open(manifest_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(output_row)


@contextmanager
def manifest_lock(
    manifest_path: str,
    *,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 0.1,
):
    lock_path = f"{manifest_path}.lock"
    start = time.monotonic()
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(f"pid={os.getpid()}\ncreated_at={utc_now_iso8601()}\n")
            break
        except FileExistsError:
            if (time.monotonic() - start) >= timeout_seconds:
                raise TimeoutError(f"Timed out waiting for manifest lock: {lock_path}")
            time.sleep(poll_interval_seconds)

    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _normalize_title_key(title: str) -> str:
    return title.casefold().strip()


def _strip_outer_quotes(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] in _QUOTE_CHARS and s[-1] in _QUOTE_CHARS:
        return s[1:-1].strip()
    return s


def _single_nonempty_line(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Title response is empty")
    if len(lines) != 1:
        raise ValueError("Title response must be a single line of text")
    return lines[0]


def _is_roman_numeral(word: str) -> bool:
    return word in _ROMAN_NUMERALS


def validate_title(title: str) -> None:
    candidate = title.strip()
    if not candidate:
        raise ValueError("Title is empty")

    if any(ch in candidate for ch in _QUOTE_CHARS):
        raise ValueError("Title must not contain quotes")

    if not _TITLE_ALLOWED_RE.match(candidate):
        raise ValueError("Title contains invalid characters (only letters, spaces, hyphen allowed)")

    words = candidate.split()
    if not (2 <= len(words) <= 4):
        raise ValueError("Title must be 2-4 words")

    for word_idx, word in enumerate(words):
        if _is_roman_numeral(word):
            continue

        if not _TITLE_WORD_RE.match(word):
            raise ValueError("Title contains invalid punctuation")

        segments = word.split("-")
        for seg_idx, seg in enumerate(segments):
            if _is_roman_numeral(seg):
                continue
            if not _TITLE_CASE_SEGMENT_RE.match(seg):
                raise ValueError("Title must be Title Case")


def sanitize_title(raw_title: str) -> str:
    text = _single_nonempty_line(raw_title)
    text = _strip_outer_quotes(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class GeneratedTitle:
    title: str
    title_source: str
    title_raw: str
    attempts: tuple[dict[str, Any], ...] = ()


def _sanitize_title_loose(raw_title: str) -> str:
    lines = [ln.strip() for ln in str(raw_title).splitlines() if ln.strip()]
    if not lines:
        return ""
    text = _strip_outer_quotes(lines[0])
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _disambiguate_title_permissive(base_title: str, existing_titles: Iterable[str]) -> str:
    existing = {_normalize_title_key(t) for t in existing_titles if t}
    base = base_title.strip()
    if not base:
        base = "Untitled"

    if _normalize_title_key(base) not in existing:
        return base

    for roman in _ROMAN_NUMERALS[1:]:  # start at II
        candidate = f"{base} {roman}"
        if _normalize_title_key(candidate) not in existing:
            return candidate

    # Last-resort suffix if we somehow exhaust roman numerals.
    for idx in range(2, 50):
        candidate = f"{base} ({idx})"
        if _normalize_title_key(candidate) not in existing:
            return candidate

    return f"{base} ({int(time.time())})"


def _disambiguate_title(base_title: str, existing_titles: Iterable[str]) -> str:
    existing = {_normalize_title_key(t) for t in existing_titles if t}
    base = base_title.strip()

    words = base.split()
    for roman in _ROMAN_NUMERALS[1:]:  # start at II
        if len(words) < 4:
            candidate = f"{base} {roman}"
        else:
            candidate = " ".join(words[:-1] + [f"{words[-1]}-{roman}"])

        try:
            validate_title(candidate)
        except Exception:
            continue

        if _normalize_title_key(candidate) not in existing:
            return candidate

    raise ValueError("Failed to disambiguate title after exhausting suffixes")


def generate_title(
    *,
    ai_text: Any,
    image_prompt: str,
    avoid_titles: Sequence[str] | None = None,
    max_attempts: int = 3,
    temperature: float = 0.2,
    logger: logging.Logger | None = None,
) -> GeneratedTitle:
    """
    Generate a short, human-friendly title for an image prompt using the existing TextAI backend.

    This function enforces hard constraints in code and retries up to `max_attempts` times.
    If the model keeps colliding with existing titles, we deterministically disambiguate (II/III/etc).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    avoid_titles = list(avoid_titles or [])
    existing_keys = {_normalize_title_key(t) for t in avoid_titles if t}
    from image_project.prompts.titles import build_title_messages

    last_raw = ""
    last_sanitized = ""
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        messages = build_title_messages(
            image_prompt=image_prompt,
            avoid_titles=avoid_titles,
            attempt=attempt,
        )

        raw = ai_text.text_chat(messages, temperature=temperature)
        last_raw = "" if raw is None else str(raw)

        attempt_record: dict[str, Any] = {"attempt": attempt, "raw": last_raw}

        try:
            sanitized = sanitize_title(last_raw)
        except Exception as exc:
            attempt_record["status"] = "rejected"
            attempt_record["reason"] = "sanitize_failed"
            attempt_record["error"] = f"{exc.__class__.__name__}: {exc}"
            attempts.append(dict(attempt_record))
            if logger:
                logger.info(
                    "Title attempt %d rejected: %s (raw=%r)",
                    attempt,
                    attempt_record["error"],
                    last_raw,
                )
            last_sanitized = ""
            continue

        try:
            validate_title(sanitized)
        except Exception as exc:
            attempt_record["status"] = "rejected"
            attempt_record["reason"] = "validate_failed"
            attempt_record["sanitized"] = sanitized
            attempt_record["error"] = f"{exc.__class__.__name__}: {exc}"
            attempts.append(dict(attempt_record))
            if logger:
                logger.info(
                    "Title attempt %d rejected: %s (sanitized=%r raw=%r)",
                    attempt,
                    attempt_record["error"],
                    sanitized,
                    last_raw,
                )
            last_sanitized = ""
            continue

        last_sanitized = sanitized
        if _normalize_title_key(sanitized) in existing_keys:
            attempt_record["status"] = "rejected"
            attempt_record["reason"] = "collision"
            attempt_record["sanitized"] = sanitized
            attempts.append(dict(attempt_record))
            if logger:
                logger.info(
                    "Title attempt %d rejected: collision (sanitized=%r raw=%r)",
                    attempt,
                    sanitized,
                    last_raw,
                )
            continue

        attempt_record["status"] = "accepted"
        attempt_record["sanitized"] = sanitized
        attempts.append(dict(attempt_record))
        return GeneratedTitle(
            title=sanitized,
            title_source=TITLE_SOURCE_LLM,
            title_raw=last_raw,
            attempts=tuple(dict(item) for item in attempts),
        )

    if not last_sanitized:
        loose = _sanitize_title_loose(last_raw)
        if not loose:
            loose = "Untitled"

        disambiguated = _disambiguate_title_permissive(loose, avoid_titles)
        if logger:
            logger.warning(
                "Title generation: using fallback title after %d invalid attempts (title=%r raw=%r)",
                max_attempts,
                disambiguated,
                last_raw,
            )

        return GeneratedTitle(
            title=disambiguated,
            title_source=TITLE_SOURCE_FALLBACK,
            title_raw=last_raw,
            attempts=tuple(dict(item) for item in attempts),
        )

    disambiguated = _disambiguate_title(last_sanitized, avoid_titles)
    return GeneratedTitle(
        title=disambiguated,
        title_source=TITLE_SOURCE_LLM,
        title_raw=last_raw,
        attempts=tuple(dict(item) for item in attempts),
    )


def append_generation_row(csv_path: str, row: dict[str, Any], fieldnames: list[str]) -> None:
    if not fieldnames:
        raise ValueError("fieldnames must be non-empty")

    missing = [name for name in fieldnames if name not in row]
    if missing:
        raise KeyError(f"Generation row missing required keys: {missing}")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if file_exists:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            existing_header: list[str] = next(reader, [])
        if existing_header and existing_header != list(fieldnames):
            raise ValueError(
                f"Existing CSV header does not match expected schema at {csv_path}. "
                f"expected={list(fieldnames)} actual={existing_header}"
            )

    output_row = {name: row.get(name, "") for name in fieldnames}

    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(output_row)


def csv_fieldnames_reader(path: str) -> list[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        return next(reader, []) or []


def append_run_index_entry(path: str, entry: Mapping[str, Any]) -> None:
    """
    Append a single JSON object to a JSONL run index file.

    The caller is responsible for building a schema_versioned entry object.
    """

    import json  # local import to keep artifacts module lightweight at import time

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), ensure_ascii=False))
        handle.write("\n")
