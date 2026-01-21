"""Artifact helpers (manifest, transcript, and artifacts index).

This package groups together the utilities responsible for writing and indexing
run artifacts. It is intentionally independent of `image_project.stages` and
`image_project.impl` (framework boundary).
"""

from .index import maybe_update_artifacts_index, update_artifacts_index, update_artifacts_index_combined
from .manifest import (
    MANIFEST_FIELDNAMES,
    REQUIRED_MANIFEST_FIELDS,
    TITLE_SOURCE_FALLBACK,
    TITLE_SOURCE_LLM,
    GeneratedTitle,
    append_generation_row,
    append_manifest_row,
    append_run_index_entry,
    csv_fieldnames_reader,
    generate_file_location,
    generate_title,
    generate_unique_id,
    get_next_seq,
    manifest_lock,
    read_manifest,
    sanitize_title,
    utc_now_iso8601,
    validate_title,
)
from .transcript import write_transcript

__all__ = [
    "MANIFEST_FIELDNAMES",
    "REQUIRED_MANIFEST_FIELDS",
    "TITLE_SOURCE_FALLBACK",
    "TITLE_SOURCE_LLM",
    "GeneratedTitle",
    "append_generation_row",
    "append_manifest_row",
    "append_run_index_entry",
    "csv_fieldnames_reader",
    "generate_file_location",
    "generate_title",
    "generate_unique_id",
    "get_next_seq",
    "manifest_lock",
    "maybe_update_artifacts_index",
    "read_manifest",
    "sanitize_title",
    "update_artifacts_index",
    "update_artifacts_index_combined",
    "utc_now_iso8601",
    "validate_title",
    "write_transcript",
]

