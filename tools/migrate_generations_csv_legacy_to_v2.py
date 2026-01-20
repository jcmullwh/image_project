from __future__ import annotations

"""
Migrate historical (legacy) `generations.csv` files into the v2 schema used by this repo.

Why this exists:
- The runtime and `image_project index-artifacts` now require the v2 generations CSV schema.
- Legacy schema sniffing/parsing has been removed to avoid silent fallbacks and hidden behavior.

Supported input schema (legacy):
- ID
- Description Prompt
- Generation Prompt
- Image URL

Output schema (v2):
- generation_id
- selected_concepts
- final_image_prompt
- image_path
- created_at
- seed

Notes:
- Legacy files typically do not contain `created_at` or `seed`; these are left blank unless the
  legacy CSV already contains columns named `created_at` and/or `seed`.
- The legacy "Image URL" column is often NOT a URL/path (it may contain a response blob). This
  tool uses a conservative heuristic for values that "look like" a URL or file path. When a value
  does not look like a URL/path, the output `image_path` is left blank and the raw value is
  preserved in a sidecar JSONL file for audit/debugging.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger("migrate_generations_csv_legacy_to_v2")

V2_FIELDNAMES: tuple[str, ...] = (
    "generation_id",
    "selected_concepts",
    "final_image_prompt",
    "image_path",
    "created_at",
    "seed",
)


def normalize_header(name: str) -> str:
    """
    Normalize a CSV header cell into a stable key.

    This matches the normalization strategy used by the artifacts indexer:
    - casefold
    - trim whitespace
    - collapse non-alphanumerics into underscores
    """

    text = (name or "").strip().casefold()
    if not text:
        return ""
    out: list[str] = []
    last_was_sep = False
    for ch in text:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            out.append(ch)
            last_was_sep = False
            continue
        if not last_was_sep:
            out.append("_")
            last_was_sep = True
    return "".join(out).strip("_")


def is_windows_abs_path(text: str) -> bool:
    """
    Return True if `text` looks like an absolute Windows path (drive letter or UNC).
    """

    s = (text or "").strip()
    if not s:
        return False
    if s.startswith("\\\\"):
        return True
    if len(s) >= 3 and s[1] == ":" and s[2] in ("/", "\\"):
        return True
    return False


def looks_like_url_or_path(value: str) -> bool:
    """
    Conservative heuristic for whether a cell value is likely a URL or a file path.

    This is used only for migrating legacy data where the "Image URL" column often contains
    non-URL content (e.g., a response blob). The heuristic is intentionally strict to avoid
    silently treating non-path content as a usable image reference.
    """

    text = (value or "").strip()
    if not text:
        return False
    lower = text.casefold()
    if lower.startswith("http://") or lower.startswith("https://"):
        return True
    if is_windows_abs_path(text):
        return True
    if text.startswith("/") or text.startswith("./") or text.startswith("../"):
        return True
    if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov")):
        return True
    return False


def _header_map(header: list[str]) -> dict[str, str]:
    """
    Build a normalized-header -> original-header mapping, ignoring empty header cells.
    """

    return {
        normalize_header(name): name
        for name in header
        if isinstance(name, str) and name.strip() and normalize_header(name)
    }


def _read_csv_header(path: Path) -> list[str]:
    """
    Read the first row of a CSV file as a header list.
    """

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, []) or []


def migrate_legacy_generations_csv_to_v2(
    *,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """
    Convert a legacy generations CSV to the v2 schema and write it to `output_path`.

    Returns a machine-readable report dict. This function is deterministic and offline.

    Raises:
      - FileNotFoundError if `input_path` does not exist.
      - ValueError if the input CSV header is not a recognized legacy generations schema.
    """

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if input_path.stat().st_size == 0:
        raise ValueError(f"Input CSV is empty: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output CSV: {output_path} (pass --overwrite to allow)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = _read_csv_header(input_path)
    header_map = _header_map(header)

    required_legacy = ("id", "description_prompt", "generation_prompt", "image_url")
    missing_legacy = [key for key in required_legacy if key not in header_map]
    if missing_legacy:
        raise ValueError(
            "Unsupported generations CSV header for legacy migration "
            f"(missing={missing_legacy} header={header}). "
            "Expected legacy columns: ID, Description Prompt, Generation Prompt, Image URL."
        )

    id_key = header_map["id"]
    desc_key = header_map["description_prompt"]
    prompt_key = header_map["generation_prompt"]
    image_key = header_map["image_url"]
    created_key = header_map.get("created_at")
    seed_key = header_map.get("seed")

    unmapped_image_values_path = output_path.with_suffix(output_path.suffix + ".unmapped_image_values.jsonl")
    report_path = output_path.with_suffix(output_path.suffix + ".migration_report.json")

    rows_total = 0
    rows_written = 0
    unmapped_image_values_count = 0
    unmapped_image_value_samples: list[dict[str, Any]] = []

    sidecar_handle = None
    try:
        with (
            open(input_path, "r", encoding="utf-8-sig", newline="") as in_handle,
            open(output_path, "w", encoding="utf-8", newline="") as out_handle,
        ):
            reader = csv.DictReader(in_handle)
            writer = csv.DictWriter(out_handle, fieldnames=list(V2_FIELDNAMES), extrasaction="ignore")
            writer.writeheader()

            for row_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                if None in row:
                    row.pop(None, None)

                rows_total += 1

                generation_id_raw = row.get(id_key)
                generation_id = str(generation_id_raw).strip() if generation_id_raw is not None else ""
                if not generation_id:
                    raise ValueError(f"{input_path}: row {row_number}: missing ID")

                selected_raw = row.get(desc_key)
                selected_concepts = str(selected_raw).strip() if selected_raw is not None else ""

                prompt_raw = row.get(prompt_key)
                final_image_prompt = str(prompt_raw).strip() if prompt_raw is not None else ""

                image_raw = row.get(image_key)
                image_value = str(image_raw).strip() if image_raw is not None else ""
                image_path = image_value if looks_like_url_or_path(image_value) else ""

                if image_value and not image_path:
                    unmapped_image_values_count += 1
                    if len(unmapped_image_value_samples) < 20:
                        unmapped_image_value_samples.append(
                            {
                                "generation_id": generation_id,
                                "row_number": row_number,
                                "preview": (image_value[:200] + "…") if len(image_value) > 200 else image_value,
                            }
                        )
                    if sidecar_handle is None:
                        sidecar_handle = open(unmapped_image_values_path, "w", encoding="utf-8")
                    sidecar_handle.write(
                        json.dumps(
                            {
                                "generation_id": generation_id,
                                "row_number": row_number,
                                "raw_image_url_field": image_value,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                created_at = ""
                if created_key is not None:
                    raw = row.get(created_key)
                    created_at = str(raw).strip() if raw is not None else ""

                seed = ""
                if seed_key is not None:
                    raw = row.get(seed_key)
                    seed = str(raw).strip() if raw is not None else ""

                writer.writerow(
                    {
                        "generation_id": generation_id,
                        "selected_concepts": selected_concepts,
                        "final_image_prompt": final_image_prompt,
                        "image_path": image_path,
                        "created_at": created_at,
                        "seed": seed,
                    }
                )
                rows_written += 1
    finally:
        if sidecar_handle is not None:
            sidecar_handle.close()

    report: dict[str, Any] = {
        "tool": "migrate_generations_csv_legacy_to_v2",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_total": int(rows_total),
        "rows_written": int(rows_written),
        "unmapped_image_values_count": int(unmapped_image_values_count),
        "unmapped_image_values_path": str(unmapped_image_values_path)
        if unmapped_image_values_count > 0
        else None,
        "unmapped_image_value_samples": unmapped_image_value_samples,
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint.

    Returns a process exit code (0 on success; non-zero on error).
    """

    parser = argparse.ArgumentParser(
        prog="migrate_generations_csv_legacy_to_v2",
        description="Convert a legacy generations.csv into the v2 schema used by this repo.",
    )
    parser.add_argument("--input", required=True, help="Path to legacy generations CSV (e.g., generations.csv).")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the v2 generations CSV (e.g., generations_v2.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_path = Path(str(args.input)).expanduser()
    output_path = Path(str(args.output)).expanduser()

    try:
        report = migrate_legacy_generations_csv_to_v2(
            input_path=input_path,
            output_path=output_path,
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s: %s", exc.__class__.__name__, exc)
        return 2

    LOGGER.info("Wrote v2 generations CSV: %s", report.get("output_path"))
    report_path = output_path.with_suffix(output_path.suffix + ".migration_report.json")
    LOGGER.info("Wrote migration report: %s", report_path)
    unmapped = int(report.get("unmapped_image_values_count") or 0)
    if unmapped > 0:
        LOGGER.warning(
            "Found %d rows where legacy 'Image URL' did not look like a URL/path; preserved raw values in %s",
            unmapped,
            report.get("unmapped_image_values_path"),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

