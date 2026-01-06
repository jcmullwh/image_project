from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]

def _path_str(path: Path, *, root: Path) -> str:
    """
    Prefer paths relative to `root`, but fall back to absolute paths when the
    file lives outside that root (e.g., indexing an external drive).
    """

    try:
        resolved_root = root.resolve()
    except Exception:
        resolved_root = root

    try:
        resolved_path = path.resolve()
    except Exception:
        resolved_path = path

    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except Exception:
        return str(resolved_path)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except Exception:
            return None
    return None


def _relpath_or_original(path: str | None, *, repo_root: Path) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None

    try:
        candidate = Path(text)
    except Exception:
        return text

    try:
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate.relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(candidate)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n")


def _atomic_write_csv(path: Path, *, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _normalize_header(name: str) -> str:
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
    normalized = "".join(out).strip("_")
    return normalized


def _record_completeness_score(record: Mapping[str, Any]) -> int:
    score = 0
    for key in ("created_at", "seed", "image_path", "final_image_prompt", "selected_concepts"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            score += 1
        elif isinstance(value, int):
            score += 1
    return score


def _prefer_record(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Prefer the more "complete" record for the same generation id.
    """

    if _record_completeness_score(incoming) > _record_completeness_score(existing):
        return incoming
    return existing


def _read_generations_csv_any(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Read a generations CSV of varying schema versions and return a mapping keyed by generation_id.

    Supported (observed) schemas:
    - v2: generation_id, selected_concepts, final_image_prompt, image_path, created_at, seed
    - legacy: ID, Description Prompt, Generation Prompt, Image URL (often not a URL; may contain a response blob)
    """

    errors: list[str] = []
    if not path.exists() or path.stat().st_size == 0:
        return {}, errors

    records: dict[str, dict[str, Any]] = {}
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            header_map = {
                _normalize_header(name): name
                for name in header
                if isinstance(name, str) and name.strip()
            }

            generation_id_key = header_map.get("generation_id") or header_map.get("id")
            if generation_id_key is None:
                return {}, [f"Missing generation id column in {path} (header={header})"]

            selected_key = (
                header_map.get("selected_concepts")
                or header_map.get("description_prompt")
                or header_map.get("description")
            )
            prompt_key = (
                header_map.get("final_image_prompt")
                or header_map.get("generation_prompt")
                or header_map.get("prompt")
            )
            image_key = (
                header_map.get("image_path")
                or header_map.get("image_url")
                or header_map.get("url")
                or header_map.get("image")
            )
            created_key = header_map.get("created_at")
            seed_key = header_map.get("seed")

            schema = "v2" if header_map.get("final_image_prompt") or header_map.get("selected_concepts") else "legacy"

            def looks_like_url_or_path(value: str) -> bool:
                text = (value or "").strip()
                if not text:
                    return False
                lower = text.casefold()
                if lower.startswith("http://") or lower.startswith("https://"):
                    return True
                if _is_windows_abs_path(text):
                    return True
                if text.startswith("/") or text.startswith("./") or text.startswith("../"):
                    return True
                if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov")):
                    return True
                return False

            for idx, row in enumerate(reader, start=2):
                if not row:
                    continue
                if None in row:
                    row.pop(None, None)

                raw_gid = row.get(generation_id_key)
                generation_id = str(raw_gid).strip() if raw_gid is not None else ""
                if not generation_id:
                    continue

                selected = None
                if selected_key is not None:
                    value = row.get(selected_key)
                    if value is not None:
                        selected = str(value).strip() or None

                final_prompt = None
                if prompt_key is not None:
                    value = row.get(prompt_key)
                    if value is not None:
                        final_prompt = str(value).strip() or None

                image_path = None
                legacy_output = None
                if image_key is not None:
                    value = row.get(image_key)
                    if value is not None:
                        raw = str(value).strip() or None
                        if schema == "legacy":
                            if raw and looks_like_url_or_path(raw):
                                image_path = raw
                            else:
                                legacy_output = raw
                        else:
                            image_path = raw

                created_at = None
                if created_key is not None:
                    value = row.get(created_key)
                    if value is not None:
                        created_at = str(value).strip() or None

                seed = _safe_int(row.get(seed_key)) if seed_key is not None else None

                record = {
                    "schema": schema,
                    "source_path": str(path),
                    "generation_id": generation_id,
                    "selected_concepts": selected,
                    "final_image_prompt": final_prompt,
                    "image_path": image_path,
                    "legacy_output": legacy_output,
                    "created_at": created_at,
                    "seed": seed,
                    "row_number": idx,
                }

                if generation_id in records:
                    records[generation_id] = dict(_prefer_record(records[generation_id], record))
                else:
                    records[generation_id] = dict(record)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: {exc.__class__.__name__}: {exc}")

    return records, errors


def _collapse_whitespace(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text)
    if not value:
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = " ".join(part.strip() for part in normalized.split("\n"))
    normalized = " ".join(normalized.split())
    return normalized or None


def _is_windows_abs_path(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if s.startswith("\\\\"):
        return True
    if len(s) >= 3 and s[1] == ":" and (s[2] == "\\" or s[2] == "/"):
        return True
    return False


def _abspath_from_base(path: str | None, *, base_dir: Path) -> str | None:
    text = (path or "").strip()
    if not text:
        return None
    if _is_windows_abs_path(text):
        return text
    try:
        candidate = Path(text)
        if candidate.is_absolute():
            return str(candidate)
    except Exception:
        return text

    try:
        return str((base_dir / text).resolve())
    except Exception:
        return str(base_dir / text)


def _guess_store_label(artifacts_path: Path) -> str:
    name = artifacts_path.name.strip() if artifacts_path.name else ""
    if name:
        return name
    drive = artifacts_path.drive.strip(":") if artifacts_path.drive else ""
    if drive:
        return drive
    return str(artifacts_path)


def _with_store_context(
    row: Mapping[str, Any],
    *,
    store_label: str,
    store_root: Path,
    base_dir: Path,
    path_keys: Sequence[str],
) -> dict[str, Any]:
    enriched: dict[str, Any] = {"store": store_label, "store_root": str(store_root)}
    for key, value in row.items():
        if key in path_keys:
            enriched[key] = _abspath_from_base(str(value) if value is not None else None, base_dir=base_dir)
        else:
            enriched[key] = value
    return enriched


def _row_score(row: Mapping[str, Any]) -> int:
    score = 0
    for key in (
        "created_at",
        "seed",
        "status",
        "run_mode",
        "image_path",
        "transcript_path",
        "final_prompt_path",
        "oplog_path",
        "run_report_json_path",
        "title",
        "image_id",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            score += 1
        elif isinstance(value, int):
            score += 1
    return score


def _dedupe_by_key(rows: Sequence[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if value not in out:
            out[value] = dict(row)
            continue
        if _row_score(row) > _row_score(out[value]):
            out[value] = dict(row)
    return list(out.values())


def _scan_generation_csv_files(root: Path) -> list[Path]:
    candidates: list[Path] = []

    # Common locations (non-recursive).
    for rel in (
        Path("data") / "generations_v2.csv",
        Path("logs") / "generations_v2.csv",
        Path("data") / "generations.csv",
        Path("logs") / "generations.csv",
        Path("archive") / "prompt_files" / "generations.csv",
        Path("prompt_files") / "generations.csv",
    ):
        path = root / rel
        if path.exists() and path.is_file():
            candidates.append(path)

    # Broader discovery (recursive) for experiments / legacy folders.
    for name in ("generations_v2.csv", "generations.csv"):
        for path in root.rglob(name):
            if path.exists() and path.is_file():
                candidates.append(path)

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return sorted(out, key=lambda p: p.as_posix())


def _scan_run_review_reports(root: Path) -> dict[str, dict[str, str]]:
    """
    Best-effort discovery of run-review reports by filename convention:
      <generation_id>_run_report.json / .html
    """

    reports: dict[str, dict[str, str]] = {}
    reviews_dir = root / "reviews"
    if not reviews_dir.exists():
        return reports

    for suffix, key in (
        ("_run_report.json", "run_report_json_path"),
        ("_run_report.html", "run_report_html_path"),
    ):
        for path in reviews_dir.rglob(f"*{suffix}"):
            if not path.is_file():
                continue
            name = path.name
            generation_id = name[: -len(suffix)].strip()
            if not generation_id:
                continue
            reports.setdefault(generation_id, {})[key] = str(path)

    return reports


@dataclass(frozen=True)
class ManifestRow:
    manifest_path: Path
    seq: int | None
    title: str | None
    generation_id: str
    image_path: str | None
    created_at: str | None
    model: str | None
    size: str | None
    quality: str | None
    seed: int | None


def _read_titles_manifest(path: Path) -> tuple[list[ManifestRow], list[str]]:
    """
    Read a titles manifest CSV but only keep lightweight columns.

    This intentionally drops the huge `image_prompt` field.
    """

    errors: list[str] = []
    if not path.exists() or path.stat().st_size == 0:
        return [], errors

    rows: list[ManifestRow] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None) or []
            header_map = {str(name).strip(): idx for idx, name in enumerate(header) if name is not None}

            def col(name: str) -> int | None:
                idx = header_map.get(name)
                return int(idx) if isinstance(idx, int) else None

            idx_seq = col("seq")
            idx_title = col("title")
            idx_generation_id = col("generation_id")
            idx_image_path = col("image_path")
            idx_created_at = col("created_at")
            idx_model = col("model")
            idx_size = col("size")
            idx_quality = col("quality")
            idx_seed = col("seed")

            if idx_generation_id is None:
                return [], [f"Missing required column 'generation_id' in {path}"]

            for raw_row in reader:
                if not raw_row:
                    continue
                try:
                    generation_id = str(raw_row[idx_generation_id]).strip()
                except Exception:
                    continue
                if not generation_id:
                    continue

                seq = _safe_int(raw_row[idx_seq]) if idx_seq is not None and idx_seq < len(raw_row) else None
                seed = _safe_int(raw_row[idx_seed]) if idx_seed is not None and idx_seed < len(raw_row) else None

                title = (
                    str(raw_row[idx_title]).strip()
                    if idx_title is not None and idx_title < len(raw_row) and raw_row[idx_title] is not None
                    else None
                )
                if title == "":
                    title = None

                image_path = (
                    str(raw_row[idx_image_path]).strip()
                    if idx_image_path is not None
                    and idx_image_path < len(raw_row)
                    and raw_row[idx_image_path] is not None
                    else None
                )
                if image_path == "":
                    image_path = None

                created_at = (
                    str(raw_row[idx_created_at]).strip()
                    if idx_created_at is not None
                    and idx_created_at < len(raw_row)
                    and raw_row[idx_created_at] is not None
                    else None
                )
                if created_at == "":
                    created_at = None

                model = (
                    str(raw_row[idx_model]).strip()
                    if idx_model is not None and idx_model < len(raw_row) and raw_row[idx_model] is not None
                    else None
                )
                if model == "":
                    model = None

                size = (
                    str(raw_row[idx_size]).strip()
                    if idx_size is not None and idx_size < len(raw_row) and raw_row[idx_size] is not None
                    else None
                )
                if size == "":
                    size = None

                quality = (
                    str(raw_row[idx_quality]).strip()
                    if idx_quality is not None
                    and idx_quality < len(raw_row)
                    and raw_row[idx_quality] is not None
                    else None
                )
                if quality == "":
                    quality = None

                rows.append(
                    ManifestRow(
                        manifest_path=path,
                        seq=seq,
                        title=title,
                        generation_id=generation_id,
                        image_path=image_path,
                        created_at=created_at,
                        model=model,
                        size=size,
                        quality=quality,
                        seed=seed,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: {exc.__class__.__name__}: {exc}")

    return rows, errors


def _scan_run_index_files(artifacts_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    entries: dict[str, dict[str, Any]] = {}

    for path in sorted(artifacts_root.rglob("runs_index.jsonl")):
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{path}:{idx}: {exc.__class__.__name__}: {exc}")
                        continue
                    if not isinstance(obj, dict):
                        errors.append(f"{path}:{idx}: expected JSON object")
                        continue
                    generation_id = obj.get("generation_id")
                    if not isinstance(generation_id, str) or not generation_id.strip():
                        continue

                    if generation_id in entries:
                        # Prefer the entry that has an image artifact (full mode), otherwise keep the first.
                        existing = entries[generation_id]
                        existing_has_image = bool(
                            isinstance(existing.get("artifacts"), dict)
                            and existing["artifacts"].get("image")
                        )
                        incoming_has_image = bool(
                            isinstance(obj.get("artifacts"), dict) and obj["artifacts"].get("image")
                        )
                        if existing_has_image or not incoming_has_image:
                            continue

                    enriched = dict(obj)
                    enriched["_source_path"] = str(path)
                    entries[generation_id] = enriched
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc.__class__.__name__}: {exc}")

    return entries, errors


def _pick_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _format_image_id(seq: int | None) -> str | None:
    if seq is None:
        return None
    return f"#{seq:03d}"


def _format_image_label(seq: int | None, title: str | None) -> str | None:
    image_id = _format_image_id(seq)
    if not image_id and not title:
        return None
    if image_id and title:
        return f"{image_id} - {title}"
    return image_id or title


def update_artifacts_index(
    *,
    artifacts_root: str | os.PathLike[str] = "_artifacts",
    output_dir: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """
    Scan `_artifacts` and write a compact, query-friendly index:

    - experiments index (JSON): high-level summary + distinct variant info per experiment
    - experiment registry (CSV): one row per planned experiment run (A1/B2/...)
    - image registry (CSV): one row per titled image (from titles_manifest.csv)
    """

    artifacts_input = Path(artifacts_root)
    if repo_root is not None:
        root = Path(repo_root)
    elif artifacts_input.is_absolute():
        # Indexing an external store; default to store-relative paths.
        root = artifacts_input
    else:
        root = _repo_root_from_here()

    root = root.resolve()
    artifacts_path = artifacts_input.resolve() if artifacts_input.is_absolute() else (root / artifacts_input).resolve()

    out_dir = Path(output_dir) if output_dir is not None else artifacts_path / "index"
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (root / out_dir).resolve()

    generated_at = _utc_now_iso8601()

    # Index sources we reuse.
    run_index_by_generation, run_index_errors = _scan_run_index_files(artifacts_path)

    manifest_rows: list[ManifestRow] = []
    manifest_errors: list[str] = []
    for manifest_path in sorted(artifacts_path.rglob("titles_manifest.csv")):
        rows, errors = _read_titles_manifest(manifest_path)
        manifest_rows.extend(rows)
        manifest_errors.extend(errors)

    manifest_by_generation: dict[str, ManifestRow] = {}
    manifest_duplicates: list[str] = []
    for row in manifest_rows:
        if row.generation_id in manifest_by_generation:
            manifest_duplicates.append(row.generation_id)
            continue
        manifest_by_generation[row.generation_id] = row

    generation_csv_paths = _scan_generation_csv_files(artifacts_path)
    generation_csv_errors: list[str] = []
    generation_csv_by_id: dict[str, dict[str, Any]] = {}
    for path in generation_csv_paths:
        records, errors = _read_generations_csv_any(path)
        generation_csv_errors.extend(errors)
        for generation_id, record in records.items():
            if generation_id in generation_csv_by_id:
                generation_csv_by_id[generation_id] = dict(
                    _prefer_record(generation_csv_by_id[generation_id], record)
                )
            else:
                generation_csv_by_id[generation_id] = dict(record)

    run_review_reports = _scan_run_review_reports(artifacts_path)

    experiments_root = artifacts_path / "experiments"
    plan_paths = sorted(experiments_root.rglob("experiment_plan.json")) if experiments_root.exists() else []

    experiment_summaries: list[dict[str, Any]] = []
    experiment_run_rows: list[dict[str, Any]] = []
    experiment_plan_errors: list[str] = []

    planned_run_by_generation: dict[str, dict[str, Any]] = {}

    for plan_path in plan_paths:
        if not plan_path.is_file():
            continue
        experiment_dir = plan_path.parent
        experiment_dir_rel = _path_str(experiment_dir, root=root)

        try:
            plan = _read_json(plan_path)
        except Exception as exc:  # noqa: BLE001
            experiment_plan_errors.append(f"{plan_path}: {exc.__class__.__name__}: {exc}")
            continue

        experiment_id = plan.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            experiment_id = experiment_dir.name

        run_mode = _pick_str(plan.get("run_mode"))
        base_seed = _safe_int(plan.get("base_seed"))

        planned_runs = plan.get("planned_runs")
        planned_runs_list: list[dict[str, Any]] = (
            list(planned_runs) if isinstance(planned_runs, list) else []
        )

        variant_key_field: str | None = None
        if planned_runs_list:
            sample = planned_runs_list[0]
            if isinstance(sample, dict):
                if "set" in sample:
                    variant_key_field = "set"
                elif "variant" in sample:
                    variant_key_field = "variant"

        variant_keys: list[str] = []
        for item in planned_runs_list:
            if not isinstance(item, dict):
                continue
            raw_key = item.get(variant_key_field) if variant_key_field else None
            key = _pick_str(raw_key)
            if key and key not in variant_keys:
                variant_keys.append(key)

        # Variant definitions: collect distinct pieces by variant key.
        variants: list[dict[str, Any]] = []
        for key in variant_keys:
            planned_subset = [
                item
                for item in planned_runs_list
                if isinstance(item, dict) and _pick_str(item.get(variant_key_field)) == key
            ]
            planned_plan_names = sorted(
                {str(item.get("plan")).strip() for item in planned_subset if _pick_str(item.get("plan"))}
            )
            planned_variant_names = sorted(
                {
                    str(item.get("variant_name")).strip()
                    for item in planned_subset
                    if _pick_str(item.get("variant_name"))
                }
            )

            exemplar_run = next(
                (
                    run_index_by_generation.get(_pick_str(item.get("generation_id")) or "")
                    for item in planned_subset
                    if _pick_str(item.get("generation_id"))
                ),
                None,
            )
            exp_variant = None
            exp_notes = None
            exp_tags: list[str] = []
            if isinstance(exemplar_run, dict):
                exp_info = exemplar_run.get("experiment")
                if isinstance(exp_info, dict):
                    exp_variant = _pick_str(exp_info.get("variant"))
                    exp_notes = _pick_str(exp_info.get("notes"))
                    raw_tags = exp_info.get("tags")
                    if isinstance(raw_tags, list):
                        exp_tags = [str(t).strip() for t in raw_tags if str(t).strip()]

            variants.append(
                {
                    "key": key,
                    "planned_plan": planned_plan_names[0]
                    if len(planned_plan_names) == 1
                    else (planned_plan_names or None),
                    "planned_variant_name": planned_variant_names[0]
                    if len(planned_variant_names) == 1
                    else (planned_variant_names or None),
                    "experiment_variant": exp_variant,
                    "notes": exp_notes,
                    "tags": exp_tags,
                    "planned_runs": len(planned_subset),
                }
            )

        titles_manifest_path = experiment_dir / "generated" / "titles_manifest.csv"
        runs_index_path = experiment_dir / "logs" / "runs_index.jsonl"
        results_path = experiment_dir / "experiment_results.json"
        plan_full_path = experiment_dir / "experiment_plan_full.json"

        images_count = 0
        if titles_manifest_path.exists():
            images_count = sum(
                1
                for row in manifest_rows
                if row.manifest_path.resolve() == titles_manifest_path.resolve()
            )

        executed_runs_count = 0
        if runs_index_path.exists():
            try:
                with open(runs_index_path, "r", encoding="utf-8") as handle:
                    executed_runs_count = sum(1 for line in handle if line.strip())
            except Exception:  # noqa: BLE001
                executed_runs_count = 0

        experiment_summaries.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "experiment_dir": experiment_dir_rel,
                "plan_path": _path_str(plan_path, root=root),
                "plan_full_path": _path_str(plan_full_path, root=root) if plan_full_path.exists() else None,
                "results_path": _path_str(results_path, root=root) if results_path.exists() else None,
                "runs_index_path": _path_str(runs_index_path, root=root) if runs_index_path.exists() else None,
                "titles_manifest_path": _path_str(titles_manifest_path, root=root)
                if titles_manifest_path.exists()
                else None,
                "run_mode": run_mode,
                "base_seed": base_seed,
                "variant_keys": list(variant_keys),
                "variants": variants,
                "planned_runs": len([item for item in planned_runs_list if isinstance(item, dict)]),
                "executed_runs": executed_runs_count,
                "images": images_count,
            }
        )

        # Per-run rows (experimental registry).
        for item in planned_runs_list:
            if not isinstance(item, dict):
                continue

            generation_id = _pick_str(item.get("generation_id"))
            if not generation_id:
                continue

            run_number = _safe_int(item.get("run"))
            variant_key = _pick_str(item.get(variant_key_field)) if variant_key_field else None
            if not variant_key:
                variant_key = generation_id.split("_", 1)[0][:1] if generation_id else None

            run_id = f"{variant_key}{run_number}" if variant_key and run_number is not None else None

            planned_run_by_generation[generation_id] = {
                "experiment_id": experiment_id,
                "experiment_dir": experiment_dir_rel,
                "run_id": run_id,
                "variant_key": variant_key,
                "planned_plan": _pick_str(item.get("plan")),
                "planned_variant_name": _pick_str(item.get("variant_name")),
                "run": run_number,
                "seed": _safe_int(item.get("seed")),
            }

            manifest = manifest_by_generation.get(generation_id)
            run_index = run_index_by_generation.get(generation_id)

            status = _pick_str(run_index.get("status")) if isinstance(run_index, dict) else None
            created_at = None
            if isinstance(run_index, dict):
                created_at = _pick_str(run_index.get("created_at"))
            if created_at is None and manifest is not None:
                created_at = manifest.created_at

            artifacts = run_index.get("artifacts") if isinstance(run_index, dict) else None
            artifacts = artifacts if isinstance(artifacts, dict) else {}

            seq = manifest.seq if manifest is not None else None
            title = manifest.title if manifest is not None else None

            experiment_run_rows.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_dir": experiment_dir_rel,
                    "run_id": run_id,
                    "generation_id": generation_id,
                    "variant_key": variant_key,
                    "run": run_number,
                    "seed": _safe_int(item.get("seed")),
                    "run_mode": run_mode,
                    "status": status or "planned",
                    "created_at": created_at,
                    "title_seq": seq,
                    "title": title,
                    "image_id": _format_image_id(seq),
                    "image_label": _format_image_label(seq, title),
                    "image_path": _relpath_or_original(
                        manifest.image_path if manifest is not None else _pick_str(artifacts.get("image")),
                        repo_root=root,
                    ),
                    "upscaled_image_path": _relpath_or_original(
                        _pick_str(artifacts.get("upscaled_image")),
                        repo_root=root,
                    ),
                    "transcript_path": _relpath_or_original(
                        _pick_str(artifacts.get("transcript")),
                        repo_root=root,
                    ),
                    "final_prompt_path": _relpath_or_original(
                        _pick_str(artifacts.get("final_prompt")),
                        repo_root=root,
                    ),
                    "oplog_path": _relpath_or_original(
                        _pick_str(artifacts.get("oplog")),
                        repo_root=root,
                    ),
                    "titles_manifest_path": _path_str(titles_manifest_path, root=root)
                    if titles_manifest_path.exists()
                    else None,
                }
            )

    # Image registry (one row per manifest row).
    image_rows: list[dict[str, Any]] = []
    for row in sorted(
        manifest_rows,
        key=lambda r: (
            r.created_at or "",
            r.manifest_path.as_posix(),
            r.seq if r.seq is not None else 0,
            r.generation_id,
        ),
    ):
        planned = planned_run_by_generation.get(row.generation_id, {})
        run_index = run_index_by_generation.get(row.generation_id, {})
        artifacts = run_index.get("artifacts") if isinstance(run_index, dict) else None
        artifacts = artifacts if isinstance(artifacts, dict) else {}

        seed = row.seed
        if seed is None and isinstance(run_index, dict):
            seed = _safe_int(run_index.get("seed"))
        if seed is None and isinstance(planned.get("seed"), int):
            seed = int(planned["seed"])

        exp_id = planned.get("experiment_id")
        exp_dir = planned.get("experiment_dir")
        if not exp_id and isinstance(run_index, dict):
            exp = run_index.get("experiment")
            if isinstance(exp, dict):
                exp_id = _pick_str(exp.get("id"))

        image_rows.append(
            {
                "generation_id": row.generation_id,
                "title_seq": row.seq,
                "title": row.title,
                "image_id": _format_image_id(row.seq),
                "image_label": _format_image_label(row.seq, row.title),
                "image_path": _relpath_or_original(row.image_path, repo_root=root),
                "created_at": row.created_at,
                "seed": seed,
                "model": row.model,
                "size": row.size,
                "quality": row.quality,
                "manifest_path": _path_str(row.manifest_path, root=root),
                "experiment_id": exp_id,
                "experiment_dir": exp_dir,
                "experiment_run_id": planned.get("run_id"),
                "variant_key": planned.get("variant_key"),
                "planned_plan": planned.get("planned_plan"),
                "planned_variant_name": planned.get("planned_variant_name"),
                "run_mode": _pick_str(run_index.get("run_mode")) if isinstance(run_index, dict) else None,
                "status": _pick_str(run_index.get("status")) if isinstance(run_index, dict) else None,
                "transcript_path": _relpath_or_original(_pick_str(artifacts.get("transcript")), repo_root=root),
            }
        )

    # Run registry (union of all sources).
    all_generation_ids: set[str] = set()
    all_generation_ids.update(run_index_by_generation.keys())
    all_generation_ids.update(manifest_by_generation.keys())
    all_generation_ids.update(generation_csv_by_id.keys())
    all_generation_ids.update(planned_run_by_generation.keys())
    all_generation_ids.update(run_review_reports.keys())

    run_rows: list[dict[str, Any]] = []
    for generation_id in sorted(all_generation_ids):
        planned = planned_run_by_generation.get(generation_id, {})
        run_index = run_index_by_generation.get(generation_id, {})
        manifest = manifest_by_generation.get(generation_id)
        gen_csv = generation_csv_by_id.get(generation_id, {})
        review = run_review_reports.get(generation_id, {})

        artifacts = run_index.get("artifacts") if isinstance(run_index, dict) else None
        artifacts = artifacts if isinstance(artifacts, dict) else {}

        exp_id = planned.get("experiment_id")
        exp_variant = None
        exp_notes = None
        exp_tags: list[str] = []
        if isinstance(run_index, dict):
            exp = run_index.get("experiment")
            if isinstance(exp, dict):
                exp_id = exp_id or _pick_str(exp.get("id"))
                exp_variant = _pick_str(exp.get("variant"))
                exp_notes = _pick_str(exp.get("notes"))
                raw_tags = exp.get("tags")
                if isinstance(raw_tags, list):
                    exp_tags = [str(t).strip() for t in raw_tags if str(t).strip()]

        created_at = None
        if isinstance(run_index, dict):
            created_at = _pick_str(run_index.get("created_at"))
        if created_at is None and manifest is not None:
            created_at = manifest.created_at
        if created_at is None:
            created_at = _pick_str(gen_csv.get("created_at"))

        seed = None
        if isinstance(run_index, dict):
            seed = _safe_int(run_index.get("seed"))
        if seed is None:
            seed = _safe_int(gen_csv.get("seed"))
        if seed is None and isinstance(planned.get("seed"), int):
            seed = int(planned["seed"])
        if seed is None and manifest is not None:
            seed = manifest.seed

        seq = manifest.seq if manifest is not None else None
        title = manifest.title if manifest is not None else None

        image_path = None
        if manifest is not None and manifest.image_path:
            image_path = manifest.image_path
        if image_path is None:
            image_path = _pick_str(artifacts.get("image"))
        if image_path is None:
            image_path = _pick_str(gen_csv.get("image_path"))

        run_rows.append(
            {
                "generation_id": generation_id,
                "created_at": created_at,
                "seed": seed,
                "status": _pick_str(run_index.get("status")) if isinstance(run_index, dict) else None,
                "phase": _pick_str(run_index.get("phase")) if isinstance(run_index, dict) else None,
                "run_mode": _pick_str(run_index.get("run_mode")) if isinstance(run_index, dict) else None,
                "experiment_id": exp_id,
                "experiment_variant": exp_variant,
                "experiment_notes": exp_notes,
                "experiment_tags": "|".join(exp_tags) if exp_tags else None,
                "experiment_run_id": planned.get("run_id"),
                "variant_key": planned.get("variant_key"),
                "run": planned.get("run"),
                "planned_plan": planned.get("planned_plan"),
                "planned_variant_name": planned.get("planned_variant_name"),
                "title_seq": seq,
                "title": title,
                "image_id": _format_image_id(seq),
                "image_label": _format_image_label(seq, title),
                "image_path": _relpath_or_original(image_path, repo_root=root),
                "upscaled_image_path": _relpath_or_original(
                    _pick_str(artifacts.get("upscaled_image")),
                    repo_root=root,
                ),
                "transcript_path": _relpath_or_original(
                    _pick_str(artifacts.get("transcript")),
                    repo_root=root,
                ),
                "final_prompt_path": _relpath_or_original(
                    _pick_str(artifacts.get("final_prompt")),
                    repo_root=root,
                ),
                "oplog_path": _relpath_or_original(
                    _pick_str(artifacts.get("oplog")),
                    repo_root=root,
                ),
                "run_report_json_path": _relpath_or_original(
                    _pick_str(review.get("run_report_json_path")),
                    repo_root=root,
                ),
                "run_report_html_path": _relpath_or_original(
                    _pick_str(review.get("run_report_html_path")),
                    repo_root=root,
                ),
                "titles_manifest_path": _path_str(manifest.manifest_path, root=root)
                if manifest is not None
                else None,
                "generations_source_path": _relpath_or_original(
                    _pick_str(gen_csv.get("source_path")),
                    repo_root=root,
                ),
                "generations_schema": _pick_str(gen_csv.get("schema")),
                "selected_concepts": _collapse_whitespace(_pick_str(gen_csv.get("selected_concepts"))),
                "final_image_prompt": _collapse_whitespace(_pick_str(gen_csv.get("final_image_prompt"))),
                "legacy_output": _collapse_whitespace(_pick_str(gen_csv.get("legacy_output"))),
            }
        )

    # Write outputs.
    experiments_index_path = out_dir / "experiments_index.json"
    experiment_registry_path = out_dir / "experiment_registry.csv"
    image_registry_path = out_dir / "image_registry.csv"
    run_registry_path = out_dir / "run_registry.csv"

    experiment_run_rows_sorted = sorted(
        experiment_run_rows,
        key=lambda r: (
            r.get("experiment_dir") or "",
            r.get("variant_key") or "",
            r.get("run") if isinstance(r.get("run"), int) else 0,
            r.get("generation_id") or "",
        ),
    )

    image_rows_sorted = sorted(
        image_rows,
        key=lambda r: (
            r.get("created_at") or "",
            r.get("manifest_path") or "",
            r.get("title_seq") if isinstance(r.get("title_seq"), int) else 0,
            r.get("generation_id") or "",
        ),
    )

    run_rows_sorted = sorted(
        run_rows,
        key=lambda r: (
            r.get("created_at") or "",
            r.get("generation_id") or "",
        ),
    )

    experiments_payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "artifacts_root": _path_str(artifacts_path, root=root),
        "output_dir": _path_str(out_dir, root=root),
        "experiments": sorted(
            experiment_summaries,
            key=lambda e: (e.get("experiment_dir") or ""),
            reverse=True,
        ),
        "counts": {
            "experiments": len(experiment_summaries),
            "experiment_runs_planned": len(experiment_run_rows),
            "manifest_files": len({row.manifest_path.resolve() for row in manifest_rows}),
            "images_titled": len(manifest_rows),
            "run_index_files": len({Path(v.get("_source_path", "")).resolve() for v in run_index_by_generation.values()}),
            "run_index_entries": len(run_index_by_generation),
            "generation_csv_files": len(generation_csv_paths),
            "generation_csv_rows": len(generation_csv_by_id),
            "runs_in_run_registry": len(run_rows),
        },
        "warnings": {
            "experiment_plan_errors": experiment_plan_errors,
            "runs_index_errors": run_index_errors,
            "titles_manifest_errors": manifest_errors,
            "titles_manifest_duplicate_generation_ids": sorted(set(manifest_duplicates)),
            "generation_csv_errors": generation_csv_errors,
        },
        "registries": {
            "experiment_registry_csv": _path_str(experiment_registry_path, root=root),
            "image_registry_csv": _path_str(image_registry_path, root=root),
            "run_registry_csv": _path_str(run_registry_path, root=root),
        },
    }

    _atomic_write_json(experiments_index_path, experiments_payload)

    _atomic_write_csv(
        experiment_registry_path,
        fieldnames=[
            "experiment_id",
            "experiment_dir",
            "run_id",
            "generation_id",
            "variant_key",
            "run",
            "seed",
            "run_mode",
            "status",
            "created_at",
            "title_seq",
            "title",
            "image_id",
            "image_label",
            "image_path",
            "upscaled_image_path",
            "transcript_path",
            "final_prompt_path",
            "oplog_path",
            "titles_manifest_path",
        ],
        rows=experiment_run_rows_sorted,
    )

    _atomic_write_csv(
        image_registry_path,
        fieldnames=[
            "generation_id",
            "title_seq",
            "title",
            "image_id",
            "image_label",
            "image_path",
            "created_at",
            "seed",
            "model",
            "size",
            "quality",
            "manifest_path",
            "experiment_id",
            "experiment_dir",
            "experiment_run_id",
            "variant_key",
            "planned_plan",
            "planned_variant_name",
            "run_mode",
            "status",
            "transcript_path",
        ],
        rows=image_rows_sorted,
    )

    _atomic_write_csv(
        run_registry_path,
        fieldnames=[
            "generation_id",
            "created_at",
            "seed",
            "status",
            "phase",
            "run_mode",
            "experiment_id",
            "experiment_variant",
            "experiment_notes",
            "experiment_tags",
            "experiment_run_id",
            "variant_key",
            "run",
            "planned_plan",
            "planned_variant_name",
            "title_seq",
            "title",
            "image_id",
            "image_label",
            "image_path",
            "upscaled_image_path",
            "transcript_path",
            "final_prompt_path",
            "oplog_path",
            "run_report_json_path",
            "run_report_html_path",
            "titles_manifest_path",
            "generations_source_path",
            "generations_schema",
            "selected_concepts",
            "final_image_prompt",
            "legacy_output",
        ],
        rows=run_rows_sorted,
    )

    return experiments_payload


def update_artifacts_index_combined(
    *,
    artifacts_roots: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """
    Build a single combined registry from multiple artifact roots (e.g. repo `_artifacts` + Drive store).

    Writes:
    - <output_dir>/experiments_index.json
    - <output_dir>/experiment_registry.csv
    - <output_dir>/image_registry.csv
    - <output_dir>/run_registry.csv
    - plus per-store caches under <output_dir>/_stores/<store>/
    """

    roots = [str(p) for p in artifacts_roots if str(p).strip()]
    if len(roots) < 2:
        raise ValueError("artifacts_roots must include at least 2 roots for a combined index")

    repo_root = _repo_root_from_here().resolve()
    if output_dir is None:
        output_dir = repo_root / "_artifacts" / "index_combined"

    out_dir = Path(output_dir)
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (repo_root / out_dir).resolve()

    used_labels: set[str] = set()
    stores: list[dict[str, Any]] = []
    combined_experiments: list[dict[str, Any]] = []
    combined_experiment_rows: list[dict[str, Any]] = []
    combined_image_rows: list[dict[str, Any]] = []
    combined_run_rows: list[dict[str, Any]] = []

    def _read_csv(path: Path) -> list[dict[str, Any]]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            out: list[dict[str, Any]] = []
            for row in reader:
                if not row:
                    continue
                if None in row:
                    row.pop(None, None)
                out.append({str(k): ("" if v is None else v) for k, v in row.items()})
            return out

    for idx, root_text in enumerate(roots):
        root_path = Path(root_text)
        store_root = root_path.resolve() if root_path.is_absolute() else (repo_root / root_path).resolve()

        label = _guess_store_label(store_root)
        if label in used_labels:
            label = f"{label}_{idx + 1}"
        used_labels.add(label)

        store_out_dir = out_dir / "_stores" / label
        store_payload = update_artifacts_index(
            artifacts_root=str(store_root),
            output_dir=str(store_out_dir),
            repo_root=None,
        )

        stores.append(
            {
                "store": label,
                "store_root": str(store_root),
                "output_dir": str(store_out_dir),
                "counts": store_payload.get("counts") if isinstance(store_payload, dict) else None,
                "warnings": store_payload.get("warnings") if isinstance(store_payload, dict) else None,
            }
        )

        experiment_rows = _read_csv(store_out_dir / "experiment_registry.csv")
        image_rows = _read_csv(store_out_dir / "image_registry.csv")
        run_rows = _read_csv(store_out_dir / "run_registry.csv")

        path_keys_experiment = [
            "experiment_dir",
            "image_path",
            "upscaled_image_path",
            "transcript_path",
            "final_prompt_path",
            "oplog_path",
            "titles_manifest_path",
        ]
        path_keys_image = [
            "image_path",
            "manifest_path",
            "experiment_dir",
            "transcript_path",
        ]
        path_keys_run = [
            "image_path",
            "upscaled_image_path",
            "transcript_path",
            "final_prompt_path",
            "oplog_path",
            "run_report_json_path",
            "run_report_html_path",
            "titles_manifest_path",
            "generations_source_path",
        ]

        combined_experiment_rows.extend(
            _with_store_context(
                row,
                store_label=label,
                store_root=store_root,
                base_dir=store_root,
                path_keys=path_keys_experiment,
            )
            for row in experiment_rows
        )
        combined_image_rows.extend(
            _with_store_context(
                row,
                store_label=label,
                store_root=store_root,
                base_dir=store_root,
                path_keys=path_keys_image,
            )
            for row in image_rows
        )
        combined_run_rows.extend(
            _with_store_context(
                row,
                store_label=label,
                store_root=store_root,
                base_dir=store_root,
                path_keys=path_keys_run,
            )
            for row in run_rows
        )

        experiments = store_payload.get("experiments") if isinstance(store_payload, dict) else None
        if isinstance(experiments, list):
            for exp in experiments:
                if not isinstance(exp, dict):
                    continue
                enriched = {
                    "store": label,
                    "store_root": str(store_root),
                    **exp,
                }
                for key in (
                    "experiment_dir",
                    "plan_path",
                    "plan_full_path",
                    "results_path",
                    "runs_index_path",
                    "titles_manifest_path",
                ):
                    value = enriched.get(key)
                    if isinstance(value, str) and value.strip():
                        enriched[key] = _abspath_from_base(value, base_dir=store_root)
                combined_experiments.append(enriched)

    combined_run_rows = _dedupe_by_key(combined_run_rows, key="generation_id")
    combined_image_rows = _dedupe_by_key(combined_image_rows, key="generation_id")

    combined_experiment_rows_sorted = sorted(
        combined_experiment_rows,
        key=lambda r: (r.get("experiment_dir") or "", r.get("generation_id") or ""),
    )
    combined_image_rows_sorted = sorted(
        combined_image_rows,
        key=lambda r: (r.get("created_at") or "", r.get("generation_id") or ""),
    )
    combined_run_rows_sorted = sorted(
        combined_run_rows,
        key=lambda r: (r.get("created_at") or "", r.get("generation_id") or ""),
    )

    experiments_index_path = out_dir / "experiments_index.json"
    experiment_registry_path = out_dir / "experiment_registry.csv"
    image_registry_path = out_dir / "image_registry.csv"
    run_registry_path = out_dir / "run_registry.csv"

    _atomic_write_csv(
        experiment_registry_path,
        fieldnames=[
            "store",
            "store_root",
            "experiment_id",
            "experiment_dir",
            "run_id",
            "generation_id",
            "variant_key",
            "run",
            "seed",
            "run_mode",
            "status",
            "created_at",
            "title_seq",
            "title",
            "image_id",
            "image_label",
            "image_path",
            "upscaled_image_path",
            "transcript_path",
            "final_prompt_path",
            "oplog_path",
            "titles_manifest_path",
        ],
        rows=combined_experiment_rows_sorted,
    )

    _atomic_write_csv(
        image_registry_path,
        fieldnames=[
            "store",
            "store_root",
            "generation_id",
            "title_seq",
            "title",
            "image_id",
            "image_label",
            "image_path",
            "created_at",
            "seed",
            "model",
            "size",
            "quality",
            "manifest_path",
            "experiment_id",
            "experiment_dir",
            "experiment_run_id",
            "variant_key",
            "planned_plan",
            "planned_variant_name",
            "run_mode",
            "status",
            "transcript_path",
        ],
        rows=combined_image_rows_sorted,
    )

    _atomic_write_csv(
        run_registry_path,
        fieldnames=[
            "store",
            "store_root",
            "generation_id",
            "created_at",
            "seed",
            "status",
            "phase",
            "run_mode",
            "experiment_id",
            "experiment_variant",
            "experiment_notes",
            "experiment_tags",
            "experiment_run_id",
            "variant_key",
            "run",
            "planned_plan",
            "planned_variant_name",
            "title_seq",
            "title",
            "image_id",
            "image_label",
            "image_path",
            "upscaled_image_path",
            "transcript_path",
            "final_prompt_path",
            "oplog_path",
            "run_report_json_path",
            "run_report_html_path",
            "titles_manifest_path",
            "generations_source_path",
            "generations_schema",
            "selected_concepts",
            "final_image_prompt",
            "legacy_output",
        ],
        rows=combined_run_rows_sorted,
    )

    payload = {
        "schema_version": 1,
        "kind": "combined_artifacts_index",
        "generated_at": _utc_now_iso8601(),
        "output_dir": str(out_dir),
        "stores": stores,
        "experiments": combined_experiments,
        "counts": {
            "stores": len(stores),
            "experiments": len(combined_experiments),
            "experiment_rows": len(combined_experiment_rows_sorted),
            "image_rows": len(combined_image_rows_sorted),
            "run_rows": len(combined_run_rows_sorted),
        },
        "registries": {
            "experiment_registry_csv": str(experiment_registry_path),
            "image_registry_csv": str(image_registry_path),
            "run_registry_csv": str(run_registry_path),
        },
    }

    _atomic_write_json(experiments_index_path, payload)
    return payload


def maybe_update_artifacts_index(
    *,
    artifacts_root: str | os.PathLike[str] = "_artifacts",
    output_dir: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
    logger: Any | None = None,
) -> None:
    """
    Best-effort wrapper for `update_artifacts_index()` so callers never fail a run.
    """

    try:
        update_artifacts_index(
            artifacts_root=artifacts_root,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        if logger is not None:
            try:
                logger.info("Updated artifacts index under %s", str(artifacts_root))
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            try:
                logger.exception("Artifacts index update failed: %s", exc)
            except Exception:
                pass
