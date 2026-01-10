from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .evolution import EvolutionThresholds
from .render_html import render_compare_html
from .report_builder import RunLoadError, build_report, diff_reports
from .report_model import RunInputs


class PairsManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PairSelection:
    all_pairs: bool = False
    run_index: int | None = None

    def validate(self) -> None:
        if self.all_pairs and self.run_index is not None:
            raise ValueError("Select either all_pairs or run_index, not both")
        if not self.all_pairs and self.run_index is None:
            raise ValueError("Select either all_pairs or a specific run_index")
        if self.run_index is not None and self.run_index <= 0:
            raise ValueError("run_index must be >= 1")


def _load_pairs_manifest(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing pairs manifest: {path}. "
            "Run the A/B experiment runner to generate <experiment_dir>/pairs.json."
        )

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise PairsManifestError("pairs.json must contain a JSON object at the top level")

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise PairsManifestError(f"Unsupported pairs.json schema_version: {schema_version!r} (expected 1)")

    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise PairsManifestError("pairs.json missing required key: pairs (expected list)")

    normalized_pairs: list[dict[str, Any]] = []
    for idx, raw in enumerate(pairs):
        if not isinstance(raw, Mapping):
            raise PairsManifestError(f"pairs[{idx}] must be an object")
        run_index = raw.get("run_index")
        if not isinstance(run_index, int) or run_index <= 0:
            raise PairsManifestError(f"pairs[{idx}].run_index must be an int >= 1")

        a_id = raw.get("a_generation_id")
        b_id = raw.get("b_generation_id")
        if not isinstance(a_id, str) or not a_id.strip():
            raise PairsManifestError(f"pairs[{idx}].a_generation_id must be a non-empty string")
        if not isinstance(b_id, str) or not b_id.strip():
            raise PairsManifestError(f"pairs[{idx}].b_generation_id must be a non-empty string")

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise PairsManifestError(f"pairs[{idx}].metadata must be an object if set")

        normalized_pairs.append(
            {
                "run_index": run_index,
                "a_generation_id": a_id.strip(),
                "b_generation_id": b_id.strip(),
                "metadata": dict(metadata),
            }
        )

    normalized_pairs.sort(key=lambda item: item["run_index"])
    return {**dict(payload), "pairs": normalized_pairs}


def _default_logs_dir(experiment_dir: str) -> str:
    candidate = os.path.join(experiment_dir, "logs")
    if os.path.isdir(candidate):
        return candidate
    return experiment_dir


def _artifact_paths(logs_dir: str, generation_id: str) -> tuple[str | None, str | None]:
    oplog = os.path.join(logs_dir, f"{generation_id}_oplog.log")
    transcript = os.path.join(logs_dir, f"{generation_id}_transcript.json")
    return (oplog if os.path.exists(oplog) else None, transcript if os.path.exists(transcript) else None)


def compare_experiment_pairs(
    *,
    experiment_dir: str,
    output_dir: str,
    selection: PairSelection,
    logs_dir: str | None = None,
    best_effort: bool = False,
    enable_evolution: bool = True,
    evolution_thresholds: EvolutionThresholds | None = None,
    print_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    selection.validate()

    experiment_dir_abs = os.path.abspath(experiment_dir)
    output_dir_abs = os.path.abspath(output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    pairs_path = os.path.join(experiment_dir_abs, "pairs.json")
    manifest = _load_pairs_manifest(pairs_path)
    experiment_id = manifest.get("experiment_id")

    raw_pairs = list(manifest["pairs"])
    if selection.run_index is not None:
        selected_pairs = [pair for pair in raw_pairs if pair["run_index"] == selection.run_index]
        if not selected_pairs:
            available = ", ".join(str(pair["run_index"]) for pair in raw_pairs) or "<none>"
            raise ValueError(
                f"pairs.json contains no pair with run_index={selection.run_index} (available: {available})"
            )
    else:
        selected_pairs = raw_pairs

    effective_logs_dir = os.path.abspath(logs_dir) if logs_dir else _default_logs_dir(experiment_dir_abs)

    results: list[dict[str, Any]] = []
    failed = 0

    for pair in selected_pairs:
        run_index = int(pair["run_index"])
        a_id = str(pair["a_generation_id"])
        b_id = str(pair["b_generation_id"])

        prefix = f"run_index={run_index} A={a_id} B={b_id}"
        oplog_a, transcript_a = _artifact_paths(effective_logs_dir, a_id)
        oplog_b, transcript_b = _artifact_paths(effective_logs_dir, b_id)

        if not oplog_a or not transcript_a or not oplog_b or not transcript_b:
            missing: list[str] = []
            if not oplog_a:
                missing.append(f"{a_id}_oplog.log")
            if not transcript_a:
                missing.append(f"{a_id}_transcript.json")
            if not oplog_b:
                missing.append(f"{b_id}_oplog.log")
            if not transcript_b:
                missing.append(f"{b_id}_transcript.json")

            failed += 1
            msg = f"{prefix} FAILED (missing artifacts in logs_dir={effective_logs_dir}: {', '.join(missing)})"
            if print_fn:
                print_fn(f"run_review: {msg}")
            results.append(
                {
                    "run_index": run_index,
                    "a_generation_id": a_id,
                    "b_generation_id": b_id,
                    "status": "error",
                    "error": msg,
                }
            )
            continue

        try:
            report_a = build_report(
                RunInputs(a_id, oplog_path=oplog_a, transcript_path=transcript_a),
                best_effort=best_effort,
                enable_evolution=enable_evolution,
                evolution_thresholds=evolution_thresholds,
            )
            report_b = build_report(
                RunInputs(b_id, oplog_path=oplog_b, transcript_path=transcript_b),
                best_effort=best_effort,
                enable_evolution=enable_evolution,
                evolution_thresholds=evolution_thresholds,
            )
            diff = diff_reports(report_a, report_b)

            html_filename = f"{a_id}_vs_{b_id}_run_compare.html"
            json_filename = f"{a_id}_vs_{b_id}_run_compare.json"
            html_out = os.path.join(output_dir_abs, html_filename)
            json_out = os.path.join(output_dir_abs, json_filename)
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

            if print_fn:
                print_fn(f"run_review: {prefix} OK -> {html_filename}")
            results.append(
                {
                    "run_index": run_index,
                    "a_generation_id": a_id,
                    "b_generation_id": b_id,
                    "status": "success",
                    "html": html_filename,
                    "json": json_filename,
                }
            )
        except (FileNotFoundError, RunLoadError) as exc:
            failed += 1
            msg = f"{prefix} FAILED ({exc})"
            if print_fn:
                print_fn(f"run_review: {msg}")
            results.append(
                {
                    "run_index": run_index,
                    "a_generation_id": a_id,
                    "b_generation_id": b_id,
                    "status": "error",
                    "error": msg,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            msg = f"{prefix} FAILED ({exc.__class__.__name__}: {exc})"
            if print_fn:
                print_fn(f"run_review: {msg}")
            results.append(
                {
                    "run_index": run_index,
                    "a_generation_id": a_id,
                    "b_generation_id": b_id,
                    "status": "error",
                    "error": msg,
                }
            )

    results.sort(key=lambda item: int(item.get("run_index", 0)))
    index_path = os.path.join(output_dir_abs, "index.html")
    summary_path = os.path.join(output_dir_abs, "summary.json")

    def link_for(item: Mapping[str, Any]) -> str | None:
        html = item.get("html")
        if isinstance(html, str) and html:
            return os.path.basename(html)
        return None

    index_lines: list[str] = []
    index_lines.append("<!doctype html>")
    index_lines.append("<html><head><meta charset='utf-8'/>")
    index_lines.append("<title>Experiment Compare Index</title>")
    index_lines.append(
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "li{margin:8px 0} .ok{color:#0a0} .fail{color:#a00} code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>"
    )
    index_lines.append("</head><body>")
    index_lines.append("<h1>Experiment Compare Index</h1>")
    if isinstance(experiment_id, str) and experiment_id.strip():
        index_lines.append(f"<p>experiment_id: <code>{experiment_id.strip()}</code></p>")
    index_lines.append("<ol>")
    for item in results:
        run_index = item.get("run_index")
        a_id = item.get("a_generation_id")
        b_id = item.get("b_generation_id")
        status = item.get("status")
        css = "ok" if status == "success" else "fail"
        link = link_for(item)
        if link:
            index_lines.append(
                f"<li class='{css}'>run {run_index}: <code>{a_id}</code> vs <code>{b_id}</code> "
                f"- <a href='{link}'>report</a></li>"
            )
        else:
            error = item.get("error", "unknown error")
            index_lines.append(
                f"<li class='{css}'>run {run_index}: <code>{a_id}</code> vs <code>{b_id}</code> "
                f"- FAILED ({error})</li>"
            )
    index_lines.append("</ol>")
    index_lines.append("</body></html>")

    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(index_lines) + "\n")

    try:
        logs_dir_rel = os.path.relpath(effective_logs_dir, experiment_dir_abs)
    except ValueError:
        logs_dir_rel = effective_logs_dir

    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "pairs_path": "pairs.json",
        "logs_dir": logs_dir_rel,
        "selected": {"all": selection.run_index is None, "run_index": selection.run_index},
        "counts": {
            "total": len(results),
            "success": sum(1 for item in results if item.get("status") == "success"),
            "failed": failed,
        },
        "results": results,
        "index_html": "index.html",
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    if print_fn:
        print_fn(f"run_review: Wrote {index_path}")
        print_fn(f"run_review: Wrote {summary_path}")

    return summary
