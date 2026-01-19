from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Ensure project root is importable when running as a loose script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_project.app.generate import run_generation
from image_project.app.experiment_dry_run import write_experiment_plan_full
from image_project.foundation.config_io import load_config
from image_project.framework.artifacts import generate_unique_id
from image_project.framework.artifacts import maybe_update_artifacts_index
from image_project.framework.config import RunConfig
from tools.experiment_manifest import build_pairs_payload, record_pair_error, write_pairs_manifest


RunMode = Literal["full", "prompt_only"]
DataMode = Literal["sample", "config"]


@dataclass(frozen=True)
class PlannedRun:
    variant_id: str
    variant_name: str
    run_index: int
    generation_id: str
    seed: int
    random_token: str
    cfg_dict: dict[str, Any]


def _deep_merge(base: Any, overlay: Any, *, path: str) -> Any:
    if overlay is None:
        return None

    if base is None:
        return overlay

    if isinstance(base, Mapping):
        if not isinstance(overlay, Mapping):
            raise ValueError(
                f"Invalid config overlay merge at {path}: base is mapping but overlay is {type(overlay).__name__}"
            )
        merged: dict[str, Any] = dict(base)
        for key, overlay_value in overlay.items():
            next_path = f"{path}.{key}" if path else str(key)
            if key in base:
                merged[key] = _deep_merge(base[key], overlay_value, path=next_path)
            else:
                merged[key] = overlay_value
        return merged

    if isinstance(base, (list, tuple)):
        if not isinstance(overlay, (list, tuple)):
            raise ValueError(
                f"Invalid config overlay merge at {path}: base is list but overlay is {type(overlay).__name__}"
            )
        return list(overlay)

    if isinstance(overlay, (Mapping, list, tuple)):
        raise ValueError(
            f"Invalid config overlay merge at {path}: base is {type(base).__name__} but overlay is {type(overlay).__name__}"
        )

    return overlay


def _merge_cfg(base_cfg: Mapping[str, Any], *overlays: Mapping[str, Any]) -> dict[str, Any]:
    merged: Any = dict(base_cfg)
    for overlay in overlays:
        merged = _deep_merge(merged, overlay, path="")
    if not isinstance(merged, dict):
        raise TypeError("Merged config must be a mapping")
    return merged


def _default_output_root() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("_artifacts", "experiments", f"{timestamp}_ab_scenespec_json_intermediary")


def _compute_random_token(seed: int) -> str:
    rng = random.Random(int(seed))
    roll = rng.randint(100000, 999999)
    return f"RV-{int(seed)}-{roll}"


def _sample_data_paths() -> tuple[str, str]:
    categories = os.path.join(
        str(PROJECT_ROOT),
        "image_project",
        "impl",
        "current",
        "data",
        "sample",
        "category_list_v1.csv",
    )
    profile = os.path.join(
        str(PROJECT_ROOT),
        "image_project",
        "impl",
        "current",
        "data",
        "sample",
        "user_profile_v4.csv",
    )
    return categories, profile


def build_plan(
    *,
    base_cfg: Mapping[str, Any],
    output_root: str,
    experiment_id: str,
    run_mode: RunMode,
    runs_per_variant: int,
    base_seed: int,
    data_mode: DataMode,
    enable_upscale: bool,
    enable_upload: bool,
) -> list[PlannedRun]:
    if runs_per_variant <= 0:
        raise ValueError("runs_per_variant must be > 0")

    output_root = os.path.abspath(output_root)
    log_dir = os.path.join(output_root, "logs")
    generation_dir = os.path.join(output_root, "generated")
    upscale_dir = os.path.join(output_root, "upscaled")

    stage_prefix = ["ab.random_token", "ab.scene_draft"]

    common_overrides: dict[str, Any] = {
        "run": {"mode": run_mode},
        "image": {
            "log_path": log_dir,
            "generation_path": generation_dir,
            "upscale_path": upscale_dir,
        },
        "prompt": {
            "plan": "custom",
            "refinement": {"policy": "none"},
            "scoring": {"enabled": False},
            "stages": {
                "sequence": [],
                "include": [],
                "exclude": [],
                "overrides": {},
            },
            "output": {"capture_stage": "ab.final_prompt_format"},
            "generations_path": os.path.join(log_dir, "generations_v2.csv"),
            "titles_manifest_path": os.path.join(generation_dir, "titles_manifest.csv"),
        },
        "experiment": {"id": experiment_id},
        "upscale": {"enabled": bool(enable_upscale)},
        "rclone": {"enabled": bool(enable_upload)},
        "context": {"enabled": False},
    }

    if data_mode == "sample":
        categories_path, profile_path = _sample_data_paths()
        common_overrides["prompt"]["categories_path"] = categories_path
        common_overrides["prompt"]["profile_path"] = profile_path

    variants: dict[str, dict[str, Any]] = {
        "A": {
            "variant_name": "prose_refine",
            "stages": ["ab.scene_refine_with_block", "ab.final_prompt_format"],
            "capture_stage": "ab.final_prompt_format",
            "experiment": {
                "variant": "A_prose_refine",
                "notes": "Draft scene -> prose refinement (refinement block) -> strict one-line final prompt.",
                "tags": ["ab", "path:prose", "plan:custom", "mode:" + run_mode],
            },
        },
        "B": {
            "variant_name": "scenespec_json",
            "stages": ["ab.scene_spec_json", "ab.final_prompt_format_from_scenespec"],
            "capture_stage": "ab.final_prompt_format_from_scenespec",
            "experiment": {
                "variant": "B_scenespec_json",
                "notes": "Draft scene -> SceneSpec JSON intermediary -> strict one-line final prompt.",
                "tags": ["ab", "path:scenespec_json", "plan:custom", "mode:" + run_mode],
            },
        },
    }

    planned: list[PlannedRun] = []
    for run_index in range(runs_per_variant):
        seed = int(base_seed) + run_index
        token = _compute_random_token(seed)

        for variant_id in ("A", "B"):
            variant = variants[variant_id]
            stage_sequence = [*stage_prefix, *list(variant["stages"])]

            generation_id = f"{variant_id}{run_index + 1}_{generate_unique_id()}"
            run_overlay = {
                "prompt": {
                    "random_seed": seed,
                    "stages": {"sequence": stage_sequence},
                    "output": {"capture_stage": str(variant["capture_stage"])},
                },
                "experiment": {
                    "variant": variant["experiment"]["variant"],
                    "notes": variant["experiment"]["notes"],
                    "tags": list(variant["experiment"]["tags"]) + [f"run:{run_index + 1}"],
                },
            }

            cfg_dict = _merge_cfg(base_cfg, common_overrides, run_overlay)
            planned.append(
                PlannedRun(
                    variant_id=variant_id,
                    variant_name=str(variant["variant_name"]),
                    run_index=run_index + 1,
                    generation_id=generation_id,
                    seed=seed,
                    random_token=token,
                    cfg_dict=cfg_dict,
                )
            )

    return planned


def _print_plan(plan: Sequence[PlannedRun]) -> None:
    for entry in plan:
        prompt_cfg = entry.cfg_dict.get("prompt") if isinstance(entry.cfg_dict.get("prompt"), dict) else {}
        stage_cfg = prompt_cfg.get("stages") if isinstance(prompt_cfg.get("stages"), dict) else {}
        sequence = stage_cfg.get("sequence") if isinstance(stage_cfg.get("sequence"), list) else []
        capture_cfg = prompt_cfg.get("output") if isinstance(prompt_cfg.get("output"), dict) else {}
        capture = capture_cfg.get("capture_stage")
        print(
            f"{entry.variant_id}{entry.run_index}: generation_id={entry.generation_id} "
            f"seed={entry.seed} token={entry.random_token} capture={capture} stages={sequence}"
        )


_FINAL_PROMPT_KEYS = (
    "SUBJECT",
    "SETTING",
    "ACTION",
    "COMPOSITION",
    "CAMERA",
    "LIGHTING",
    "COLOR",
    "STYLE",
    "TEXT_IN_SCENE",
    "AR",
)

_SPECIFICITY_MIN_WORDS: dict[str, int] = {
    "SUBJECT": 5,
    "SETTING": 6,
    "ACTION": 5,
    "COMPOSITION": 5,
    "CAMERA": 3,
    "LIGHTING": 3,
    "COLOR": 3,
    "STYLE": 3,
}

_CLICHE_PHRASES = (
    "masterpiece",
    "highly detailed",
    "ultra detailed",
    "ultra-detailed",
    "ultra realistic",
    "ultra-realistic",
    "hyper realistic",
    "hyper-realistic",
    "photorealistic",
    "cinematic",
    "award-winning",
    "epic",
    "stunning",
    "breathtaking",
    "8k",
    "hdr",
)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _word_count(text: str) -> int:
    return len([part for part in re.split(r"\s+", text.strip()) if part])


def _cliche_hits(text: str) -> list[str]:
    haystack = text.lower()
    hits: list[str] = []
    for phrase in _CLICHE_PHRASES:
        if phrase in haystack:
            hits.append(phrase)
    return hits


def _is_generic_subject(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if not cleaned:
        return True
    generic = (
        "a person",
        "person",
        "someone",
        "somebody",
        "a man",
        "a woman",
        "a figure",
        "a character",
        "a human",
        "a scene",
        "a landscape",
        "a city",
        "a building",
        "a creature",
        "an animal",
    )
    if cleaned in generic:
        return True
    return any(cleaned.startswith(prefix + " ") for prefix in generic)


def _parse_final_prompt_line(*, final_prompt: Any, token: str) -> dict[str, Any]:
    raw = _to_text(final_prompt)
    issues: list[dict[str, str]] = []
    parsed_fields: dict[str, str] = {}

    if not raw:
        return {
            "ok": False,
            "issues": [{"code": "final_prompt_missing", "message": "Final prompt is empty or missing"}],
        }

    if "\n" in raw or "\r" in raw:
        issues.append({"code": "final_prompt_multiline", "message": "Final prompt contains newline(s)"})

    parts = [part.strip() for part in raw.split(" | ")]
    if len(parts) != len(_FINAL_PROMPT_KEYS):
        issues.append(
            {
                "code": "final_prompt_segment_count",
                "message": f"Expected {len(_FINAL_PROMPT_KEYS)} segments but got {len(parts)}",
            }
        )

    for idx, key in enumerate(_FINAL_PROMPT_KEYS):
        if idx >= len(parts):
            break
        segment = parts[idx]
        if "=" not in segment:
            issues.append({"code": "final_prompt_bad_segment", "message": f"Missing '=' in segment: {segment!r}"})
            continue
        seg_key, seg_value = segment.split("=", 1)
        seg_key = seg_key.strip()
        seg_value = seg_value.strip()
        if seg_key != key:
            issues.append(
                {
                    "code": "final_prompt_bad_key_order",
                    "message": f"Expected key {key} at position {idx + 1} but got {seg_key}",
                }
            )
        parsed_fields[seg_key] = seg_value

    missing_keys = [key for key in _FINAL_PROMPT_KEYS if key not in parsed_fields]
    if missing_keys:
        issues.append(
            {
                "code": "final_prompt_missing_keys",
                "message": f"Missing keys: {', '.join(missing_keys)}",
            }
        )

    empty_keys: list[str] = []
    placeholder_keys: list[str] = []
    word_counts: dict[str, int] = {}
    specificity_violations: dict[str, int] = {}

    for key, value in parsed_fields.items():
        cleaned = value.strip()
        if not cleaned:
            empty_keys.append(key)
        if "<...>" in cleaned or cleaned in {"<...>", "..."}:
            placeholder_keys.append(key)
        word_counts[key] = _word_count(cleaned.strip('"'))
        min_words = _SPECIFICITY_MIN_WORDS.get(key)
        if min_words is not None and word_counts[key] < min_words:
            specificity_violations[key] = word_counts[key]

    if empty_keys:
        issues.append({"code": "final_prompt_empty_fields", "message": f"Empty fields: {', '.join(empty_keys)}"})
    if placeholder_keys:
        issues.append(
            {
                "code": "final_prompt_placeholders",
                "message": f"Placeholder values present: {', '.join(placeholder_keys)}",
            }
        )

    text_in_scene_raw = parsed_fields.get("TEXT_IN_SCENE", "")
    text_in_scene_value = text_in_scene_raw
    if text_in_scene_raw.startswith('"') and text_in_scene_raw.endswith('"') and len(text_in_scene_raw) >= 2:
        text_in_scene_value = text_in_scene_raw[1:-1]
    else:
        issues.append(
            {
                "code": "final_prompt_text_in_scene_quotes",
                "message": "TEXT_IN_SCENE value must be quoted like \"...\"",
            }
        )

    token_ok = text_in_scene_value == token
    if not token_ok:
        issues.append(
            {
                "code": "final_prompt_token_mismatch",
                "message": f'TEXT_IN_SCENE "{text_in_scene_value}" does not match token "{token}"',
            }
        )

    ar_ok = parsed_fields.get("AR") == "16:9"
    if not ar_ok:
        issues.append({"code": "final_prompt_ar_mismatch", "message": 'AR must equal "16:9"'})

    cliche_hits = _cliche_hits(raw)

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "fields": parsed_fields,
        "word_counts": word_counts,
        "specificity": {
            "min_words": dict(_SPECIFICITY_MIN_WORDS),
            "violations": specificity_violations,
        },
        "cliche_hits": cliche_hits,
        "token_expected": token,
        "text_in_scene": text_in_scene_value,
        "token_ok": token_ok,
        "ar_ok": ar_ok,
    }


def _parse_scenespec_json(*, spec_json: Any, token: str) -> dict[str, Any] | None:
    raw = _to_text(spec_json)
    if not raw:
        return None

    required_keys = (
        "subject",
        "setting",
        "action",
        "composition",
        "camera",
        "lighting",
        "color",
        "style",
        "text_in_scene",
        "must_keep",
        "avoid",
    )

    issues: list[dict[str, str]] = []
    parsed: Any
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "issues": [{"code": "scenespec_json_parse_error", "message": str(exc)}],
            "raw": raw,
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "issues": [{"code": "scenespec_json_not_object", "message": "SceneSpec must be a JSON object"}],
        }

    missing = [key for key in required_keys if key not in parsed]
    if missing:
        issues.append({"code": "scenespec_missing_keys", "message": f"Missing keys: {', '.join(missing)}"})

    extra = [key for key in parsed.keys() if key not in required_keys]
    if extra:
        issues.append({"code": "scenespec_extra_keys", "message": f"Extra keys present: {', '.join(extra)}"})

    empty_fields: list[str] = []
    type_errors: list[str] = []
    word_counts: dict[str, int] = {}

    for key in required_keys:
        if key not in parsed:
            continue
        value = parsed[key]
        if key in {"must_keep", "avoid"}:
            if not isinstance(value, list):
                type_errors.append(f"{key} (expected list)")
                continue
            cleaned_items = [str(item).strip() for item in value if str(item).strip()]
            if len(cleaned_items) < 3:
                empty_fields.append(key)
            parsed[key] = cleaned_items
            continue
        if not isinstance(value, str):
            type_errors.append(f"{key} (expected string)")
            continue
        cleaned = value.strip()
        if not cleaned:
            empty_fields.append(key)
        word_counts[key] = _word_count(cleaned)
        parsed[key] = cleaned

    if empty_fields:
        issues.append({"code": "scenespec_empty_fields", "message": f"Empty/too-short fields: {', '.join(empty_fields)}"})
    if type_errors:
        issues.append({"code": "scenespec_type_errors", "message": f"Type issues: {', '.join(type_errors)}"})

    subject_text = str(parsed.get("subject", "")).strip()
    subject_generic = _is_generic_subject(subject_text)
    if subject_generic:
        issues.append({"code": "scenespec_generic_subject", "message": "subject appears generic"})

    token_value = str(parsed.get("text_in_scene", "")).strip()
    token_ok = token_value == token
    if not token_ok:
        issues.append(
            {
                "code": "scenespec_token_mismatch",
                "message": f'text_in_scene "{token_value}" does not match token "{token}"',
            }
        )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "parsed": parsed,
        "word_counts": word_counts,
        "token_expected": token,
        "token_ok": token_ok,
        "subject_generic": subject_generic,
    }


def _compute_auto_checks(*, variant_id: str, outputs: Mapping[str, Any], token: str) -> dict[str, Any]:
    scene_draft = _to_text(outputs.get("ab_scene_draft"))
    scene_refined = _to_text(outputs.get("ab_scene_refined"))
    scenespec_json = outputs.get("ab_scene_spec_json")
    final_prompt = outputs.get("image_prompt")

    final_prompt_checks = _parse_final_prompt_line(final_prompt=final_prompt, token=token)
    scenespec_checks = _parse_scenespec_json(spec_json=scenespec_json, token=token) if variant_id == "B" else None

    token_presence = {
        "in_scene_draft": token in scene_draft if scene_draft else False,
        "in_scene_refined": token in scene_refined if scene_refined else False,
        "in_scenespec_raw": token in _to_text(scenespec_json) if scenespec_json is not None else False,
    }

    return {
        "final_prompt": final_prompt_checks,
        "scenespec_json": scenespec_checks,
        "token_presence": token_presence,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_experiment_ab_scenespec_json_intermediary",
        description="Run an A/B prompt experiment testing whether a SceneSpec JSON intermediary improves strict one-line prompts.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional explicit pipeline config path (otherwise uses config/config.yaml + config/config.local.yaml).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for experiment artifacts (logs/generated/upscaled). Defaults under ./_artifacts/experiments/.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment id written into transcript metadata. Defaults to a timestamped id.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Run pairs to execute (A and B are run for each index).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prompt_only",
        choices=("prompt_only", "full"),
        help="Run mode: prompt_only skips image/upscale/upload; full generates images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed used to derive per-run prompt.random_seed values.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="config",
        choices=("sample", "config"),
        help="Data source for categories/profile: config uses paths from the loaded config; sample uses repo sample CSVs.",
    )
    parser.add_argument(
        "--enable-upscale",
        action="store_true",
        help="Enable upscaling (uses the base config's upscale settings).",
    )
    parser.add_argument(
        "--enable-upload",
        action="store_true",
        help="Enable rclone upload (uses the base config's rclone settings).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and print the run plan without calling any AI or generating images.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining runs if a run fails.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.runs <= 0:
        raise SystemExit("--runs must be > 0")

    output_root = args.output_root or _default_output_root()
    output_root = os.path.abspath(output_root)

    experiment_id = (args.experiment_id or "").strip()
    if not experiment_id:
        experiment_id = "exp_ab_scenespec_json_intermediary_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    base_seed = int(args.seed) if args.seed is not None else int(datetime.now().timestamp())

    base_cfg, cfg_meta = load_config(config_path=args.config_path)

    plan = build_plan(
        base_cfg=base_cfg,
        output_root=output_root,
        experiment_id=experiment_id,
        run_mode=args.mode,  # type: ignore[arg-type]
        runs_per_variant=args.runs,
        base_seed=base_seed,
        data_mode=args.data,  # type: ignore[arg-type]
        enable_upscale=bool(args.enable_upscale),
        enable_upload=bool(args.enable_upload),
    )

    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, "experiment_plan.json")
    pairs_payload = build_pairs_payload(plan, experiment_id=experiment_id, run_mode=str(args.mode))
    pairs_path = write_pairs_manifest(output_root, pairs_payload)
    print(f"Wrote pairs manifest to {pairs_path}")

    validation_errors: list[dict[str, Any]] = []
    for entry in plan:
        try:
            RunConfig.from_dict(entry.cfg_dict)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(
                {
                    "variant": entry.variant_id,
                    "run": entry.run_index,
                    "generation_id": entry.generation_id,
                    "error": str(exc),
                }
            )

    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "base_seed": base_seed,
        "config_meta": cfg_meta,
        "output_root": output_root,
        "run_mode": args.mode,
        "runs_per_variant": args.runs,
        "data_mode": args.data,
        "enable_upscale": bool(args.enable_upscale),
        "enable_upload": bool(args.enable_upload),
        "validation_errors": validation_errors,
        "planned_runs": [
            {
                "variant": entry.variant_id,
                "variant_name": entry.variant_name,
                "run": entry.run_index,
                "generation_id": entry.generation_id,
                "seed": entry.seed,
                "random_token": entry.random_token,
            }
            for entry in plan
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    full_plan_path = os.path.join(output_root, "experiment_plan_full.json")
    write_experiment_plan_full(
        full_plan_path,
        summary_payload=payload,
        runs=[
            {
                "variant": entry.variant_id,
                "variant_name": entry.variant_name,
                "run": entry.run_index,
                "generation_id": entry.generation_id,
                "seed": entry.seed,
                "random_token": entry.random_token,
                "cfg_dict": entry.cfg_dict,
            }
            for entry in plan
        ],
    )

    _print_plan(plan)
    print(f"Wrote experiment plan to {summary_path}")
    print(f"Wrote expanded experiment plan to {full_plan_path}")

    if validation_errors:
        print("Config validation errors:")
        print(json.dumps(validation_errors, ensure_ascii=False, indent=2))
        maybe_update_artifacts_index(repo_root=str(PROJECT_ROOT))
        return 2

    if args.dry_run:
        maybe_update_artifacts_index(repo_root=str(PROJECT_ROOT))
        return 0

    results: list[dict[str, Any]] = []
    failures = 0
    attempted: set[tuple[int, str]] = set()

    for entry in plan:
        attempted.add((entry.run_index, entry.variant_id))
        try:
            ctx = run_generation(entry.cfg_dict, generation_id=entry.generation_id, config_meta=cfg_meta)
            outputs = ctx.outputs if isinstance(ctx.outputs, dict) else {}
            results.append(
                {
                    "variant": entry.variant_id,
                    "run": entry.run_index,
                    "generation_id": entry.generation_id,
                    "seed": entry.seed,
                    "random_token": entry.random_token,
                    "status": "success",
                    "image_path": getattr(ctx, "image_path", None),
                    "final_prompt": outputs.get("image_prompt"),
                    "auto_checks": _compute_auto_checks(
                        variant_id=entry.variant_id,
                        outputs=outputs,
                        token=entry.random_token,
                    ),
                    "outputs": {"prompt_pipeline": outputs.get("prompt_pipeline")},
                    "error": ctx.error,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record_pair_error(
                pairs_payload,
                run_index=entry.run_index,
                variant_id=entry.variant_id,
                error={"type": exc.__class__.__name__, "message": str(exc)},
            )
            results.append(
                {
                    "variant": entry.variant_id,
                    "run": entry.run_index,
                    "generation_id": entry.generation_id,
                    "seed": entry.seed,
                    "random_token": entry.random_token,
                    "status": "error",
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                }
            )
            if not args.continue_on_error:
                break

    if failures and not args.continue_on_error:
        for entry in plan:
            key = (entry.run_index, entry.variant_id)
            if key in attempted:
                continue
            record_pair_error(
                pairs_payload,
                run_index=entry.run_index,
                variant_id=entry.variant_id,
                error={"type": "Skipped", "message": "Skipped due to an earlier failure"},
            )

    results_path = os.path.join(output_root, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "results": results}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    pairs_path = write_pairs_manifest(output_root, pairs_payload)
    print(f"Wrote pairs manifest to {pairs_path}")
    print(f"Wrote experiment results to {results_path}")
    maybe_update_artifacts_index(repo_root=str(PROJECT_ROOT))
    return 0 if failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
