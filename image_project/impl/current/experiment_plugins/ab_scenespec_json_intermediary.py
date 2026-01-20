from __future__ import annotations

"""A/B experiment: prose refinement vs SceneSpec JSON intermediary."""

import os
import random
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from image_project.foundation.config_io import deep_merge, find_repo_root
from image_project.framework.artifacts import generate_unique_id
from image_project.framework.runtime import RunContext
from image_project.impl.current.experiments import ExperimentBase, RunSpec, register_experiment

RunMode = Literal["full", "prompt_only"]
DataMode = Literal["sample", "config"]


def _merge_cfg(base_cfg: Mapping[str, Any], *overlays: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge overlays onto a base mapping."""

    merged: Any = dict(base_cfg)
    for overlay in overlays:
        merged = deep_merge(merged, overlay, path="")
    if not isinstance(merged, dict):
        raise TypeError("Merged config must be a mapping")
    return merged


def _compute_random_token(seed: int) -> str:
    """Compute the deterministic random token used by `ab.random_token`."""

    rng = random.Random(int(seed))
    roll = rng.randint(100000, 999999)
    return f"RV-{int(seed)}-{roll}"


def _sample_data_paths() -> tuple[str, str]:
    """Return repo-local sample CSV paths for categories and user profile."""

    repo_root = find_repo_root()
    categories = os.path.join(
        repo_root,
        "image_project",
        "impl",
        "current",
        "data",
        "sample",
        "category_list_v1.csv",
    )
    profile = os.path.join(
        repo_root,
        "image_project",
        "impl",
        "current",
        "data",
        "sample",
        "user_profile_v4.csv",
    )
    for path, label in ((categories, "sample categories"), (profile, "sample profile")):
        if not os.path.exists(path):
            raise ValueError(f"Missing {label} CSV for --data sample: {path}")
        if os.path.isdir(path):
            raise ValueError(f"{label} CSV path is a directory: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"{label} CSV is empty: {path}")
    return categories, profile


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
    """Best-effort convert a value to a stripped string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _word_count(text: str) -> int:
    """Count whitespace-separated tokens in text."""

    return len([part for part in re.split(r"\s+", text.strip()) if part])


def _cliche_hits(text: str) -> list[str]:
    """Return cliché phrase hits in a case-insensitive scan."""

    haystack = text.lower()
    hits: list[str] = []
    for phrase in _CLICHE_PHRASES:
        if phrase in haystack:
            hits.append(phrase)
    return hits


def _is_generic_subject(text: str) -> bool:
    """Heuristic: identify overly generic subject values."""

    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if not cleaned:
        return True
    generic = (
        "a person",
        "a man",
        "a woman",
        "a child",
        "someone",
        "a figure",
        "a portrait",
        "a scene",
        "an object",
        "a thing",
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
    """Parse and validate a strict one-line final prompt template."""

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
                "message": 'TEXT_IN_SCENE value must be quoted like "..."',
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
        "token_expected": token,
        "token_ok": token_ok,
        "ar_ok": ar_ok,
        "cliches": cliche_hits,
    }


def _parse_scenespec_json(*, spec_json: Any, token: str) -> dict[str, Any] | None:
    """Validate the SceneSpec JSON intermediary stage output."""

    raw = _to_text(spec_json)
    if not raw:
        return {
            "ok": False,
            "issues": [{"code": "scenespec_missing", "message": "SceneSpec JSON is empty or missing"}],
        }

    issues: list[dict[str, str]] = []
    parsed: dict[str, Any] = {}

    try:
        import json

        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "issues": [{"code": "scenespec_invalid_json", "message": f"Invalid JSON: {exc}"}],
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "issues": [{"code": "scenespec_not_object", "message": "SceneSpec JSON must be an object"}],
        }

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
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        issues.append({"code": "scenespec_missing_keys", "message": f"Missing keys: {', '.join(missing)}"})

    empty_fields: list[str] = []
    type_errors: list[str] = []
    word_counts: dict[str, int] = {}

    for key in required_keys:
        if key not in parsed:
            continue
        value = parsed.get(key)
        if key in {"must_keep", "avoid"}:
            if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
                type_errors.append(f"{key} (expected non-empty list[str])")
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


def _compute_auto_checks(*, variant: str, outputs: Mapping[str, Any], token: str) -> dict[str, Any]:
    """Compute structured checks for strict prompt formatting and token usage."""

    scene_draft = _to_text(outputs.get("ab_scene_draft"))
    scene_refined = _to_text(outputs.get("ab_scene_refined"))
    scenespec_json = outputs.get("ab_scene_spec_json")
    final_prompt = outputs.get("image_prompt")

    final_prompt_checks = _parse_final_prompt_line(final_prompt=final_prompt, token=token)
    scenespec_checks = _parse_scenespec_json(spec_json=scenespec_json, token=token) if variant == "B" else None

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


@register_experiment
class AbSceneSpecJsonIntermediaryExperiment(ExperimentBase):
    """Run pairs A/B to compare prose refinement vs SceneSpec JSON."""

    name = "ab_scenespec_json_intermediary"
    summary = "A/B: prose refinement path vs SceneSpec JSON intermediary path for strict one-line prompts."
    default_run_mode: RunMode = "prompt_only"

    def add_cli_args(self, parser: Any) -> None:
        """Register CLI args for this experiment."""

        parser.add_argument(
            "--runs",
            type=int,
            default=5,
            help="Run pairs to execute (A and B are run for each index).",
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
            help="Data source for categories/profile: config uses the loaded config; sample uses repo sample CSVs.",
        )

    def build_plan(
        self,
        *,
        base_cfg: Mapping[str, Any],
        runner_cfg: Mapping[str, Any],
        output_root: str,
        experiment_id: str,
        run_mode: str,
        cli_args: Mapping[str, Any],
    ) -> tuple[list[RunSpec], dict[str, Any]]:
        """Build an A/B plan with shared per-index seeds/tokens."""

        _ = (runner_cfg, output_root, experiment_id)  # unused; retained for interface consistency

        runs_per_variant = int(cli_args.get("runs", 0) or 0)
        if runs_per_variant <= 0:
            raise ValueError("--runs must be > 0")

        base_seed = int(cli_args.get("seed")) if cli_args.get("seed") is not None else int(datetime.now().timestamp())
        data_mode: DataMode = str(cli_args.get("data") or "config")  # type: ignore[assignment]
        if data_mode not in ("sample", "config"):
            raise ValueError(f"Unknown --data: {data_mode!r} (expected: sample|config)")

        stage_prefix = ["ab.random_token", "ab.scene_draft"]

        common_overrides: dict[str, Any] = {
            "prompt": {
                "plan": "custom",
                "stages": {
                    "sequence": [],
                    "include": [],
                    "exclude": [],
                    "overrides": {},
                },
                "output": {"capture_stage": "ab.final_prompt_format"},
            },
            "context": {"enabled": False},
        }

        if data_mode == "sample":
            categories_path, profile_path = _sample_data_paths()
            common_overrides.setdefault("prompt", {})
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

        planned: list[RunSpec] = []
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
                    RunSpec(
                        variant=str(variant_id),
                        variant_name=str(variant["variant_name"]),
                        run=run_index + 1,
                        generation_id=generation_id,
                        seed=seed,
                        cfg_dict=cfg_dict,
                        meta={"random_token": token, "data_mode": data_mode},
                    )
                )

        plan_meta = {
            "base_seed": base_seed,
            "runs_per_variant": runs_per_variant,
            "data_mode": data_mode,
        }
        return planned, plan_meta

    def analyze_run(self, *, run_spec: RunSpec, ctx: RunContext) -> dict[str, Any] | None:
        """Attach strict-format auto-checks into experiment results."""

        outputs = ctx.outputs if isinstance(ctx.outputs, dict) else {}
        token = str(run_spec.meta.get("random_token") or "").strip()
        return _compute_auto_checks(variant=run_spec.variant, outputs=outputs, token=token)

    def build_pairs_manifest(
        self,
        plan: list[RunSpec],
        *,
        experiment_id: str,
        run_mode: str,
    ) -> dict[str, Any] | None:
        """Return an A/B pairing manifest for run-review."""

        from image_project.framework.experiment_manifest import build_pairs_payload  # noqa: PLC0415

        return build_pairs_payload(plan, experiment_id=experiment_id, run_mode=run_mode)
