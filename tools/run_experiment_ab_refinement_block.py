from __future__ import annotations

import argparse
import json
import os
import random
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
from image_project.foundation.config_io import load_config
from image_project.framework.artifacts import generate_unique_id
from image_project.framework.config import RunConfig


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
    return os.path.join("_artifacts", "experiments", f"{timestamp}_ab_refinement_block")


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
    stage_suffix = ["ab.final_prompt_format"]

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
            "variant_name": "no_refinement_block",
            "refine_stage": "ab.scene_refine_no_block",
            "experiment": {
                "variant": "A_no_refinement_block",
                "notes": "Middle prompt is a minimal refinement instruction set.",
                "tags": ["ab", "refinement_block:no", "plan:custom", "mode:" + run_mode],
            },
        },
        "B": {
            "variant_name": "with_refinement_block",
            "refine_stage": "ab.scene_refine_with_block",
            "experiment": {
                "variant": "B_with_refinement_block",
                "notes": "Middle prompt includes an explicit refinement block checklist.",
                "tags": ["ab", "refinement_block:yes", "plan:custom", "mode:" + run_mode],
            },
        },
    }

    planned: list[PlannedRun] = []
    for run_index in range(runs_per_variant):
        seed = int(base_seed) + run_index
        token = _compute_random_token(seed)

        for variant_id in ("A", "B"):
            variant = variants[variant_id]
            stage_sequence = [*stage_prefix, str(variant["refine_stage"]), *stage_suffix]

            generation_id = f"{variant_id}{run_index + 1}_{generate_unique_id()}"
            run_overlay = {
                "prompt": {
                    "random_seed": seed,
                    "stages": {"sequence": stage_sequence},
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
        print(
            f"{entry.variant_id}{entry.run_index}: generation_id={entry.generation_id} "
            f"seed={entry.seed} token={entry.random_token} stages={sequence}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_experiment_ab_refinement_block",
        description="Run an A/B prompt-only experiment testing whether a refinement block helps the middle prompt.",
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
        default="sample",
        choices=("sample", "config"),
        help="Data source for categories/profile: sample uses repo sample CSVs; config uses paths from the loaded config.",
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
        experiment_id = "exp_ab_refinement_block_" + datetime.now().strftime("%Y%m%d_%H%M%S")

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

    _print_plan(plan)
    print(f"Wrote experiment plan to {summary_path}")

    if validation_errors:
        print("Config validation errors:")
        print(json.dumps(validation_errors, ensure_ascii=False, indent=2))
        return 2

    if args.dry_run:
        return 0

    results: list[dict[str, Any]] = []
    failures = 0

    for entry in plan:
        try:
            ctx = run_generation(entry.cfg_dict, generation_id=entry.generation_id, config_meta=cfg_meta)
            results.append(
                {
                    "variant": entry.variant_id,
                    "run": entry.run_index,
                    "generation_id": entry.generation_id,
                    "seed": entry.seed,
                    "random_token": entry.random_token,
                    "status": "success",
                    "image_path": getattr(ctx, "image_path", None),
                    "final_prompt": ctx.outputs.get("image_prompt"),
                    "outputs": {"prompt_pipeline": ctx.outputs.get("prompt_pipeline")},
                    "error": ctx.error,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
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

    results_path = os.path.join(output_root, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "results": results}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote experiment results to {results_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

