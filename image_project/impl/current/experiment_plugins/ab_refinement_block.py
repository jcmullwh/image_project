from __future__ import annotations

"""A/B experiment: refinement stage with/without an explicit refinement block."""

import os
import random
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from image_project.foundation.config_io import deep_merge, find_repo_root
from image_project.framework.artifacts import generate_unique_id
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


@register_experiment
class AbRefinementBlockExperiment(ExperimentBase):
    """Run pairs A/B to test a refinement-block prompt checklist."""

    name = "ab_refinement_block"
    summary = "A/B: middle refinement stage with vs without an explicit refinement block checklist."
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

        _ = runner_cfg  # unused; retained for interface consistency

        runs_per_variant = int(cli_args.get("runs", 0) or 0)
        if runs_per_variant <= 0:
            raise ValueError("--runs must be > 0")

        base_seed = int(cli_args.get("seed")) if cli_args.get("seed") is not None else int(datetime.now().timestamp())
        data_mode: DataMode = str(cli_args.get("data") or "config")  # type: ignore[assignment]
        if data_mode not in ("sample", "config"):
            raise ValueError(f"Unknown --data: {data_mode!r} (expected: sample|config)")

        stage_prefix = ["ab.random_token", "ab.scene_draft"]
        stage_suffix = ["ab.final_prompt_format"]

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

        planned: list[RunSpec] = []
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
