from __future__ import annotations

"""3x3 experiment: compare three prompt pipeline variants (A/B/C) across shared concepts."""

import random
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from image_project.foundation.config_io import deep_merge
from image_project.framework.artifacts import generate_unique_id
from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline.pipeline_overrides import PromptPipelineConfig
from image_project.impl.current.experiments import ExperimentBase, RunSpec, register_experiment
from image_project.prompts.preprompt import load_prompt_data, select_random_concepts

RunMode = Literal["full", "prompt_only"]


def _merge_cfg(base_cfg: Mapping[str, Any], *overlays: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge overlays onto a base mapping."""

    merged: Any = dict(base_cfg)
    for overlay in overlays:
        merged = deep_merge(merged, overlay, path="")
    if not isinstance(merged, dict):
        raise TypeError("Merged config must be a mapping")
    return merged


def _normalize_string_list(items: Sequence[Any] | None) -> list[str]:
    """Normalize a sequence of arbitrary values into a list[str] without empties."""

    if not items:
        return []
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _dedupe_preserve_order(items: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate items while preserving first-seen order."""

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _parse_sets(value: str) -> tuple[str, ...]:
    """Parse a comma-separated A/B/C selector list."""

    raw = (value or "").strip()
    if not raw:
        return ("A", "B", "C")
    tokens = [token.strip().upper() for token in raw.split(",") if token.strip()]
    allowed = {"A", "B", "C"}
    unknown = sorted(set(tokens) - allowed)
    if unknown:
        raise ValueError(f"Unknown set id(s): {unknown} (expected: A,B,C)")
    return tuple(tokens)


def _sample_shared_concepts_by_run(
    *,
    base_cfg: Mapping[str, Any],
    runs_per_set: int,
    base_seed: int,
    pinned_concepts: Sequence[str],
) -> tuple[list[int], list[tuple[str, ...]]]:
    """Sample run-indexed concepts once and reuse them across sets (A1/B1/C1 share, etc.)."""

    if runs_per_set <= 0:
        raise ValueError("runs_per_set must be > 0")

    cfg, _cfg_warnings = RunConfig.from_dict(dict(base_cfg))
    prompt_cfg, _prompt_warnings = PromptPipelineConfig.from_root_dict(
        base_cfg,
        run_mode=cfg.run_mode,
        generation_dir=cfg.generation_dir,
    )

    prompt_data = load_prompt_data(prompt_cfg.categories_path)
    pinned = _normalize_string_list(pinned_concepts)

    seeds: list[int] = []
    concepts_by_run: list[tuple[str, ...]] = []
    used: set[tuple[str, ...]] = set()

    for idx in range(runs_per_set):
        attempt = 0
        while True:
            concept_seed = int(base_seed) + idx + attempt * 1_000
            rng = random.Random(concept_seed)
            sampled = select_random_concepts(prompt_data, rng)
            sampled_clean = _normalize_string_list(sampled)

            combined = _dedupe_preserve_order([*pinned, *sampled_clean])
            if combined and combined not in used:
                seeds.append(concept_seed)
                concepts_by_run.append(combined)
                used.add(combined)
                break

            attempt += 1
            if attempt >= 50:
                seeds.append(concept_seed)
                concepts_by_run.append(combined or tuple(pinned))
                break

    return seeds, concepts_by_run


@register_experiment
class ThreeByThreeExperiment(ExperimentBase):
    """Compare three prompt pipelines A/B/C across shared per-run concepts."""

    name = "3x3"
    summary = "3x3: A/B blackbox profile routing (generator_hints vs raw), plus C simple_no_concepts."
    default_run_mode: RunMode = "full"

    def add_cli_args(self, parser: Any) -> None:
        """Register CLI args for this experiment."""

        parser.add_argument(
            "--sets",
            type=str,
            default="A,B,C",
            help="Comma-separated set ids to run (A,B,C).",
        )
        parser.add_argument("--runs-per-set", type=int, default=3, help="Runs per set.")
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Base seed used to derive per-run prompt.random_seed values.",
        )
        parser.add_argument(
            "--concept",
            action="append",
            default=None,
            help="Optional pinned concept (repeatable). If omitted, concepts are randomly sampled per run index and shared across sets.",
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
        """Build the 3x3 matrix plan with shared concepts."""

        _ = (runner_cfg, output_root, experiment_id)

        runs_per_set = int(cli_args.get("runs_per_set", 0) or 0)
        if runs_per_set <= 0:
            raise ValueError("--runs-per-set must be > 0")

        enabled_sets = _parse_sets(str(cli_args.get("sets") or "A,B,C"))
        base_seed = int(cli_args.get("seed")) if cli_args.get("seed") is not None else int(datetime.now().timestamp())
        pinned_concepts = _normalize_string_list(cli_args.get("concept") or [])

        concept_seeds_by_run, concepts_by_run = _sample_shared_concepts_by_run(
            base_cfg=base_cfg,
            runs_per_set=runs_per_set,
            base_seed=base_seed,
            pinned_concepts=pinned_concepts,
        )

        common_overrides: dict[str, Any] = {
            "prompt": {
                "stage_configs": {
                    "defaults": {
                        # 3x3 experiment disables concept filtering for parity with fixed concept injection.
                        "preprompt.filter_concepts": {"enabled": False},
                    }
                }
            }
        }

        variant_overrides: dict[str, dict[str, Any]] = {
            "A": {
                "prompt": {
                    "plan": "blackbox",
                    "stage_configs": {
                        "defaults": {
                            "blackbox.generate_idea_cards": {"num_ideas": 8},
                            "blackbox.idea_cards_judge_score": {"judge_profile_source": "generator_hints"},
                            "blackbox.select_idea_card": {"num_ideas": 8},
                            "blackbox.image_prompt_creation": {"final_profile_source": "generator_hints"},
                        }
                    },
                },
                "experiment": {
                    "variant": "A_blackbox_hints",
                    "notes": "blackbox plan; judge/final profile sources use generator_hints; shared per-run concepts; concept filters disabled",
                    "tags": ["exp3x3", "set:A", "plan:blackbox", "profile:hints", "mode:" + run_mode],
                },
            },
            "B": {
                "prompt": {
                    "plan": "blackbox",
                    "stage_configs": {
                        "defaults": {
                            "blackbox.generate_idea_cards": {"num_ideas": 8},
                            "blackbox.idea_cards_judge_score": {"judge_profile_source": "raw"},
                            "blackbox.select_idea_card": {"num_ideas": 8},
                            "blackbox.image_prompt_creation": {"final_profile_source": "raw"},
                        }
                    },
                },
                "experiment": {
                    "variant": "B_blackbox_raw",
                    "notes": "blackbox plan; judge/final profile sources use raw profile; shared per-run concepts; concept filters disabled",
                    "tags": ["exp3x3", "set:B", "plan:blackbox", "profile:raw", "mode:" + run_mode],
                },
            },
            "C": {
                "prompt": {"plan": "simple_no_concepts"},
                "experiment": {
                    "variant": "C_simple_no_concepts",
                    "notes": "simple_no_concepts plan (concept selection ignored); shared per-run concepts generated for parity",
                    "tags": ["exp3x3", "set:C", "plan:simple_no_concepts", "mode:" + run_mode],
                },
                "context": {"enabled": False},
            },
        }

        planned: list[RunSpec] = []
        for set_id in enabled_sets:
            set_overrides = variant_overrides[set_id]
            set_cfg_overlay = _merge_cfg(common_overrides, set_overrides)

            for run_index in range(runs_per_set):
                seed = int(base_seed) + (ord(set_id) - ord("A")) * 10_000 + run_index
                concept_seed = int(concept_seeds_by_run[run_index])
                concepts = _dedupe_preserve_order([str(c) for c in concepts_by_run[run_index]])
                generation_id = f"{set_id}{run_index + 1}_{generate_unique_id()}"

                run_overlay: dict[str, Any] = {
                    "prompt": {"random_seed": seed},
                    "experiment": {"tags": list(set_overrides["experiment"]["tags"]) + [f"run:{run_index + 1}"]},
                }

                # Only plans that include preprompt.select_concepts can accept an instance config override.
                if set_id in {"A", "B"}:
                    run_overlay["prompt"]["stage_configs"] = {
                        "instances": {
                            "preprompt.select_concepts": {
                                "strategy": "fixed",
                                "fixed": list(concepts),
                            }
                        }
                    }

                cfg_dict = _merge_cfg(base_cfg, set_cfg_overlay, run_overlay)
                planned.append(
                    RunSpec(
                        variant=set_id,
                        variant_name=str(set_overrides["experiment"]["variant"]),
                        run=run_index + 1,
                        generation_id=generation_id,
                        seed=seed,
                        cfg_dict=cfg_dict,
                        meta={
                            "concept_seed": concept_seed,
                            "concepts": list(concepts),
                        },
                    )
                )

        plan_meta = {
            "base_seed": base_seed,
            "runs_per_set": runs_per_set,
            "sets": list(enabled_sets),
            "pinned_concepts": list(pinned_concepts),
            "concepts_by_run": [
                {"run": idx + 1, "concept_seed": concept_seeds_by_run[idx], "concepts": list(concepts_by_run[idx])}
                for idx in range(len(concepts_by_run))
            ],
        }
        return planned, plan_meta

