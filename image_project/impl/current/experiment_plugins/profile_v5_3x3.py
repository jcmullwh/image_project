from __future__ import annotations

"""3x3 experiment: compare v5 profile formats and a direct prompt path."""

import os
import random
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from image_project.foundation.config_io import deep_merge, find_repo_root
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


def _require_file_path(path: str, *, label: str) -> str:
    """Validate that a file path exists and is non-empty; return absolute path."""

    text = str(path or "").strip()
    if not text:
        raise ValueError(f"{label} is required (got empty)")
    expanded = os.path.expandvars(os.path.expanduser(text))
    if not os.path.isabs(expanded):
        expanded = os.path.join(find_repo_root(), expanded)
    expanded = os.path.abspath(expanded)
    if not os.path.exists(expanded):
        raise ValueError(f"{label} not found: {expanded}")
    if os.path.isdir(expanded):
        raise ValueError(f"{label} is a directory (expected file): {expanded}")
    if os.path.getsize(expanded) == 0:
        raise ValueError(f"{label} is empty: {expanded}")
    return expanded


def _resolve_required_path(
    *,
    cli_value: str | None,
    runner_cfg: Mapping[str, Any],
    runner_key: str,
    label: str,
) -> str:
    """Resolve a required file path from CLI or runner_cfg, failing loudly."""

    if cli_value:
        return _require_file_path(cli_value, label=label)
    cfg_value = runner_cfg.get(runner_key)
    if isinstance(cfg_value, str) and cfg_value.strip():
        return _require_file_path(cfg_value, label=label)
    raise ValueError(
        f"Missing {label}. Provide CLI flag or set experiment_runners.profile_v5_3x3.{runner_key} in config."
    )


def _resolve_optional_path(
    *,
    cli_value: str | None,
    runner_cfg: Mapping[str, Any],
    runner_key: str,
    label: str,
) -> str | None:
    """Resolve an optional file path from CLI or runner_cfg."""

    if cli_value:
        return _require_file_path(cli_value, label=label)
    cfg_value = runner_cfg.get(runner_key)
    if isinstance(cfg_value, str) and cfg_value.strip():
        return _require_file_path(cfg_value, label=label)
    return None


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

    max_random_concepts = 2
    for idx in range(runs_per_set):
        attempt = 0
        while True:
            concept_seed = int(base_seed) + idx + attempt * 1_000
            rng = random.Random(concept_seed)
            sampled = select_random_concepts(prompt_data, rng)
            sampled_clean = _normalize_string_list(sampled)[:max_random_concepts]

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
class ProfileV5ThreeByThreeExperiment(ExperimentBase):
    """Compare v5 profile formats under a 3x3 plan."""

    name = "profile_v5_3x3"
    summary = "3x3: A/B blackbox_refine with v5 profile variants, plus C direct prompt from concepts."
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
        parser.add_argument(
            "--profile-like-dislike-path",
            type=str,
            default=None,
            help=(
                "CSV path for the v5 like/dislike profile format (A set). "
                "If omitted, uses experiment_runners.profile_v5_3x3.profile_like_dislike_path from the loaded config."
            ),
        )
        parser.add_argument(
            "--profile-love-like-dislike-hate-path",
            type=str,
            default=None,
            help=(
                "CSV path for the v5 love/like/dislike/hate profile format (B/C sets). "
                "If omitted, uses experiment_runners.profile_v5_3x3.profile_love_like_dislike_hate_path from the loaded config."
            ),
        )
        parser.add_argument(
            "--generator-profile-hints-path",
            type=str,
            default=None,
            help=(
                "Optional CSV path for generator profile hints (abstraction). "
                "If omitted, uses experiment_runners.profile_v5_3x3.generator_profile_hints_path when set."
            ),
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

        _ = (output_root, experiment_id)

        runs_per_set = int(cli_args.get("runs_per_set", 0) or 0)
        if runs_per_set <= 0:
            raise ValueError("--runs-per-set must be > 0")

        enabled_sets = _parse_sets(str(cli_args.get("sets") or "A,B,C"))
        base_seed = int(cli_args.get("seed")) if cli_args.get("seed") is not None else int(datetime.now().timestamp())
        pinned_concepts = _normalize_string_list(cli_args.get("concept") or [])

        profile_like_dislike_path = _resolve_required_path(
            cli_value=cli_args.get("profile_like_dislike_path"),
            runner_cfg=runner_cfg,
            runner_key="profile_like_dislike_path",
            label="profile_like_dislike_path",
        )
        profile_love_like_dislike_hate_path = _resolve_required_path(
            cli_value=cli_args.get("profile_love_like_dislike_hate_path"),
            runner_cfg=runner_cfg,
            runner_key="profile_love_like_dislike_hate_path",
            label="profile_love_like_dislike_hate_path",
        )
        generator_profile_hints_path = _resolve_optional_path(
            cli_value=cli_args.get("generator_profile_hints_path"),
            runner_cfg=runner_cfg,
            runner_key="generator_profile_hints_path",
            label="generator_profile_hints_path",
        )

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
                        # Profile v5 experiment runs concept filters (dislike rewrite) by default.
                        "preprompt.filter_concepts": {"enabled": True},
                    }
                }
            }
        }

        variant_overrides: dict[str, dict[str, Any]] = {
            "A": {
                "prompt": {
                    "plan": "blackbox_refine",
                    "profile_path": profile_like_dislike_path,
                    "stage_configs": {
                        "defaults": {
                            "blackbox.prepare": {"novelty": {"enabled": False, "window": 0}},
                            "blackbox.generator_profile_hints": {"mode": "abstract"},
                            "blackbox.generate_idea_cards": {
                                "num_ideas": 8,
                                "idea_profile_source": "generator_hints_plus_dislikes",
                            },
                            "blackbox.idea_cards_judge_score": {"judge_profile_source": "raw"},
                            "blackbox.select_idea_card": {
                                "num_ideas": 8,
                                "novelty": {"enabled": False, "window": 0},
                            },
                            "blackbox_refine.seed_prompt": {"final_profile_source": "raw"},
                            "blackbox_refine.loop": {
                                "variation_prompt": {
                                    "include_profile": True,
                                    "profile_source": "dislikes_only",
                                },
                                "judge_profile_source": "raw",
                                "novelty": {"enabled": False, "window": 0},
                            },
                        }
                    },
                },
                "experiment": {
                    "variant": "A_blackbox_refine_like_dislike",
                    "notes": "blackbox_refine plan; v5 like/dislike profile; shared per-run concepts; concept filters enabled",
                    "tags": [
                        "exp3x3",
                        "profile_v5",
                        "set:A",
                        "plan:blackbox_refine",
                        "profile:like_dislike",
                        "mode:" + run_mode,
                    ],
                },
            },
            "B": {
                "prompt": {
                    "plan": "blackbox_refine",
                    "profile_path": profile_love_like_dislike_hate_path,
                    "stage_configs": {
                        "defaults": {
                            "blackbox.prepare": {"novelty": {"enabled": False, "window": 0}},
                            "blackbox.generator_profile_hints": {"mode": "abstract"},
                            "blackbox.generate_idea_cards": {
                                "num_ideas": 8,
                                "idea_profile_source": "generator_hints_plus_dislikes",
                            },
                            "blackbox.idea_cards_judge_score": {"judge_profile_source": "raw"},
                            "blackbox.select_idea_card": {
                                "num_ideas": 8,
                                "novelty": {"enabled": False, "window": 0},
                            },
                            "blackbox_refine.seed_prompt": {"final_profile_source": "raw"},
                            "blackbox_refine.loop": {
                                "variation_prompt": {
                                    "include_profile": True,
                                    "profile_source": "dislikes_only",
                                },
                                "judge_profile_source": "raw",
                                "novelty": {"enabled": False, "window": 0},
                            },
                        }
                    },
                },
                "experiment": {
                    "variant": "B_blackbox_refine_love_like_dislike_hate",
                    "notes": "blackbox_refine plan; v5 love/like/dislike/hate profile; shared per-run concepts; concept filters enabled",
                    "tags": [
                        "exp3x3",
                        "profile_v5",
                        "set:B",
                        "plan:blackbox_refine",
                        "profile:love_like_dislike_hate",
                        "mode:" + run_mode,
                    ],
                },
            },
            "C": {
                "prompt": {
                    "plan": "direct",
                    "profile_path": profile_love_like_dislike_hate_path,
                },
                "experiment": {
                    "variant": "C_direct_none_love_like_dislike_hate",
                    "notes": "direct plan (single-turn final prompt from concepts + profile); v5 profile; shared per-run concepts",
                    "tags": [
                        "exp3x3",
                        "profile_v5",
                        "set:C",
                        "plan:direct",
                        "profile:love_like_dislike_hate",
                        "mode:" + run_mode,
                    ],
                },
            },
        }

        if generator_profile_hints_path:
            for set_id in ("A", "B"):
                prompt_cfg = variant_overrides[set_id].get("prompt")
                if not isinstance(prompt_cfg, dict):
                    raise TypeError(f"Expected prompt mapping for set {set_id}")
                stage_cfgs = prompt_cfg.setdefault("stage_configs", {})
                if not isinstance(stage_cfgs, dict):
                    raise TypeError(f"Expected prompt.stage_configs mapping for set {set_id}")
                defaults = stage_cfgs.setdefault("defaults", {})
                if not isinstance(defaults, dict):
                    raise TypeError(f"Expected prompt.stage_configs.defaults mapping for set {set_id}")
                hints_cfg = defaults.setdefault("blackbox.generator_profile_hints", {})
                if not isinstance(hints_cfg, dict):
                    raise TypeError(
                        f"Expected prompt.stage_configs.defaults.blackbox.generator_profile_hints mapping for set {set_id}"
                    )
                hints_cfg["mode"] = "file"
                hints_cfg["hints_path"] = generator_profile_hints_path

        planned: list[RunSpec] = []
        for set_id in enabled_sets:
            set_overrides = variant_overrides[set_id]
            set_cfg_overlay = _merge_cfg(common_overrides, set_overrides)

            for run_index in range(runs_per_set):
                seed = int(base_seed) + (ord(set_id) - ord("A")) * 10_000 + run_index
                concept_seed = int(concept_seeds_by_run[run_index])
                concepts = _dedupe_preserve_order([str(c) for c in concepts_by_run[run_index]])
                generation_id = f"{set_id}{run_index + 1}_{generate_unique_id()}"

                run_overlay = {
                    "prompt": {
                        "random_seed": seed,
                        "stage_configs": {
                            "instances": {
                                "preprompt.select_concepts": {
                                    "strategy": "fixed",
                                    "fixed": list(concepts),
                                }
                            }
                        },
                    },
                    "experiment": {"tags": list(set_overrides["experiment"]["tags"]) + [f"run:{run_index + 1}"]},
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
                            "profiles": {
                                "like_dislike": profile_like_dislike_path,
                                "love_like_dislike_hate": profile_love_like_dislike_hate_path,
                                "generator_hints": generator_profile_hints_path,
                            },
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
            "profiles": {
                "like_dislike": profile_like_dislike_path,
                "love_like_dislike_hate": profile_love_like_dislike_hate_path,
                "generator_profile_hints_path": generator_profile_hints_path,
            },
        }
        return planned, plan_meta
