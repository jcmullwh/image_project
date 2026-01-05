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
from image_project.app.experiment_dry_run import write_experiment_plan_full
from image_project.foundation.config_io import load_config
from image_project.framework.artifacts import generate_unique_id
from image_project.framework.artifacts_index import maybe_update_artifacts_index
from image_project.framework.config import RunConfig


RunMode = Literal["full", "prompt_only"]


@dataclass(frozen=True)
class PlannedRun:
    set_id: str
    set_name: str
    run_index: int
    generation_id: str
    seed: int
    concept_seed: int
    concepts: tuple[str, ...]
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
    return os.path.join("_artifacts", "experiments", f"{timestamp}_profile_v5_3x3")


def _parse_sets(value: str) -> tuple[str, ...]:
    raw = (value or "").strip()
    if not raw:
        return ("A", "B", "C")
    tokens = [token.strip().upper() for token in raw.split(",") if token.strip()]
    allowed = {"A", "B", "C"}
    unknown = sorted(set(tokens) - allowed)
    if unknown:
        raise ValueError(f"Unknown set id(s): {unknown} (expected: A,B,C)")
    return tuple(tokens)


def _normalize_string_list(items: Sequence[Any] | None) -> list[str]:
    if not items:
        return []
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _dedupe_preserve_order(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def generate_shared_concepts_by_run(
    *,
    base_cfg: Mapping[str, Any],
    runs_per_set: int,
    base_seed: int,
    pinned_concepts: Sequence[str],
) -> tuple[list[int], list[tuple[str, ...]]]:
    """
    Generate run-indexed concepts once and reuse them across sets:
      - run 1 concepts used for A1/B1/C1
      - run 2 concepts used for A2/B2/C2
      - ...
    """

    if runs_per_set <= 0:
        raise ValueError("runs_per_set must be > 0")

    # Resolve + normalize the categories path the same way the pipeline does.
    cfg_for_paths, _warnings = RunConfig.from_dict(_merge_cfg(base_cfg, {"run": {"mode": "prompt_only"}}))

    from image_project.impl.current import prompting as prompt_impl

    prompt_data = prompt_impl.load_prompt_data(cfg_for_paths.categories_path)

    pinned = _normalize_string_list(pinned_concepts)

    seeds: list[int] = []
    concepts_by_run: list[tuple[str, ...]] = []
    used: set[tuple[str, ...]] = set()

    for idx in range(runs_per_set):
        attempt = 0
        while True:
            concept_seed = int(base_seed) + idx + attempt * 1_000
            rng = random.Random(concept_seed)
            sampled = prompt_impl.select_random_concepts(prompt_data, rng)
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


def build_plan(
    *,
    base_cfg: Mapping[str, Any],
    output_root: str,
    experiment_id: str,
    run_mode: RunMode,
    runs_per_set: int,
    base_seed: int,
    enabled_sets: Sequence[str],
    concept_seeds_by_run: Sequence[int],
    concepts_by_run: Sequence[Sequence[str]],
    enable_upscale: bool,
    enable_upload: bool,
    profile_like_dislike_path: str,
    profile_love_like_dislike_hate_path: str,
    generator_profile_hints_path: str | None,
) -> list[PlannedRun]:
    if len(concept_seeds_by_run) != runs_per_set or len(concepts_by_run) != runs_per_set:
        raise ValueError("concept_seeds_by_run and concepts_by_run must match runs_per_set length")

    output_root = os.path.abspath(output_root)
    log_dir = os.path.join(output_root, "logs")
    generation_dir = os.path.join(output_root, "generated")
    upscale_dir = os.path.join(output_root, "upscaled")

    common_overrides: dict[str, Any] = {
        "run": {"mode": run_mode},
        "image": {
            "log_path": log_dir,
            "generation_path": generation_dir,
            "upscale_path": upscale_dir,
        },
        "prompt": {
            "generations_path": os.path.join(log_dir, "generations_v2.csv"),
            "titles_manifest_path": os.path.join(generation_dir, "titles_manifest.csv"),
        },
        "experiment": {"id": experiment_id},
        "upscale": {"enabled": bool(enable_upscale)},
        "rclone": {"enabled": bool(enable_upload)},
    }

    variant_overrides: dict[str, dict[str, Any]] = {
        # A: blackbox refine (idea generation + explicit refinement), using v5 like/dislike profile.
        "A": {
            "prompt": {
                "plan": "blackbox_refine",
                "profile_path": profile_like_dislike_path,
                "refinement": {"policy": "none"},
                "scoring": {
                    "enabled": True,
                    "num_ideas": 8,
                    "judge_profile_source": "raw",
                    "idea_profile_source": "generator_hints_plus_dislikes",
                    "final_profile_source": "raw",
                    "generator_profile_abstraction": True,
                    "novelty": {"enabled": False, "window": 0},
                },
                "blackbox_refine": {
                    "variation_prompt": {
                        "include_profile": True,
                        "profile_source": "dislikes_only",
                    }
                },
            },
            "experiment": {
                "variant": "A_blackbox_refine_like_dislike",
                "notes": "blackbox_refine plan + scoring; v5 like/dislike profile; shared per-run concepts; concept filters enabled",
                "tags": [
                    "exp3x3",
                    "profile_v5",
                    "set:A",
                    "plan:blackbox_refine",
                    "refinement:none",
                    "profile:like_dislike",
                ],
            },
        },
        # B: blackbox refine (idea generation + explicit refinement), using v5 love/like/dislike/hate profile.
        "B": {
            "prompt": {
                "plan": "blackbox_refine",
                "profile_path": profile_love_like_dislike_hate_path,
                "refinement": {"policy": "none"},
                "scoring": {
                    "enabled": True,
                    "num_ideas": 8,
                    "judge_profile_source": "raw",
                    "idea_profile_source": "generator_hints_plus_dislikes",
                    "final_profile_source": "raw",
                    "generator_profile_abstraction": True,
                    "novelty": {"enabled": False, "window": 0},
                },
                "blackbox_refine": {
                    "variation_prompt": {
                        "include_profile": True,
                        "profile_source": "dislikes_only",
                    }
                },
            },
            "experiment": {
                "variant": "B_blackbox_refine_love_like_dislike_hate",
                "notes": "blackbox_refine plan + scoring; v5 love/like/dislike/hate profile; shared per-run concepts; concept filters enabled",
                "tags": [
                    "exp3x3",
                    "profile_v5",
                    "set:B",
                    "plan:blackbox_refine",
                    "refinement:none",
                    "profile:love_like_dislike_hate",
                ],
            },
        },
        # C: one-shot final prompt creation from concepts + profile, no scoring/refinement.
        "C": {
            "prompt": {
                "plan": "direct",
                "profile_path": profile_love_like_dislike_hate_path,
                "refinement": {"policy": "none"},
                "scoring": {"enabled": False},
            },
            "experiment": {
                "variant": "C_direct_none_love_like_dislike_hate",
                "notes": "direct plan (single-turn final prompt from concepts + profile); v5 love/like/dislike/hate profile; shared per-run concepts; concept filters enabled",
                "tags": [
                    "exp3x3",
                    "profile_v5",
                    "set:C",
                    "plan:direct",
                    "refinement:none",
                    "profile:love_like_dislike_hate",
                ],
            },
        },
    }

    if generator_profile_hints_path:
        for set_id in ("A", "B"):
            scoring_cfg = variant_overrides[set_id]["prompt"]["scoring"]
            if not isinstance(scoring_cfg, dict):
                raise TypeError(f"Expected prompt.scoring mapping for set {set_id}")
            scoring_cfg["generator_profile_hints_path"] = generator_profile_hints_path

    planned: list[PlannedRun] = []
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
                    "concepts": {
                        "selection": {"strategy": "fixed", "fixed": list(concepts)},
                        "filters": {"enabled": True},
                    },
                },
                "experiment": {
                    "tags": list(set_overrides["experiment"]["tags"]) + [f"run:{run_index + 1}"]
                },
            }

            cfg_dict = _merge_cfg(base_cfg, set_cfg_overlay, run_overlay)
            planned.append(
                PlannedRun(
                    set_id=set_id,
                    set_name=str(set_overrides["prompt"]["plan"]),
                    run_index=run_index + 1,
                    generation_id=generation_id,
                    seed=seed,
                    concept_seed=concept_seed,
                    concepts=concepts,
                    cfg_dict=cfg_dict,
                )
            )

    return planned


def _print_plan(plan: Sequence[PlannedRun]) -> None:
    for entry in plan:
        prompt_cfg = entry.cfg_dict.get("prompt") if isinstance(entry.cfg_dict.get("prompt"), dict) else {}
        plan_name = prompt_cfg.get("plan")
        refinement = (
            (prompt_cfg.get("refinement") or {}).get("policy")
            if isinstance(prompt_cfg.get("refinement"), dict)
            else None
        )
        scoring_enabled = (
            (prompt_cfg.get("scoring") or {}).get("enabled")
            if isinstance(prompt_cfg.get("scoring"), dict)
            else None
        )
        profile_path = prompt_cfg.get("profile_path")
        concepts_preview = ", ".join(entry.concepts) if entry.concepts else "<none>"
        print(
            f"{entry.set_id}{entry.run_index}: generation_id={entry.generation_id} seed={entry.seed} "
            f"plan={plan_name} refinement={refinement} scoring={scoring_enabled} "
            f"profile={profile_path} concepts=[{concepts_preview}]"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_experiment_profile_v5_3x3",
        description=(
            "Run a 3x3 experiment (A/B/C variants x 3 runs) to compare v5 profile formats and a direct one-shot plan."
        ),
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
        help="Experiment id written into transcript metadata. Defaults to a timestamped exp id.",
    )
    parser.add_argument(
        "--sets",
        type=str,
        default="A,B,C",
        help="Comma-separated set ids to run (A,B,C).",
    )
    parser.add_argument("--runs-per-set", type=int, default=3, help="Runs (images) per set.")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=("full", "prompt_only"),
        help="Run mode: full generates images (required unless --dry-run); prompt_only skips image/upscale/upload.",
    )
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
        default="./image_project/impl/current/data/sample/user_profile_v5_like_dislike.csv",
        help="CSV path for the v5 like/dislike profile format (A set).",
    )
    parser.add_argument(
        "--profile-love-like-dislike-hate-path",
        type=str,
        default="./image_project/impl/current/data/sample/user_profile_v5_love_like_dislike_hate.csv",
        help="CSV path for the v5 love/like/dislike/hate profile format (B/C sets).",
    )
    parser.add_argument(
        "--generator-profile-hints-path",
        type=str,
        default=None,
        help=(
            "Optional path to a generator-safe profile hints file (e.g. v5 abstraction CSV). "
            "When set, A/B load hints from this file instead of generating them."
        ),
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
        help="Only print the planned runs + validate configs; do not execute the pipeline.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining planned runs even if one fails.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    enabled_sets = _parse_sets(args.sets)

    if not args.dry_run and args.mode != "full":
        raise SystemExit(
            "This experiment runner is intended to generate images; use --mode full (or --dry-run)."
        )

    output_root = args.output_root or _default_output_root()
    experiment_id = args.experiment_id or f"profile_v5_3x3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    base_seed = int(args.seed) if args.seed is not None else int(datetime.now().timestamp())

    pinned_concepts = _normalize_string_list(args.concept or [])

    base_cfg, cfg_meta = load_config(config_path=args.config_path)

    concept_seeds_by_run, concepts_by_run = generate_shared_concepts_by_run(
        base_cfg=base_cfg,
        runs_per_set=args.runs_per_set,
        base_seed=base_seed,
        pinned_concepts=pinned_concepts,
    )

    plan = build_plan(
        base_cfg=base_cfg,
        output_root=output_root,
        experiment_id=experiment_id,
        run_mode=args.mode,  # type: ignore[arg-type]
        runs_per_set=args.runs_per_set,
        base_seed=base_seed,
        enabled_sets=enabled_sets,
        concept_seeds_by_run=concept_seeds_by_run,
        concepts_by_run=concepts_by_run,
        enable_upscale=bool(args.enable_upscale),
        enable_upload=bool(args.enable_upload),
        profile_like_dislike_path=str(args.profile_like_dislike_path),
        profile_love_like_dislike_hate_path=str(args.profile_love_like_dislike_hate_path),
        generator_profile_hints_path=(str(args.generator_profile_hints_path) if args.generator_profile_hints_path else None),
    )

    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, "experiment_plan.json")

    # Validate configs up front (no AI calls).
    validation_errors: list[dict[str, Any]] = []
    for entry in plan:
        try:
            RunConfig.from_dict(entry.cfg_dict)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(
                {
                    "set": entry.set_id,
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
        "runs_per_set": args.runs_per_set,
        "sets": list(enabled_sets),
        "pinned_concepts": list(pinned_concepts),
        "concepts_by_run": [
            {"run": idx + 1, "concept_seed": concept_seeds_by_run[idx], "concepts": list(concepts_by_run[idx])}
            for idx in range(len(concepts_by_run))
        ],
        "profiles": {
            "like_dislike": str(args.profile_like_dislike_path),
            "love_like_dislike_hate": str(args.profile_love_like_dislike_hate_path),
            "generator_profile_hints_path": str(args.generator_profile_hints_path)
            if args.generator_profile_hints_path
            else None,
        },
        "enable_upscale": bool(args.enable_upscale),
        "enable_upload": bool(args.enable_upload),
        "validation_errors": validation_errors,
        "planned_runs": [
            {
                "set": entry.set_id,
                "plan": entry.set_name,
                "run": entry.run_index,
                "generation_id": entry.generation_id,
                "seed": entry.seed,
                "concept_seed": entry.concept_seed,
                "concepts": list(entry.concepts),
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
                "set": entry.set_id,
                "set_name": entry.set_name,
                "run": entry.run_index,
                "generation_id": entry.generation_id,
                "seed": entry.seed,
                "concept_seed": entry.concept_seed,
                "concepts": list(entry.concepts),
                "cfg_dict": entry.cfg_dict,
            }
            for entry in plan
        ],
    )

    if validation_errors:
        _print_plan(plan)
        print(f"Wrote experiment plan to {summary_path}")
        print(f"Wrote expanded experiment plan to {full_plan_path}")
        print("Config validation errors:")
        print(json.dumps(validation_errors, ensure_ascii=False, indent=2))
        maybe_update_artifacts_index(repo_root=str(PROJECT_ROOT))
        return 2

    _print_plan(plan)
    print(f"Wrote experiment plan to {summary_path}")
    print(f"Wrote expanded experiment plan to {full_plan_path}")

    if args.dry_run:
        maybe_update_artifacts_index(repo_root=str(PROJECT_ROOT))
        return 0

    results: list[dict[str, Any]] = []
    failures = 0
    for entry in plan:
        try:
            ctx = run_generation(entry.cfg_dict, generation_id=entry.generation_id, config_meta=cfg_meta)
            results.append(
                {
                    "set": entry.set_id,
                    "run": entry.run_index,
                    "generation_id": entry.generation_id,
                    "seed": entry.seed,
                    "status": "success",
                    "image_path": getattr(ctx, "image_path", None),
                    "outputs": {
                        "title_generation": ctx.outputs.get("title_generation"),
                        "prompt_pipeline": ctx.outputs.get("prompt_pipeline"),
                    },
                    "error": ctx.error,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            results.append(
                {
                    "set": entry.set_id,
                    "run": entry.run_index,
                    "generation_id": entry.generation_id,
                    "seed": entry.seed,
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
    maybe_update_artifacts_index(repo_root=str(PROJECT_ROOT))
    return 0 if failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
