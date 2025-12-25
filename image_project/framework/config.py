from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from image_project.foundation.config_io import find_repo_root
from image_project.framework.context import ContextManager


RunMode = Literal["full", "prompt_only"]
ScoringProfileSource = Literal["raw", "generator_hints"]
ScoringJudgeProfileSource = Literal["raw", "generator_hints", "generator_hints_plus_dislikes"]
ScoringIdeaProfileSource = Literal["raw", "generator_hints", "none"]
PromptNoveltyMethod = Literal["df_overlap_v1", "legacy_v0"]
NoveltyPenaltyScaling = Literal["linear", "sqrt", "quadratic"]
BlackboxRefineAlgorithm = Literal["hillclimb", "beam"]
BlackboxRefineProfileSource = Literal[
    "raw",
    "generator_hints",
    "likes_dislikes",
    "dislikes_only",
    "combined",
]
BlackboxRefineVariationTemplate = Literal["v1", "v2"]
BlackboxRefineMutationMode = Literal["none", "random", "cycle", "fixed"]
BlackboxRefineJudgeRubric = Literal["default", "strict", "novelty_heavy"]
BlackboxRefineAggregation = Literal["mean", "median", "min", "max", "trimmed_mean"]
BlackboxRefineTieBreaker = Literal["stable_id", "prefer_shorter", "prefer_novel"]


@dataclass(frozen=True)
class ExperimentConfig:
    id: str | None = None
    variant: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunReviewConfig:
    enabled: bool = False
    review_path: str | None = None


def parse_bool(value: Any, path: str) -> bool:
    """
    Strict boolean parsing to avoid bool('false') footguns.

    Accepts:
      - True/False
      - 0/1 (ints)
      - strings: true/false/1/0/yes/no (case-insensitive, surrounding whitespace ignored)

    Raises:
      ValueError for anything else, with the provided config key path.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Invalid boolean for {path}: {value!r}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Invalid boolean for {path}: {value!r}")

    raise ValueError(f"Invalid boolean for {path}: {value!r}")


def parse_float(value: Any, path: str) -> float:
    if value is None:
        raise ValueError(f"Invalid config value for {path}: None")
    if isinstance(value, bool):
        raise ValueError(f"Invalid config type for {path}: expected float, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"Invalid config value for {path}: must be a float")
        try:
            return float(value.strip())
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid config value for {path}: must be a float") from exc
    raise ValueError(f"Invalid config type for {path}: expected float")


def parse_int(value: Any, path: str) -> int:
    if value is None:
        raise ValueError(f"Invalid config value for {path}: None")
    if isinstance(value, bool):
        raise ValueError(f"Invalid config type for {path}: expected int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"Invalid config value for {path}: must be an int")
        try:
            return int(value.strip())
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid config value for {path}: must be an int") from exc
    raise ValueError(f"Invalid config type for {path}: expected int")


def parse_aspect_ratio(value: Any, path: str) -> float:
    """
    Parse an aspect ratio (width/height) from a few common formats.

    Accepts:
      - numeric (int/float) values > 0
      - strings like "16:9", "9/16", or "1.7778"
      - 2-element lists/tuples
    """

    ratio: float
    if isinstance(value, (int, float)):
        ratio = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"Invalid aspect ratio for {path}: {value!r}")
        if ":" in raw:
            parts = raw.split(":")
        elif "/" in raw:
            parts = raw.split("/")
        else:
            parts = [raw]

        if len(parts) == 1:
            try:
                ratio = float(parts[0])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid aspect ratio for {path}: {value!r}") from exc
        elif len(parts) == 2:
            try:
                num = float(parts[0])
                denom = float(parts[1])
                ratio = num / denom
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid aspect ratio for {path}: {value!r}") from exc
        else:
            raise ValueError(f"Invalid aspect ratio for {path}: {value!r}")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            ratio = float(value[0]) / float(value[1])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid aspect ratio for {path}: {value!r}") from exc
    else:
        raise ValueError(f"Invalid aspect ratio for {path}: {value!r}")

    if ratio <= 0:
        raise ValueError(f"Invalid aspect ratio for {path}: must be > 0 (got {value!r})")
    return ratio


@dataclass(frozen=True)
class PromptNoveltyConfig:
    enabled: bool
    window: int
    method: PromptNoveltyMethod = "df_overlap_v1"
    df_min: int = 3
    max_motifs: int = 200
    min_token_len: int = 3
    stopwords_extra: tuple[str, ...] = ()
    max_penalty: int = 20
    df_cap: int = 10
    alpha_only: bool = True
    scaling: NoveltyPenaltyScaling = "linear"


@dataclass(frozen=True)
class PromptScoringConfig:
    enabled: bool
    num_ideas: int
    exploration_rate: float
    judge_temperature: float
    judge_model: str | None
    judge_profile_source: ScoringJudgeProfileSource
    idea_profile_source: ScoringIdeaProfileSource
    final_profile_source: ScoringProfileSource
    generator_profile_abstraction: bool
    novelty: PromptNoveltyConfig


@dataclass(frozen=True)
class PromptBlackboxRefineVariationPromptConfig:
    template: BlackboxRefineVariationTemplate
    include_concepts: bool
    include_context_guidance: bool
    include_profile: bool
    profile_source: BlackboxRefineProfileSource
    include_novelty_summary: bool
    include_mutation_directive: bool
    include_scoring_rubric: bool


@dataclass(frozen=True)
class PromptBlackboxRefineMutationPerAxisConfig:
    enabled: bool
    axes: tuple[str, ...]


@dataclass(frozen=True)
class PromptBlackboxRefineMutationDirectivesConfig:
    mode: BlackboxRefineMutationMode
    directives: tuple[str, ...]
    per_axis: PromptBlackboxRefineMutationPerAxisConfig


@dataclass(frozen=True)
class PromptBlackboxRefineJudgeConfig:
    id: str
    model: str | None
    temperature: float | None
    weight: float
    rubric: BlackboxRefineJudgeRubric


@dataclass(frozen=True)
class PromptBlackboxRefineJudgingConfig:
    judges: tuple[PromptBlackboxRefineJudgeConfig, ...]
    aggregation: BlackboxRefineAggregation
    trimmed_mean_drop: int


@dataclass(frozen=True)
class PromptBlackboxRefineSelectionConfig:
    exploration_rate_override: float | None
    group_by_beam: bool
    tie_breaker: BlackboxRefineTieBreaker


@dataclass(frozen=True)
class PromptBlackboxRefineConfig:
    enabled: bool
    iterations: int
    algorithm: BlackboxRefineAlgorithm
    beam_width: int
    branching_factor: int
    include_parents_as_candidates: bool
    generator_model: str | None
    generator_temperature: float
    max_prompt_chars: int | None
    variation_prompt: PromptBlackboxRefineVariationPromptConfig
    mutation_directives: PromptBlackboxRefineMutationDirectivesConfig
    judging: PromptBlackboxRefineJudgingConfig
    selection: PromptBlackboxRefineSelectionConfig


ConceptSelectionStrategy = Literal["random", "fixed", "file"]
ContextInjectionLocation = Literal["system", "prompt", "both"]


@dataclass(frozen=True)
class PromptConceptSelectionConfig:
    strategy: ConceptSelectionStrategy
    fixed: tuple[str, ...]
    file_path: str | None


@dataclass(frozen=True)
class DislikeRewriteFilterConfig:
    enabled: bool
    temperature: float


@dataclass(frozen=True)
class PromptConceptFiltersConfig:
    enabled: bool
    order: tuple[str, ...]
    dislike_rewrite: DislikeRewriteFilterConfig


@dataclass(frozen=True)
class PromptConceptsConfig:
    selection: PromptConceptSelectionConfig
    filters: PromptConceptFiltersConfig


@dataclass(frozen=True)
class PromptStageOverride:
    temperature: float | None = None
    params: dict[str, Any] | None = None
    refinement_policy: str | None = None


@dataclass(frozen=True)
class RunConfig:
    generation_dir: str | None
    upscale_dir: str | None
    log_dir: str

    run_mode: RunMode
    experiment: ExperimentConfig
    run_review: RunReviewConfig

    categories_path: str
    profile_path: str
    generations_csv_path: str | None

    titles_manifest_path: str | None
    caption_font_path: str | None

    random_seed: int | None
    prompt_scoring: PromptScoringConfig
    prompt_blackbox_refine: PromptBlackboxRefineConfig | None
    prompt_concepts: PromptConceptsConfig

    prompt_plan: str
    prompt_refinement_policy: str
    prompt_stages_include: tuple[str, ...]
    prompt_stages_exclude: tuple[str, ...]
    prompt_stages_sequence: tuple[str, ...]
    prompt_stages_overrides: Mapping[str, PromptStageOverride]
    prompt_output_capture_stage: str | None
    prompt_refine_only_draft: str | None
    prompt_refine_only_draft_path: str | None

    context_enabled: bool
    context_injection_location: ContextInjectionLocation
    context_injectors: tuple[str, ...]
    context_cfg: Mapping[str, Any]

    rclone_enabled: bool
    rclone_remote: str | None
    rclone_album: str | None

    upscale_enabled: bool
    upscale_target_long_edge_px: int
    upscale_target_width_px: int | None
    upscale_target_height_px: int | None
    upscale_target_aspect_ratio: float | None
    upscale_engine: str
    upscale_realesrgan_binary: str | None
    upscale_model_name: str
    upscale_model_path: str | None
    upscale_tile_size: int
    upscale_tta: bool
    upscale_allow_fallback_resize: bool

    @staticmethod
    def from_dict(cfg: Mapping[str, Any]) -> tuple["RunConfig", list[str]]:
        """
        Parse and validate configuration, returning (RunConfig, warnings).

        Raises:
            ValueError: if required keys are missing or invalid.
        """

        if not isinstance(cfg, Mapping):
            raise ValueError("Config must be a mapping")

        warnings: list[str] = []
        repo_root: str | None = None

        strict_unknown_keys = False
        if "strict" in cfg:
            strict_unknown_keys = parse_bool(cfg.get("strict"), "strict")

        ANY: object = object()

        def collect_unknown_keys(
            mapping: Any,
            schema: Mapping[str, Any],
            *,
            prefix: str,
            exempt_keys: set[str] | None = None,
        ) -> list[str]:
            if not isinstance(mapping, Mapping):
                return []
            unknown: list[str] = []
            for key, value in mapping.items():
                if not isinstance(key, str):
                    continue
                if exempt_keys and key in exempt_keys:
                    continue
                if key not in schema:
                    unknown.append(f"{prefix}.{key}")
                    continue
                subschema = schema.get(key)
                if subschema is ANY:
                    continue
                if isinstance(subschema, Mapping):
                    unknown.extend(
                        collect_unknown_keys(
                            value,
                            subschema,
                            prefix=f"{prefix}.{key}",
                            exempt_keys=None,
                        )
                    )
            return unknown

        prompt_schema: Mapping[str, Any] = {
            "categories_path": None,
            "profile_path": None,
            "generations_path": None,
            "titles_manifest_path": None,
            "random_seed": None,
            "plan": None,
            "refinement": {"policy": None},
            "concepts": {
                "selection": {
                    "strategy": None,
                    "fixed": None,
                    "file_path": None,
                },
                "filters": {
                    "enabled": None,
                    "order": None,
                    "dislike_rewrite": {
                        "enabled": None,
                        "temperature": None,
                    },
                },
            },
            "stages": {
                "include": None,
                "exclude": None,
                "sequence": None,
                "overrides": ANY,  # validated explicitly elsewhere (dynamic keys).
            },
            "output": {"capture_stage": None},
            "refine_only": {"draft": None, "draft_path": None},
            "scoring": {
                "enabled": None,
                "num_ideas": None,
                "exploration_rate": None,
                "judge_temperature": None,
                "judge_model": None,
                "judge_profile_source": None,
                "idea_profile_source": None,
                "final_profile_source": None,
                "generator_profile_abstraction": None,
                "novelty": {
                    "enabled": None,
                    "window": None,
                    "method": None,
                    "df_min": None,
                    "max_motifs": None,
                    "min_token_len": None,
                    "stopwords_extra": None,
                    "max_penalty": None,
                    "df_cap": None,
                    "alpha_only": None,
                    "scaling": None,
                },
            },
            "blackbox_refine": {
                "enabled": None,
                "iterations": None,
                "algorithm": None,
                "beam_width": None,
                "branching_factor": None,
                "include_parents_as_candidates": None,
                "generator_model": None,
                "generator_temperature": None,
                "max_prompt_chars": None,
                "variation_prompt": {
                    "template": None,
                    "include_concepts": None,
                    "include_context_guidance": None,
                    "include_profile": None,
                    "profile_source": None,
                    "include_novelty_summary": None,
                    "include_mutation_directive": None,
                    "include_scoring_rubric": None,
                },
                "mutation_directives": {
                    "mode": None,
                    "directives": None,
                    "per_axis": {"enabled": None, "axes": None},
                },
                "judging": {
                    "judges": None,
                    "aggregation": None,
                    "trimmed_mean_drop": None,
                },
                "selection": {
                    "exploration_rate_override": None,
                    "group_by_beam": None,
                    "tie_breaker": None,
                },
            },
            "extensions": ANY,
        }

        run_schema: Mapping[str, Any] = {"mode": None}

        image_schema: Mapping[str, Any] = {
            "generation_path": None,
            "save_path": None,
            "upscale_path": None,
            "log_path": None,
            "caption_font_path": None,
        }

        rclone_schema: Mapping[str, Any] = {"enabled": None, "remote": None, "album": None}

        upscale_schema: Mapping[str, Any] = {
            "enabled": None,
            "target_long_edge_px": None,
            "target_width_px": None,
            "target_height_px": None,
            "target_aspect_ratio": None,
            "engine": None,
            "realesrgan_binary": None,
            "model_name": None,
            "model_path": None,
            "tile_size": None,
            "tta": None,
            "allow_fallback_resize": None,
        }

        context_schema: Mapping[str, Any] = {
            "enabled": None,
            "injectors": None,
            "injection_location": None,
            "calendar": {"enabled": None},
        }

        experiment_schema: Mapping[str, Any] = {
            "id": None,
            "variant": None,
            "notes": None,
            "tags": None,
        }

        run_review_schema: Mapping[str, Any] = {
            "enabled": None,
            "review_path": None,
        }

        unknown_keys: list[str] = []
        unknown_keys.extend(collect_unknown_keys(cfg.get("run"), run_schema, prefix="run"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("prompt"), prompt_schema, prefix="prompt"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("image"), image_schema, prefix="image"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("rclone"), rclone_schema, prefix="rclone"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("upscale"), upscale_schema, prefix="upscale"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("experiment"), experiment_schema, prefix="experiment"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("run-review"), run_review_schema, prefix="run-review"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("run_review"), run_review_schema, prefix="run_review"))

        context_payload = cfg.get("context")
        exempt_context_keys: set[str] = set()
        if isinstance(context_payload, Mapping):
            for key, value in context_payload.items():
                if not isinstance(key, str):
                    continue
                if key in context_schema:
                    continue
                # Treat unknown mapping keys as injector plugin config blocks.
                if isinstance(value, Mapping):
                    exempt_context_keys.add(key)

        unknown_keys.extend(
            collect_unknown_keys(
                context_payload,
                context_schema,
                prefix="context",
                exempt_keys=exempt_context_keys or None,
            )
        )

        if unknown_keys:
            unknown_keys = sorted(set(unknown_keys))
            if strict_unknown_keys:
                raise ValueError("Unknown config keys: " + ", ".join(unknown_keys))
            warnings.extend(f"Unknown config key: {key}" for key in unknown_keys)

        def normalize_path(value: str) -> str:
            expanded = os.path.expandvars(os.path.expanduser(value.strip()))
            if not os.path.isabs(expanded):
                nonlocal repo_root
                if repo_root is None:
                    repo_root = find_repo_root()
                expanded = os.path.join(repo_root, expanded)
            return os.path.abspath(expanded)

        def normalize_optional_path(value: str | None) -> str | None:
            if value is None:
                return None
            normalized = value.strip()
            if not normalized:
                return None
            return normalize_path(normalized)

        def get_mapping(path: str) -> Mapping[str, Any]:
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping):
                    return {}
                cur = cur.get(part)
            return cur if isinstance(cur, Mapping) else {}

        def require_str(path: str, *, reason: str | None = None) -> str:
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping) or part not in cur:
                    message = f"Missing required config: {path}"
                    if reason:
                        message += f" ({reason})"
                    raise ValueError(message)
                cur = cur[part]
            if cur is None:
                message = f"Missing required config: {path}"
                if reason:
                    message += f" ({reason})"
                raise ValueError(message)
            if not isinstance(cur, str):
                raise ValueError(f"Invalid config type for {path}: expected string")
            if not cur.strip():
                message = f"Missing required config: {path}"
                if reason:
                    message += f" ({reason})"
                raise ValueError(message)
            return cur

        def optional_str(path: str) -> str | None:
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping) or part not in cur:
                    return None
                cur = cur[part]
            if cur is None:
                return None
            if not isinstance(cur, str):
                raise ValueError(f"Invalid config type for {path}: expected string")
            if not cur.strip():
                return None
            return cur

        def optional_int(path: str) -> int | None:
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping) or part not in cur:
                    return None
                cur = cur[part]
            if cur is None:
                return None
            if isinstance(cur, bool):
                raise ValueError(f"Invalid config type for {path}: expected int, got bool")
            if isinstance(cur, int):
                return cur
            if isinstance(cur, str):
                if not cur.strip():
                    return None
                try:
                    return int(cur.strip())
                except Exception as exc:
                    raise ValueError(f"Invalid config value for {path}: must be an int") from exc
            raise ValueError(f"Invalid config type for {path}: expected int")

        def optional_bool(path: str, *, default: bool) -> bool:
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping) or part not in cur:
                    return default
                cur = cur[part]
            if cur is None:
                raise ValueError(f"Invalid boolean for {path}: None")
            return parse_bool(cur, path)

        run_cfg = get_mapping("run")
        raw_run_mode: Any = run_cfg.get("mode", "full") if run_cfg else "full"
        if raw_run_mode is None:
            raise ValueError("Unknown run.mode: None")
        if not isinstance(raw_run_mode, str):
            raise ValueError(f"Unknown run.mode: {raw_run_mode!r}")
        run_mode = raw_run_mode.strip().lower()
        if run_mode not in ("full", "prompt_only"):
            raise ValueError(f"Unknown run.mode: {raw_run_mode}")

        upscale_enabled = optional_bool("upscale.enabled", default=False)

        generation_dir_raw = optional_str("image.generation_path")
        save_dir_raw = optional_str("image.save_path")
        if generation_dir_raw and save_dir_raw:
            warnings.append(
                "Both image.generation_path and image.save_path are set; using image.generation_path."
            )
        if not generation_dir_raw and save_dir_raw:
            warnings.append(
                "Config key image.save_path is deprecated; use image.generation_path instead."
            )
            generation_dir_raw = save_dir_raw

        if run_mode == "full" and not generation_dir_raw:
            raise ValueError("Missing required config: image.generation_path")

        log_dir_raw = require_str("image.log_path")
        if run_mode == "full" and upscale_enabled:
            upscale_dir_raw = require_str(
                "image.upscale_path", reason="required when upscale.enabled=true"
            )
        else:
            upscale_dir_raw = optional_str("image.upscale_path")

        prompt_cfg = get_mapping("prompt")

        categories_path_raw = require_str("prompt.categories_path")
        profile_path_raw = require_str("prompt.profile_path")
        if run_mode == "full":
            generations_csv_raw = require_str("prompt.generations_path")
        else:
            generations_csv_raw = optional_str("prompt.generations_path")

        titles_manifest_raw = optional_str("prompt.titles_manifest_path")
        if run_mode == "full" and not titles_manifest_raw:
            assert generation_dir_raw is not None
            titles_manifest_raw = os.path.join(generation_dir_raw, "titles_manifest.csv")
            warnings.append(
                "Config key prompt.titles_manifest_path is not set; defaulting to "
                f"{titles_manifest_raw}"
            )

        caption_font_raw = optional_str("image.caption_font_path")

        random_seed = optional_int("prompt.random_seed")

        raw_experiment_cfg: Any = cfg.get("experiment")
        experiment_cfg: Mapping[str, Any]
        if raw_experiment_cfg is None:
            experiment_cfg = {}
        elif not isinstance(raw_experiment_cfg, Mapping):
            raise ValueError("Invalid config type for experiment: expected mapping")
        else:
            experiment_cfg = raw_experiment_cfg

        def normalize_experiment_text(value: Any, path: str) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError(f"Invalid config type for {path}: expected string or null")
            trimmed = value.strip()
            return trimmed or None

        experiment_id = normalize_experiment_text(experiment_cfg.get("id"), "experiment.id")
        experiment_variant = normalize_experiment_text(experiment_cfg.get("variant"), "experiment.variant")
        experiment_notes = normalize_experiment_text(experiment_cfg.get("notes"), "experiment.notes")

        tags_value = experiment_cfg.get("tags", ())
        if tags_value is None:
            raise ValueError("Invalid config type for experiment.tags: expected list[str]")
        if not isinstance(tags_value, (list, tuple)):
            raise ValueError("Invalid config type for experiment.tags: expected list[str]")
        tags: list[str] = []
        for idx, item in enumerate(tags_value):
            if not isinstance(item, str):
                raise ValueError(f"Invalid config type for experiment.tags[{idx}]: expected string")
            tag = item.strip()
            if not tag:
                raise ValueError("experiment.tags must not contain empty strings")
            tags.append(tag)

        experiment = ExperimentConfig(
            id=experiment_id,
            variant=experiment_variant,
            notes=experiment_notes,
            tags=tuple(tags),
        )

        raw_run_review_cfg: Any = cfg.get("run-review")
        raw_run_review_cfg_alt: Any = cfg.get("run_review")
        if raw_run_review_cfg is not None and raw_run_review_cfg_alt is not None:
            raise ValueError("Config cannot specify both run-review and run_review")

        run_review_cfg: Any = raw_run_review_cfg if raw_run_review_cfg is not None else raw_run_review_cfg_alt
        run_review_prefix = "run-review" if raw_run_review_cfg is not None else "run_review"

        if run_review_cfg is None:
            run_review = RunReviewConfig()
        elif not isinstance(run_review_cfg, Mapping):
            raise ValueError(f"Invalid config type for {run_review_prefix}: expected mapping")
        else:
            run_review_enabled = parse_bool(
                run_review_cfg.get("enabled", False), f"{run_review_prefix}.enabled"
            )

            raw_review_path: Any = run_review_cfg.get("review_path")
            if raw_review_path is None:
                review_path_raw = None
            elif not isinstance(raw_review_path, str):
                raise ValueError(
                    f"Invalid config type for {run_review_prefix}.review_path: expected string"
                )
            else:
                review_path_raw = raw_review_path

            if run_review_enabled and not (review_path_raw and review_path_raw.strip()):
                raise ValueError(
                    f"Missing required config: {run_review_prefix}.review_path "
                    f"(required when {run_review_prefix}.enabled=true)"
                )

            run_review = RunReviewConfig(
                enabled=run_review_enabled,
                review_path=normalize_optional_path(review_path_raw),
            )

        if "plan" not in prompt_cfg:
            prompt_plan = "auto"
        else:
            raw_plan: Any = prompt_cfg.get("plan")
            if raw_plan is None:
                raise ValueError("Invalid config value for prompt.plan: None")
            if not isinstance(raw_plan, str):
                raise ValueError("Invalid config type for prompt.plan: expected string")
            prompt_plan = raw_plan.strip().lower()
            if not prompt_plan:
                raise ValueError("Invalid config value for prompt.plan: must be a non-empty string")

        raw_refinement_cfg: Any = prompt_cfg.get("refinement")
        if raw_refinement_cfg is None:
            refinement_cfg: Mapping[str, Any] = {}
        elif not isinstance(raw_refinement_cfg, Mapping):
            raise ValueError("Invalid config type for prompt.refinement: expected mapping")
        else:
            refinement_cfg = raw_refinement_cfg

        raw_refinement_policy: Any = refinement_cfg.get("policy", "tot")
        if raw_refinement_policy is None:
            raise ValueError("Invalid config value for prompt.refinement.policy: None")
        if not isinstance(raw_refinement_policy, str):
            raise ValueError("Invalid config type for prompt.refinement.policy: expected string")
        prompt_refinement_policy = raw_refinement_policy.strip().lower()
        if not prompt_refinement_policy:
            raise ValueError("Invalid config value for prompt.refinement.policy: must be a non-empty string")

        stages_cfg_raw: Any = prompt_cfg.get("stages")
        if stages_cfg_raw is None:
            stages_cfg: Mapping[str, Any] = {}
        elif not isinstance(stages_cfg_raw, Mapping):
            raise ValueError("Invalid config type for prompt.stages: expected mapping")
        else:
            stages_cfg = stages_cfg_raw

        def parse_string_list(value: Any, path: str) -> tuple[str, ...]:
            if value is None:
                return ()
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Invalid config type for {path}: expected list[str]")
            items: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError(f"Invalid config type for {path}: expected list[str]")
                trimmed = item.strip()
                if not trimmed:
                    raise ValueError(f"Invalid config value for {path}: empty string")
                items.append(trimmed)
            return tuple(items)

        prompt_stages_include = parse_string_list(stages_cfg.get("include"), "prompt.stages.include")
        prompt_stages_exclude = parse_string_list(stages_cfg.get("exclude"), "prompt.stages.exclude")
        prompt_stages_sequence = parse_string_list(stages_cfg.get("sequence"), "prompt.stages.sequence")

        raw_overrides: Any = stages_cfg.get("overrides")
        overrides_cfg: Mapping[str, Any]
        if raw_overrides is None:
            overrides_cfg = {}
        elif not isinstance(raw_overrides, Mapping):
            raise ValueError("Invalid config type for prompt.stages.overrides: expected mapping")
        else:
            overrides_cfg = raw_overrides

        prompt_stages_overrides: dict[str, PromptStageOverride] = {}
        for raw_stage_id, raw_override in overrides_cfg.items():
            if not isinstance(raw_stage_id, str) or not raw_stage_id.strip():
                raise ValueError("prompt.stages.overrides keys must be non-empty strings")
            stage_id = raw_stage_id.strip()
            if not isinstance(raw_override, Mapping):
                raise ValueError(
                    f"Invalid config type for prompt.stages.overrides.{stage_id}: expected mapping"
                )

            allowed_keys = {"temperature", "params", "refinement_policy"}
            for key in raw_override.keys():
                if key not in allowed_keys:
                    raise ValueError(f"Unknown config key prompt.stages.overrides.{stage_id}.{key}")

            override_temperature = raw_override.get("temperature")
            temperature: float | None = None
            if override_temperature is not None:
                temperature = parse_float(
                    override_temperature, f"prompt.stages.overrides.{stage_id}.temperature"
                )

            override_params = raw_override.get("params")
            params: dict[str, Any] | None = None
            if override_params is not None:
                if not isinstance(override_params, Mapping):
                    raise ValueError(
                        f"Invalid config type for prompt.stages.overrides.{stage_id}.params: expected mapping"
                    )
                if "temperature" in override_params:
                    raise ValueError(
                        f"prompt.stages.overrides.{stage_id}.params sets temperature; use .temperature instead"
                    )
                params = dict(override_params)

            override_ref_policy = raw_override.get("refinement_policy")
            refinement_policy: str | None = None
            if override_ref_policy is not None:
                if not isinstance(override_ref_policy, str) or not override_ref_policy.strip():
                    raise ValueError(
                        f"Invalid config value for prompt.stages.overrides.{stage_id}.refinement_policy: must be a non-empty string"
                    )
                refinement_policy = override_ref_policy.strip().lower()

            prompt_stages_overrides[stage_id] = PromptStageOverride(
                temperature=temperature,
                params=params,
                refinement_policy=refinement_policy,
            )

        raw_output_cfg: Any = prompt_cfg.get("output")
        if raw_output_cfg is None:
            output_cfg: Mapping[str, Any] = {}
        elif not isinstance(raw_output_cfg, Mapping):
            raise ValueError("Invalid config type for prompt.output: expected mapping")
        else:
            output_cfg = raw_output_cfg

        raw_capture_stage = output_cfg.get("capture_stage")
        prompt_output_capture_stage: str | None
        if raw_capture_stage is None:
            prompt_output_capture_stage = None
        else:
            if not isinstance(raw_capture_stage, str):
                raise ValueError(
                    "Invalid config type for prompt.output.capture_stage: expected string or null"
                )
            prompt_output_capture_stage = raw_capture_stage.strip() or None
            if raw_capture_stage is not None and prompt_output_capture_stage is None:
                raise ValueError(
                    "Invalid config value for prompt.output.capture_stage: must be a non-empty string or null"
                )

        refine_only_cfg_raw: Any = prompt_cfg.get("refine_only")
        if refine_only_cfg_raw is None:
            refine_only_cfg: Mapping[str, Any] = {}
        elif not isinstance(refine_only_cfg_raw, Mapping):
            raise ValueError("Invalid config type for prompt.refine_only: expected mapping")
        else:
            refine_only_cfg = refine_only_cfg_raw

        prompt_refine_only_draft: str | None = None
        raw_draft = refine_only_cfg.get("draft")
        if raw_draft is not None:
            if not isinstance(raw_draft, str):
                raise ValueError("Invalid config type for prompt.refine_only.draft: expected string")
            prompt_refine_only_draft = raw_draft.strip() or None

        prompt_refine_only_draft_path_raw = optional_str("prompt.refine_only.draft_path")
        prompt_refine_only_draft_path = (
            normalize_path(prompt_refine_only_draft_path_raw) if prompt_refine_only_draft_path_raw else None
        )

        if prompt_refine_only_draft and prompt_refine_only_draft_path:
            raise ValueError(
                "Provide only one of prompt.refine_only.draft or prompt.refine_only.draft_path"
            )

        raw_scoring: Any = prompt_cfg.get("scoring")
        scoring_cfg: Mapping[str, Any]
        if raw_scoring is None:
            scoring_cfg = {}
        elif not isinstance(raw_scoring, Mapping):
            raise ValueError("Invalid config type for prompt.scoring: expected mapping")
        else:
            scoring_cfg = raw_scoring

        scoring_enabled = parse_bool(scoring_cfg.get("enabled", False), "prompt.scoring.enabled")

        num_ideas = parse_int(scoring_cfg.get("num_ideas", 6), "prompt.scoring.num_ideas")
        if num_ideas < 2:
            raise ValueError("Invalid config value for prompt.scoring.num_ideas: must be >= 2")

        exploration_rate = parse_float(
            scoring_cfg.get("exploration_rate", 0.15), "prompt.scoring.exploration_rate"
        )
        if not (0.0 <= exploration_rate <= 0.5):
            raise ValueError(
                "Invalid config value for prompt.scoring.exploration_rate: must be in [0, 0.5]"
            )

        judge_temperature = parse_float(
            scoring_cfg.get("judge_temperature", 0.0), "prompt.scoring.judge_temperature"
        )
        if scoring_enabled and judge_temperature != 0.0:
            warnings.append(
                "prompt.scoring.judge_temperature is nonzero; this may reduce determinism "
                f"(value={judge_temperature})"
            )

        judge_model = optional_str("prompt.scoring.judge_model")

        raw_judge_profile_source: Any = scoring_cfg.get("judge_profile_source", "raw")
        if raw_judge_profile_source is None:
            raise ValueError("Invalid config value for prompt.scoring.judge_profile_source: None")
        if not isinstance(raw_judge_profile_source, str):
            raise ValueError(
                "Invalid config type for prompt.scoring.judge_profile_source: expected string"
            )
        judge_profile_source = raw_judge_profile_source.strip().lower()
        if judge_profile_source not in ("raw", "generator_hints", "generator_hints_plus_dislikes"):
            raise ValueError(
                "Unknown prompt.scoring.judge_profile_source: "
                f"{raw_judge_profile_source!r} (expected: raw|generator_hints|generator_hints_plus_dislikes)"
            )

        raw_final_profile_source: Any = scoring_cfg.get("final_profile_source", "raw")
        if raw_final_profile_source is None:
            raise ValueError("Invalid config value for prompt.scoring.final_profile_source: None")
        if not isinstance(raw_final_profile_source, str):
            raise ValueError(
                "Invalid config type for prompt.scoring.final_profile_source: expected string"
            )
        final_profile_source = raw_final_profile_source.strip().lower()
        if final_profile_source not in ("raw", "generator_hints"):
            raise ValueError(
                "Unknown prompt.scoring.final_profile_source: "
                f"{raw_final_profile_source!r} (expected: raw|generator_hints)"
            )

        generator_profile_abstraction = parse_bool(
            scoring_cfg.get("generator_profile_abstraction", True),
            "prompt.scoring.generator_profile_abstraction",
        )

        if judge_profile_source in ("generator_hints", "generator_hints_plus_dislikes") and not generator_profile_abstraction:
            warnings.append(
                f"prompt.scoring.judge_profile_source={judge_profile_source} but "
                "prompt.scoring.generator_profile_abstraction=false; generator_hints will equal raw profile in blackbox.prepare"
            )

        raw_idea_profile_source: Any = scoring_cfg.get("idea_profile_source", "generator_hints")
        if raw_idea_profile_source is None:
            raise ValueError("Invalid config value for prompt.scoring.idea_profile_source: None")
        if not isinstance(raw_idea_profile_source, str):
            raise ValueError(
                "Invalid config type for prompt.scoring.idea_profile_source: expected string"
            )
        idea_profile_source = raw_idea_profile_source.strip().lower()
        if idea_profile_source not in ("raw", "generator_hints", "none"):
            raise ValueError(
                "Unknown prompt.scoring.idea_profile_source: "
                f"{raw_idea_profile_source!r} (expected: raw|generator_hints|none)"
            )

        if idea_profile_source == "generator_hints" and not generator_profile_abstraction:
            warnings.append(
                "prompt.scoring.idea_profile_source=generator_hints but "
                "prompt.scoring.generator_profile_abstraction=false; generator_hints will equal raw profile in blackbox.prepare"
            )

        raw_novelty: Any = scoring_cfg.get("novelty")
        novelty_cfg: Mapping[str, Any]
        if raw_novelty is None:
            novelty_cfg = {}
        elif not isinstance(raw_novelty, Mapping):
            raise ValueError("Invalid config type for prompt.scoring.novelty: expected mapping")
        else:
            novelty_cfg = raw_novelty

        novelty_enabled = parse_bool(
            novelty_cfg.get("enabled", True), "prompt.scoring.novelty.enabled"
        )
        novelty_window = parse_int(novelty_cfg.get("window", 25), "prompt.scoring.novelty.window")
        if novelty_window < 0:
            raise ValueError("Invalid config value for prompt.scoring.novelty.window: must be >= 0")

        raw_novelty_method: Any = novelty_cfg.get("method", "df_overlap_v1")
        if raw_novelty_method is None:
            raise ValueError("Invalid config value for prompt.scoring.novelty.method: None")
        if not isinstance(raw_novelty_method, str):
            raise ValueError("Invalid config type for prompt.scoring.novelty.method: expected string")
        novelty_method = raw_novelty_method.strip().lower()
        if novelty_method not in ("df_overlap_v1", "legacy_v0"):
            raise ValueError(
                "Unknown prompt.scoring.novelty.method: "
                f"{raw_novelty_method!r} (expected: df_overlap_v1|legacy_v0)"
            )

        df_min = parse_int(novelty_cfg.get("df_min", 3), "prompt.scoring.novelty.df_min")
        if df_min < 1:
            raise ValueError("Invalid config value for prompt.scoring.novelty.df_min: must be >= 1")

        max_motifs = parse_int(novelty_cfg.get("max_motifs", 200), "prompt.scoring.novelty.max_motifs")
        if max_motifs < 1:
            raise ValueError("Invalid config value for prompt.scoring.novelty.max_motifs: must be >= 1")

        min_token_len = parse_int(
            novelty_cfg.get("min_token_len", 3), "prompt.scoring.novelty.min_token_len"
        )
        if min_token_len < 1:
            raise ValueError(
                "Invalid config value for prompt.scoring.novelty.min_token_len: must be >= 1"
            )

        stopwords_extra = parse_string_list(
            novelty_cfg.get("stopwords_extra"), "prompt.scoring.novelty.stopwords_extra"
        )

        max_penalty = parse_int(novelty_cfg.get("max_penalty", 20), "prompt.scoring.novelty.max_penalty")
        if max_penalty < 0 or max_penalty > 100:
            raise ValueError(
                "Invalid config value for prompt.scoring.novelty.max_penalty: must be in [0, 100]"
            )

        df_cap = parse_int(novelty_cfg.get("df_cap", 10), "prompt.scoring.novelty.df_cap")
        if df_cap < 1:
            raise ValueError("Invalid config value for prompt.scoring.novelty.df_cap: must be >= 1")

        alpha_only = parse_bool(
            novelty_cfg.get("alpha_only", True), "prompt.scoring.novelty.alpha_only"
        )

        raw_scaling: Any = novelty_cfg.get("scaling", "linear")
        if raw_scaling is None:
            raise ValueError("Invalid config value for prompt.scoring.novelty.scaling: None")
        if not isinstance(raw_scaling, str):
            raise ValueError("Invalid config type for prompt.scoring.novelty.scaling: expected string")
        novelty_scaling = raw_scaling.strip().lower()
        if novelty_scaling not in ("linear", "sqrt", "quadratic"):
            raise ValueError(
                "Unknown prompt.scoring.novelty.scaling: "
                f"{raw_scaling!r} (expected: linear|sqrt|quadratic)"
            )

        prompt_scoring = PromptScoringConfig(
            enabled=scoring_enabled,
            num_ideas=num_ideas,
            exploration_rate=exploration_rate,
            judge_temperature=judge_temperature,
            judge_model=judge_model,
            judge_profile_source=judge_profile_source,  # type: ignore[arg-type]
            idea_profile_source=idea_profile_source,  # type: ignore[arg-type]
            final_profile_source=final_profile_source,  # type: ignore[arg-type]
            generator_profile_abstraction=generator_profile_abstraction,
            novelty=PromptNoveltyConfig(
                enabled=novelty_enabled,
                window=novelty_window,
                method=novelty_method,  # type: ignore[arg-type]
                df_min=df_min,
                max_motifs=max_motifs,
                min_token_len=min_token_len,
                stopwords_extra=stopwords_extra,
                max_penalty=max_penalty,
                df_cap=df_cap,
                alpha_only=alpha_only,
                scaling=novelty_scaling,  # type: ignore[arg-type]
            ),
        )

        raw_blackbox_refine: Any = prompt_cfg.get("blackbox_refine")
        blackbox_refine_cfg: Mapping[str, Any]
        blackbox_refine_present = raw_blackbox_refine is not None
        if raw_blackbox_refine is None:
            blackbox_refine_cfg = {}
        elif not isinstance(raw_blackbox_refine, Mapping):
            raise ValueError("Invalid config type for prompt.blackbox_refine: expected mapping")
        else:
            blackbox_refine_cfg = raw_blackbox_refine

        plan_requires_blackbox_refine = prompt_plan in ("blackbox_refine", "blackbox_refine_only")
        prompt_blackbox_refine: PromptBlackboxRefineConfig | None = None

        if blackbox_refine_present or plan_requires_blackbox_refine:
            enabled_default = True if plan_requires_blackbox_refine or blackbox_refine_present else False
            blackbox_refine_enabled = parse_bool(
                blackbox_refine_cfg.get("enabled", enabled_default),
                "prompt.blackbox_refine.enabled",
            )

            if plan_requires_blackbox_refine and not blackbox_refine_enabled:
                raise ValueError(
                    f"prompt.plan={prompt_plan} requires prompt.blackbox_refine.enabled=true"
                )

            iterations = parse_int(
                blackbox_refine_cfg.get("iterations", 3),
                "prompt.blackbox_refine.iterations",
            )
            if iterations < 1:
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.iterations: must be >= 1"
                )

            raw_algorithm: Any = blackbox_refine_cfg.get("algorithm", "hillclimb")
            if raw_algorithm is None:
                raise ValueError("Invalid config value for prompt.blackbox_refine.algorithm: None")
            if not isinstance(raw_algorithm, str):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.algorithm: expected string"
                )
            algorithm = raw_algorithm.strip().lower()
            if algorithm not in ("hillclimb", "beam"):
                raise ValueError(
                    "Unknown prompt.blackbox_refine.algorithm: "
                    f"{raw_algorithm!r} (expected: hillclimb|beam)"
                )

            beam_width = parse_int(
                blackbox_refine_cfg.get("beam_width", 3),
                "prompt.blackbox_refine.beam_width",
            )
            if beam_width < 1:
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.beam_width: must be >= 1"
                )
            if algorithm == "beam" and beam_width < 2:
                raise ValueError(
                    "prompt.blackbox_refine.algorithm=beam requires prompt.blackbox_refine.beam_width >= 2"
                )

            branching_factor = parse_int(
                blackbox_refine_cfg.get("branching_factor", 6),
                "prompt.blackbox_refine.branching_factor",
            )
            if branching_factor < 1:
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.branching_factor: must be >= 1"
                )

            include_parents_as_candidates = parse_bool(
                blackbox_refine_cfg.get("include_parents_as_candidates", True),
                "prompt.blackbox_refine.include_parents_as_candidates",
            )

            if algorithm == "beam":
                available_initial = branching_factor + (1 if include_parents_as_candidates else 0)
                if available_initial < beam_width:
                    raise ValueError(
                        "prompt.blackbox_refine.algorithm=beam requires enough initial candidates: "
                        "branching_factor + (include_parents_as_candidates?1:0) must be >= beam_width "
                        f"(got branching_factor={branching_factor}, include_parents_as_candidates={include_parents_as_candidates}, beam_width={beam_width})"
                    )

            generator_model = optional_str("prompt.blackbox_refine.generator_model")
            generator_temperature = parse_float(
                blackbox_refine_cfg.get("generator_temperature", 0.9),
                "prompt.blackbox_refine.generator_temperature",
            )
            if not (0.0 <= generator_temperature <= 1.5):
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.generator_temperature: "
                    "must be in [0, 1.5]"
                )

            max_prompt_chars: int | None
            if "max_prompt_chars" not in blackbox_refine_cfg:
                max_prompt_chars = 3500
            else:
                raw_max_chars = blackbox_refine_cfg.get("max_prompt_chars")
                if raw_max_chars is None:
                    max_prompt_chars = None
                else:
                    max_prompt_chars = parse_int(raw_max_chars, "prompt.blackbox_refine.max_prompt_chars")
                    if max_prompt_chars <= 0:
                        raise ValueError(
                            "Invalid config value for prompt.blackbox_refine.max_prompt_chars: must be > 0 or null"
                        )

            raw_variation_prompt: Any = blackbox_refine_cfg.get("variation_prompt")
            variation_prompt_cfg: Mapping[str, Any]
            if raw_variation_prompt is None:
                variation_prompt_cfg = {}
            elif not isinstance(raw_variation_prompt, Mapping):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.variation_prompt: expected mapping"
                )
            else:
                variation_prompt_cfg = raw_variation_prompt

            raw_template: Any = variation_prompt_cfg.get("template", "v1")
            if raw_template is None:
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.variation_prompt.template: None"
                )
            if not isinstance(raw_template, str):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.variation_prompt.template: expected string"
                )
            template = raw_template.strip().lower()
            if template not in ("v1", "v2"):
                raise ValueError(
                    "Unknown prompt.blackbox_refine.variation_prompt.template: "
                    f"{raw_template!r} (expected: v1|v2)"
                )

            include_concepts = parse_bool(
                variation_prompt_cfg.get("include_concepts", True),
                "prompt.blackbox_refine.variation_prompt.include_concepts",
            )
            include_context_guidance = parse_bool(
                variation_prompt_cfg.get("include_context_guidance", False),
                "prompt.blackbox_refine.variation_prompt.include_context_guidance",
            )
            include_profile = parse_bool(
                variation_prompt_cfg.get("include_profile", True),
                "prompt.blackbox_refine.variation_prompt.include_profile",
            )

            raw_profile_source: Any = variation_prompt_cfg.get("profile_source", "likes_dislikes")
            if raw_profile_source is None:
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.variation_prompt.profile_source: None"
                )
            if not isinstance(raw_profile_source, str):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.variation_prompt.profile_source: expected string"
                )
            profile_source = raw_profile_source.strip().lower()
            if profile_source not in (
                "raw",
                "generator_hints",
                "likes_dislikes",
                "dislikes_only",
                "combined",
            ):
                raise ValueError(
                    "Unknown prompt.blackbox_refine.variation_prompt.profile_source: "
                    f"{raw_profile_source!r} (expected: raw|generator_hints|likes_dislikes|dislikes_only|combined)"
                )

            if (
                include_profile
                and profile_source == "generator_hints"
                and not prompt_scoring.generator_profile_abstraction
            ):
                warnings.append(
                    "prompt.blackbox_refine.variation_prompt.profile_source=generator_hints but "
                    "prompt.scoring.generator_profile_abstraction=false; generator_hints will equal raw profile in blackbox.prepare"
                )

            include_novelty_summary = parse_bool(
                variation_prompt_cfg.get("include_novelty_summary", True),
                "prompt.blackbox_refine.variation_prompt.include_novelty_summary",
            )
            include_mutation_directive = parse_bool(
                variation_prompt_cfg.get("include_mutation_directive", True),
                "prompt.blackbox_refine.variation_prompt.include_mutation_directive",
            )
            include_scoring_rubric = parse_bool(
                variation_prompt_cfg.get("include_scoring_rubric", True),
                "prompt.blackbox_refine.variation_prompt.include_scoring_rubric",
            )

            variation_prompt = PromptBlackboxRefineVariationPromptConfig(
                template=template,  # type: ignore[arg-type]
                include_concepts=include_concepts,
                include_context_guidance=include_context_guidance,
                include_profile=include_profile,
                profile_source=profile_source,  # type: ignore[arg-type]
                include_novelty_summary=include_novelty_summary,
                include_mutation_directive=include_mutation_directive,
                include_scoring_rubric=include_scoring_rubric,
            )

            raw_mutation_directives: Any = blackbox_refine_cfg.get("mutation_directives")
            mutation_directives_cfg: Mapping[str, Any]
            if raw_mutation_directives is None:
                mutation_directives_cfg = {}
            elif not isinstance(raw_mutation_directives, Mapping):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.mutation_directives: expected mapping"
                )
            else:
                mutation_directives_cfg = raw_mutation_directives

            raw_mutation_mode: Any = mutation_directives_cfg.get("mode", "random")
            if raw_mutation_mode is None:
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.mutation_directives.mode: None"
                )
            if not isinstance(raw_mutation_mode, str):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.mutation_directives.mode: expected string"
                )
            mutation_mode = raw_mutation_mode.strip().lower()
            if mutation_mode not in ("none", "random", "cycle", "fixed"):
                raise ValueError(
                    "Unknown prompt.blackbox_refine.mutation_directives.mode: "
                    f"{raw_mutation_mode!r} (expected: none|random|cycle|fixed)"
                )

            default_directives = (
                "Mutate composition and camera perspective while preserving subject and mood.",
                "Increase specificity: add concrete materials, lighting, and lens details.",
                "Introduce a subtle narrative twist; avoid clichés.",
            )
            directives_raw: Any
            if "directives" in mutation_directives_cfg:
                directives_raw = mutation_directives_cfg.get("directives")
                directives = parse_string_list(
                    directives_raw,
                    "prompt.blackbox_refine.mutation_directives.directives",
                )
            else:
                directives = tuple(default_directives)

            if mutation_mode != "none" and not directives:
                raise ValueError(
                    "prompt.blackbox_refine.mutation_directives.mode requires prompt.blackbox_refine.mutation_directives.directives"
                )

            raw_per_axis: Any = mutation_directives_cfg.get("per_axis")
            per_axis_cfg: Mapping[str, Any]
            if raw_per_axis is None:
                per_axis_cfg = {}
            elif not isinstance(raw_per_axis, Mapping):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.mutation_directives.per_axis: expected mapping"
                )
            else:
                per_axis_cfg = raw_per_axis

            per_axis_enabled = parse_bool(
                per_axis_cfg.get("enabled", False),
                "prompt.blackbox_refine.mutation_directives.per_axis.enabled",
            )
            per_axis_axes = parse_string_list(
                per_axis_cfg.get("axes"),
                "prompt.blackbox_refine.mutation_directives.per_axis.axes",
            )
            if per_axis_enabled:
                raise ValueError(
                    "prompt.blackbox_refine.mutation_directives.per_axis.enabled=true is not implemented yet"
                )

            mutation_directives = PromptBlackboxRefineMutationDirectivesConfig(
                mode=mutation_mode,  # type: ignore[arg-type]
                directives=tuple(directives),
                per_axis=PromptBlackboxRefineMutationPerAxisConfig(
                    enabled=per_axis_enabled,
                    axes=tuple(per_axis_axes),
                ),
            )

            raw_judging: Any = blackbox_refine_cfg.get("judging")
            judging_cfg: Mapping[str, Any]
            if raw_judging is None:
                judging_cfg = {}
            elif not isinstance(raw_judging, Mapping):
                raise ValueError("Invalid config type for prompt.blackbox_refine.judging: expected mapping")
            else:
                judging_cfg = raw_judging

            judges_list_raw = judging_cfg.get("judges")
            judges: list[PromptBlackboxRefineJudgeConfig] = []
            if judges_list_raw is None:
                warnings.append(
                    "prompt.blackbox_refine.judging.judges not set; defaulting to one judge (id=j1)"
                )
                judges.append(
                    PromptBlackboxRefineJudgeConfig(
                        id="j1",
                        model=None,
                        temperature=None,
                        weight=1.0,
                        rubric="default",  # type: ignore[arg-type]
                    )
                )
            else:
                if not isinstance(judges_list_raw, list):
                    raise ValueError(
                        "Invalid config type for prompt.blackbox_refine.judging.judges: expected list[object]"
                    )
                if not judges_list_raw:
                    raise ValueError("prompt.blackbox_refine.judging.judges must not be empty")

                seen_judge_ids: set[str] = set()
                for idx, judge_raw in enumerate(judges_list_raw):
                    if not isinstance(judge_raw, Mapping):
                        raise ValueError(
                            f"Invalid config type for prompt.blackbox_refine.judging.judges[{idx}]: expected mapping"
                        )

                    allowed_judge_keys = {"id", "model", "temperature", "weight", "rubric"}
                    for key in judge_raw.keys():
                        if key not in allowed_judge_keys:
                            raise ValueError(
                                f"Unknown config key prompt.blackbox_refine.judging.judges[{idx}].{key}"
                            )

                    raw_judge_id = judge_raw.get("id")
                    if raw_judge_id is None:
                        raise ValueError(
                            f"Missing required config: prompt.blackbox_refine.judging.judges[{idx}].id"
                        )
                    if not isinstance(raw_judge_id, str) or not raw_judge_id.strip():
                        raise ValueError(
                            f"Invalid config value for prompt.blackbox_refine.judging.judges[{idx}].id: must be a non-empty string"
                        )
                    judge_id = raw_judge_id.strip()
                    if judge_id in seen_judge_ids:
                        raise ValueError(
                            f"Duplicate judge id in prompt.blackbox_refine.judging.judges: {judge_id}"
                        )
                    seen_judge_ids.add(judge_id)

                    judge_model: str | None = None
                    raw_judge_model = judge_raw.get("model")
                    if raw_judge_model is not None:
                        if not isinstance(raw_judge_model, str):
                            raise ValueError(
                                f"Invalid config type for prompt.blackbox_refine.judging.judges[{idx}].model: expected string or null"
                            )
                        judge_model = raw_judge_model.strip() or None
                        if raw_judge_model is not None and judge_model is None:
                            raise ValueError(
                                f"Invalid config value for prompt.blackbox_refine.judging.judges[{idx}].model: must be a non-empty string or null"
                            )

                    judge_temp: float | None = None
                    raw_judge_temp = judge_raw.get("temperature")
                    if raw_judge_temp is not None:
                        judge_temp = parse_float(
                            raw_judge_temp,
                            f"prompt.blackbox_refine.judging.judges[{idx}].temperature",
                        )
                        if not (0.0 <= judge_temp <= 1.5):
                            raise ValueError(
                                f"Invalid config value for prompt.blackbox_refine.judging.judges[{idx}].temperature: must be in [0, 1.5]"
                            )

                    judge_weight = parse_float(
                        judge_raw.get("weight", 1.0),
                        f"prompt.blackbox_refine.judging.judges[{idx}].weight",
                    )
                    if judge_weight <= 0:
                        raise ValueError(
                            f"Invalid config value for prompt.blackbox_refine.judging.judges[{idx}].weight: must be > 0"
                        )

                    raw_rubric: Any = judge_raw.get("rubric", "default")
                    if raw_rubric is None:
                        raise ValueError(
                            f"Invalid config value for prompt.blackbox_refine.judging.judges[{idx}].rubric: None"
                        )
                    if not isinstance(raw_rubric, str):
                        raise ValueError(
                            f"Invalid config type for prompt.blackbox_refine.judging.judges[{idx}].rubric: expected string"
                        )
                    rubric = raw_rubric.strip().lower()
                    if rubric not in ("default", "strict", "novelty_heavy"):
                        raise ValueError(
                            f"Unknown prompt.blackbox_refine.judging.judges[{idx}].rubric: "
                            f"{raw_rubric!r} (expected: default|strict|novelty_heavy)"
                        )

                    judges.append(
                        PromptBlackboxRefineJudgeConfig(
                            id=judge_id,
                            model=judge_model,
                            temperature=judge_temp,
                            weight=float(judge_weight),
                            rubric=rubric,  # type: ignore[arg-type]
                        )
                    )

            raw_aggregation: Any = judging_cfg.get("aggregation", "mean")
            if raw_aggregation is None:
                raise ValueError("Invalid config value for prompt.blackbox_refine.judging.aggregation: None")
            if not isinstance(raw_aggregation, str):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.judging.aggregation: expected string"
                )
            aggregation = raw_aggregation.strip().lower()
            if aggregation not in ("mean", "median", "min", "max", "trimmed_mean"):
                raise ValueError(
                    "Unknown prompt.blackbox_refine.judging.aggregation: "
                    f"{raw_aggregation!r} (expected: mean|median|min|max|trimmed_mean)"
                )

            trimmed_mean_drop = parse_int(
                judging_cfg.get("trimmed_mean_drop", 0),
                "prompt.blackbox_refine.judging.trimmed_mean_drop",
            )
            if trimmed_mean_drop < 0:
                raise ValueError(
                    "Invalid config value for prompt.blackbox_refine.judging.trimmed_mean_drop: must be >= 0"
                )
            if aggregation != "trimmed_mean" and "trimmed_mean_drop" in judging_cfg and trimmed_mean_drop != 0:
                raise ValueError(
                    "prompt.blackbox_refine.judging.trimmed_mean_drop is only valid when aggregation=trimmed_mean"
                )
            if aggregation == "trimmed_mean" and trimmed_mean_drop * 2 >= len(judges):
                raise ValueError(
                    "prompt.blackbox_refine.judging.trimmed_mean_drop is too large for judge count "
                    f"(drop={trimmed_mean_drop}, judges={len(judges)})"
                )

            judging = PromptBlackboxRefineJudgingConfig(
                judges=tuple(judges),
                aggregation=aggregation,  # type: ignore[arg-type]
                trimmed_mean_drop=trimmed_mean_drop,
            )

            raw_selection: Any = blackbox_refine_cfg.get("selection")
            selection_cfg: Mapping[str, Any]
            if raw_selection is None:
                selection_cfg = {}
            elif not isinstance(raw_selection, Mapping):
                raise ValueError("Invalid config type for prompt.blackbox_refine.selection: expected mapping")
            else:
                selection_cfg = raw_selection

            raw_expl_override = selection_cfg.get("exploration_rate_override")
            exploration_rate_override: float | None = None
            if raw_expl_override is not None:
                exploration_rate_override = parse_float(
                    raw_expl_override, "prompt.blackbox_refine.selection.exploration_rate_override"
                )
                if not (0.0 <= exploration_rate_override <= 0.5):
                    raise ValueError(
                        "Invalid config value for prompt.blackbox_refine.selection.exploration_rate_override: must be in [0, 0.5]"
                    )

            group_by_beam = parse_bool(
                selection_cfg.get("group_by_beam", False),
                "prompt.blackbox_refine.selection.group_by_beam",
            )

            raw_tie_breaker: Any = selection_cfg.get("tie_breaker", "stable_id")
            if raw_tie_breaker is None:
                raise ValueError("Invalid config value for prompt.blackbox_refine.selection.tie_breaker: None")
            if not isinstance(raw_tie_breaker, str):
                raise ValueError(
                    "Invalid config type for prompt.blackbox_refine.selection.tie_breaker: expected string"
                )
            tie_breaker = raw_tie_breaker.strip().lower()
            if tie_breaker not in ("stable_id", "prefer_shorter", "prefer_novel"):
                raise ValueError(
                    "Unknown prompt.blackbox_refine.selection.tie_breaker: "
                    f"{raw_tie_breaker!r} (expected: stable_id|prefer_shorter|prefer_novel)"
                )

            selection = PromptBlackboxRefineSelectionConfig(
                exploration_rate_override=exploration_rate_override,
                group_by_beam=group_by_beam,
                tie_breaker=tie_breaker,  # type: ignore[arg-type]
            )

            prompt_blackbox_refine = PromptBlackboxRefineConfig(
                enabled=blackbox_refine_enabled,
                iterations=iterations,
                algorithm=algorithm,  # type: ignore[arg-type]
                beam_width=beam_width,
                branching_factor=branching_factor,
                include_parents_as_candidates=include_parents_as_candidates,
                generator_model=generator_model,
                generator_temperature=generator_temperature,
                max_prompt_chars=max_prompt_chars,
                variation_prompt=variation_prompt,
                mutation_directives=mutation_directives,
                judging=judging,
                selection=selection,
            )

        concepts_cfg_raw: Any = prompt_cfg.get("concepts")
        concepts_cfg: Mapping[str, Any]
        if concepts_cfg_raw is None:
            concepts_cfg = {}
        elif not isinstance(concepts_cfg_raw, Mapping):
            raise ValueError("Invalid config type for prompt.concepts: expected mapping")
        else:
            concepts_cfg = concepts_cfg_raw

        selection_cfg_raw: Any = concepts_cfg.get("selection")
        selection_cfg: Mapping[str, Any]
        if selection_cfg_raw is None:
            selection_cfg = {}
        elif not isinstance(selection_cfg_raw, Mapping):
            raise ValueError("Invalid config type for prompt.concepts.selection: expected mapping")
        else:
            selection_cfg = selection_cfg_raw

        raw_strategy: Any = selection_cfg.get("strategy", "random")
        if raw_strategy is None:
            raise ValueError("Invalid config value for prompt.concepts.selection.strategy: None")
        if not isinstance(raw_strategy, str):
            raise ValueError(
                "Invalid config type for prompt.concepts.selection.strategy: expected string"
            )
        selection_strategy = raw_strategy.strip().lower()
        if not selection_strategy:
            raise ValueError(
                "Invalid config value for prompt.concepts.selection.strategy: must be a non-empty string"
            )
        if selection_strategy not in ("random", "fixed", "file"):
            raise ValueError(
                "Unknown prompt.concepts.selection.strategy: "
                f"{raw_strategy!r} (expected: random|fixed|file)"
            )

        selection_fixed = parse_string_list(
            selection_cfg.get("fixed"),
            "prompt.concepts.selection.fixed",
        )

        selection_file_path_raw = optional_str("prompt.concepts.selection.file_path")
        selection_file_path = (
            normalize_path(selection_file_path_raw) if selection_file_path_raw else None
        )

        if selection_strategy == "fixed" and not selection_fixed:
            raise ValueError(
                "prompt.concepts.selection.strategy=fixed requires prompt.concepts.selection.fixed"
            )
        if selection_strategy == "file":
            if not selection_file_path:
                raise ValueError(
                    "prompt.concepts.selection.strategy=file requires prompt.concepts.selection.file_path"
                )
            if not os.path.exists(selection_file_path):
                raise ValueError(
                    "prompt.concepts.selection.file_path not found: "
                    f"{selection_file_path_raw}"
                )
            file_concepts: list[str] = []
            with open(selection_file_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = str(raw_line).strip()
                    if not line or line.startswith("#"):
                        continue
                    file_concepts.append(line)
            if not file_concepts:
                raise ValueError(
                    "prompt.concepts.selection.file_path is empty (no non-empty lines found)"
                )
            selection_fixed = tuple(file_concepts)

        filters_cfg_raw: Any = concepts_cfg.get("filters")
        filters_cfg: Mapping[str, Any]
        if filters_cfg_raw is None:
            filters_cfg = {}
        elif not isinstance(filters_cfg_raw, Mapping):
            raise ValueError("Invalid config type for prompt.concepts.filters: expected mapping")
        else:
            filters_cfg = filters_cfg_raw

        concept_filters_enabled = parse_bool(
            filters_cfg.get("enabled", True), "prompt.concepts.filters.enabled"
        )

        order_default: list[str] = ["dislike_rewrite"]
        order_raw = filters_cfg.get("order", order_default)
        concept_filter_order = tuple(
            item.strip().lower()
            for item in parse_string_list(order_raw, "prompt.concepts.filters.order")
        )

        dislike_rewrite_cfg_raw: Any = filters_cfg.get("dislike_rewrite")
        dislike_rewrite_cfg: Mapping[str, Any]
        if dislike_rewrite_cfg_raw is None:
            dislike_rewrite_cfg = {}
        elif not isinstance(dislike_rewrite_cfg_raw, Mapping):
            raise ValueError(
                "Invalid config type for prompt.concepts.filters.dislike_rewrite: expected mapping"
            )
        else:
            dislike_rewrite_cfg = dislike_rewrite_cfg_raw

        dislike_rewrite_enabled = parse_bool(
            dislike_rewrite_cfg.get("enabled", True),
            "prompt.concepts.filters.dislike_rewrite.enabled",
        )
        dislike_rewrite_temperature = parse_float(
            dislike_rewrite_cfg.get("temperature", 0.25),
            "prompt.concepts.filters.dislike_rewrite.temperature",
        )
        if not (0.0 <= dislike_rewrite_temperature <= 2.0):
            raise ValueError(
                "Invalid config value for prompt.concepts.filters.dislike_rewrite.temperature: "
                f"must be in [0, 2] (got {dislike_rewrite_temperature})"
            )

        known_concept_filters = {"dislike_rewrite"}
        unknown_filters = sorted(set(concept_filter_order) - known_concept_filters)
        if unknown_filters:
            raise ValueError(
                "Unknown prompt.concepts.filters.order entries: "
                f"{unknown_filters} (known: {sorted(known_concept_filters)})"
            )

        prompt_concepts = PromptConceptsConfig(
            selection=PromptConceptSelectionConfig(
                strategy=selection_strategy,  # type: ignore[arg-type]
                fixed=selection_fixed,
                file_path=selection_file_path,
            ),
            filters=PromptConceptFiltersConfig(
                enabled=concept_filters_enabled,
                order=concept_filter_order,
                dislike_rewrite=DislikeRewriteFilterConfig(
                    enabled=dislike_rewrite_enabled,
                    temperature=dislike_rewrite_temperature,
                ),
            ),
        )

        context_cfg_raw = get_mapping("context")
        context_enabled = optional_bool("context.enabled", default=False)
        raw_injection_location: Any = context_cfg_raw.get("injection_location", "system")
        if raw_injection_location is None:
            raise ValueError("Invalid config value for context.injection_location: None")
        if not isinstance(raw_injection_location, str):
            raise ValueError(
                "Invalid config type for context.injection_location: expected string"
            )
        context_injection_location = raw_injection_location.strip().lower() or "system"
        if context_injection_location not in ("system", "prompt", "both"):
            raise ValueError(
                "Invalid config value for context.injection_location: "
                f"{raw_injection_location!r} (expected: system|prompt|both)"
            )

        calendar_enabled = optional_bool("context.calendar.enabled", default=False)
        if calendar_enabled:
            raise ValueError(
                "Calendar context injector is not implemented yet. Set context.calendar.enabled=false (default) "
                "or remove 'calendar' from context.injectors."
            )

        context_injectors: tuple[str, ...] = ()
        if context_enabled:
            injectors_value = context_cfg_raw.get("injectors")
            if injectors_value is None:
                context_injectors = ("season", "holiday")
                warnings.append(
                    "Config key context.injectors is not set; defaulting to ['season', 'holiday']"
                )
            else:
                if not isinstance(injectors_value, list):
                    raise ValueError("Invalid config type for context.injectors: expected list[str]")
                normalized: list[str] = []
                for idx, item in enumerate(injectors_value):
                    if not isinstance(item, str):
                        raise ValueError(
                            f"Invalid config type for context.injectors[{idx}]: expected string"
                        )
                    name = item.strip().lower()
                    if not name:
                        raise ValueError("context.injectors must not contain empty strings")
                    normalized.append(name)
                context_injectors = tuple(normalized)

            if "calendar" in context_injectors:
                raise ValueError(
                    "Calendar context injector is not implemented yet. Set context.calendar.enabled=false (default) "
                    "or remove 'calendar' from context.injectors."
                )

            known = set(ContextManager.available_injectors())
            for injector in context_injectors:
                if injector not in known:
                    raise ValueError(f"Unknown context injector: {injector}")

        context_cfg: Mapping[str, Any] = dict(context_cfg_raw)
        context_cfg.pop("enabled", None)
        context_cfg.pop("injectors", None)
        context_cfg.pop("injection_location", None)

        rclone_enabled = optional_bool("rclone.enabled", default=False)
        rclone_remote = optional_str("rclone.remote")
        rclone_album = optional_str("rclone.album")
        if run_mode == "full" and rclone_enabled:
            if not rclone_remote:
                raise ValueError(
                    "Missing required config: rclone.remote (required when rclone.enabled=true)"
                )
            if not rclone_album:
                raise ValueError(
                    "Missing required config: rclone.album (required when rclone.enabled=true)"
                )

        upscale_cfg = get_mapping("upscale")
        upscale_enabled = optional_bool("upscale.enabled", default=False)
        upscale_target_long_edge_px = (
            int(upscale_cfg.get("target_long_edge_px", 3840)) if upscale_cfg else 3840
        )
        if upscale_target_long_edge_px <= 0:
            raise ValueError("Invalid config value for upscale.target_long_edge_px: must be > 0")

        upscale_target_width_px = optional_int("upscale.target_width_px")
        upscale_target_height_px = optional_int("upscale.target_height_px")
        if (upscale_target_width_px is None) != (upscale_target_height_px is None):
            raise ValueError(
                "Both upscale.target_width_px and upscale.target_height_px must be provided together"
            )
        if upscale_target_width_px is not None and upscale_target_width_px <= 0:
            raise ValueError("Invalid config value for upscale.target_width_px: must be > 0")
        if upscale_target_height_px is not None and upscale_target_height_px <= 0:
            raise ValueError("Invalid config value for upscale.target_height_px: must be > 0")

        raw_aspect_ratio = upscale_cfg.get("target_aspect_ratio") if upscale_cfg else None
        upscale_target_aspect_ratio: float | None = None
        if raw_aspect_ratio is not None:
            upscale_target_aspect_ratio = parse_aspect_ratio(
                raw_aspect_ratio, "upscale.target_aspect_ratio"
            )

        if upscale_target_aspect_ratio is not None and upscale_target_width_px is not None:
            raise ValueError(
                "Provide either upscale.target_width_px/target_height_px or "
                "upscale.target_aspect_ratio, not both"
            )
        upscale_engine = (
            str(upscale_cfg.get("engine", "realesrgan-ncnn-vulkan"))
            if upscale_cfg
            else "realesrgan-ncnn-vulkan"
        )
        upscale_realesrgan_binary = optional_str("upscale.realesrgan_binary")
        upscale_model_name = (
            str(upscale_cfg.get("model_name", "realesrgan-x4plus"))
            if upscale_cfg
            else "realesrgan-x4plus"
        )
        upscale_model_path = optional_str("upscale.model_path")
        upscale_tile_size = int(upscale_cfg.get("tile_size", 0)) if upscale_cfg else 0
        upscale_tta = optional_bool("upscale.tta", default=False)
        upscale_allow_fallback_resize = optional_bool("upscale.allow_fallback_resize", default=False)

        return (
            RunConfig(
                generation_dir=normalize_optional_path(generation_dir_raw),
                upscale_dir=normalize_optional_path(upscale_dir_raw),
                log_dir=normalize_path(log_dir_raw),
                run_mode=run_mode,  # type: ignore[arg-type]
                experiment=experiment,
                run_review=run_review,
                categories_path=normalize_path(categories_path_raw),
                profile_path=normalize_path(profile_path_raw),
                generations_csv_path=normalize_optional_path(generations_csv_raw),
                titles_manifest_path=normalize_optional_path(titles_manifest_raw),
                caption_font_path=normalize_path(caption_font_raw) if caption_font_raw else None,
                random_seed=random_seed,
                prompt_scoring=prompt_scoring,
                prompt_blackbox_refine=prompt_blackbox_refine,
                prompt_concepts=prompt_concepts,
                prompt_plan=prompt_plan,
                prompt_refinement_policy=prompt_refinement_policy,
                prompt_stages_include=prompt_stages_include,
                prompt_stages_exclude=prompt_stages_exclude,
                prompt_stages_sequence=prompt_stages_sequence,
                prompt_stages_overrides=prompt_stages_overrides,
                prompt_output_capture_stage=prompt_output_capture_stage,
                prompt_refine_only_draft=prompt_refine_only_draft,
                prompt_refine_only_draft_path=prompt_refine_only_draft_path,
                context_enabled=context_enabled,
                context_injection_location=context_injection_location,  # type: ignore[arg-type]
                context_injectors=context_injectors,
                context_cfg=context_cfg,
                rclone_enabled=rclone_enabled,
                rclone_remote=rclone_remote,
                rclone_album=rclone_album,
                upscale_enabled=upscale_enabled,
                upscale_target_long_edge_px=upscale_target_long_edge_px,
                upscale_target_width_px=upscale_target_width_px,
                upscale_target_height_px=upscale_target_height_px,
                upscale_target_aspect_ratio=upscale_target_aspect_ratio,
                upscale_engine=upscale_engine,
                upscale_realesrgan_binary=(
                    normalize_path(upscale_realesrgan_binary) if upscale_realesrgan_binary else None
                ),
                upscale_model_name=upscale_model_name,
                upscale_model_path=normalize_path(upscale_model_path) if upscale_model_path else None,
                upscale_tile_size=upscale_tile_size,
                upscale_tta=upscale_tta,
                upscale_allow_fallback_resize=upscale_allow_fallback_resize,
            ),
            warnings,
        )
