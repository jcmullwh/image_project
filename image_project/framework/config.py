from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from image_project.foundation.config_io import find_repo_root
from image_project.framework.context import ContextManager


RunMode = Literal["full", "prompt_only"]
ScoringProfileSource = Literal["raw", "generator_hints"]


@dataclass(frozen=True)
class ExperimentConfig:
    id: str | None = None
    variant: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class PromptScoringConfig:
    enabled: bool
    num_ideas: int
    exploration_rate: float
    judge_temperature: float
    judge_model: str | None
    judge_profile_source: ScoringProfileSource
    final_profile_source: ScoringProfileSource
    generator_profile_abstraction: bool
    novelty: PromptNoveltyConfig


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

    categories_path: str
    profile_path: str
    generations_csv_path: str | None

    titles_manifest_path: str | None
    caption_font_path: str | None

    random_seed: int | None
    prompt_scoring: PromptScoringConfig
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
                "final_profile_source": None,
                "generator_profile_abstraction": None,
                "novelty": {"enabled": None, "window": None},
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

        unknown_keys: list[str] = []
        unknown_keys.extend(collect_unknown_keys(cfg.get("run"), run_schema, prefix="run"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("prompt"), prompt_schema, prefix="prompt"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("image"), image_schema, prefix="image"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("rclone"), rclone_schema, prefix="rclone"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("upscale"), upscale_schema, prefix="upscale"))
        unknown_keys.extend(collect_unknown_keys(cfg.get("experiment"), experiment_schema, prefix="experiment"))

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
        if judge_profile_source not in ("raw", "generator_hints"):
            raise ValueError(
                "Unknown prompt.scoring.judge_profile_source: "
                f"{raw_judge_profile_source!r} (expected: raw|generator_hints)"
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

        prompt_scoring = PromptScoringConfig(
            enabled=scoring_enabled,
            num_ideas=num_ideas,
            exploration_rate=exploration_rate,
            judge_temperature=judge_temperature,
            judge_model=judge_model,
            judge_profile_source=judge_profile_source,  # type: ignore[arg-type]
            final_profile_source=final_profile_source,  # type: ignore[arg-type]
            generator_profile_abstraction=generator_profile_abstraction,
            novelty=PromptNoveltyConfig(enabled=novelty_enabled, window=novelty_window),
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
                categories_path=normalize_path(categories_path_raw),
                profile_path=normalize_path(profile_path_raw),
                generations_csv_path=normalize_optional_path(generations_csv_raw),
                titles_manifest_path=normalize_optional_path(titles_manifest_raw),
                caption_font_path=normalize_path(caption_font_raw) if caption_font_raw else None,
                random_seed=random_seed,
                prompt_scoring=prompt_scoring,
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
