from __future__ import annotations

"""Prompt pipeline configuration parsing (routing only).

This module owns parsing of prompt-pipeline configuration that is *not* part of
framework infra (`RunConfig`) and is *not* stage-owned knob schema validation.

Key invariants:
- Stage knob schemas are validated only by the stage that consumes them via
  `pipelinekit.config_namespace.ConfigNamespace`.
- This module may validate routing structure (lists/mappings/strings) and known
  keys, but must treat stage knob payloads as opaque.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from image_project.foundation.config_io import find_repo_root
from image_project.framework.config import RunMode, parse_bool, parse_float


@dataclass(frozen=True)
class StageOverride:
    """Pipeline-level stage override (routing only).

    This is not stage-owned configuration; it is an orchestration patch applied
    by the pipeline compiler (e.g., temperature overrides for chat stages).
    """

    temperature: float | None = None
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class PipelineOverrides:
    """Prompt pipeline selection modifiers (routing only)."""

    include: tuple[str, ...]
    exclude: tuple[str, ...]
    sequence: tuple[Any, ...]
    overrides: Mapping[str, StageOverride]
    capture_stage: str | None


@dataclass(frozen=True)
class StageConfigRouting:
    """Parsed `prompt.stage_configs` payload (routing only; knobs are opaque)."""

    defaults: Mapping[str, Mapping[str, Any]]
    instances: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class RefineOnlyInputs:
    """Optional prompt inputs for plans that refine an existing draft."""

    draft: str | None
    draft_path: str | None


@dataclass(frozen=True)
class PromptPipelineConfig:
    """Parsed prompt pipeline configuration (non-infra, non-stage-schema)."""

    categories_path: str
    profile_path: str
    generations_csv_path: str | None
    titles_manifest_path: str | None
    random_seed: int | None

    plan: str
    refine_only: RefineOnlyInputs

    stages: PipelineOverrides
    stage_configs: StageConfigRouting

    @property
    def stage_configs_defaults(self) -> Mapping[str, Mapping[str, Any]]:
        """Convenience accessor for `pipelinekit.compiler.compile_stage_nodes`."""

        return self.stage_configs.defaults

    @property
    def stage_configs_instances(self) -> Mapping[str, Mapping[str, Any]]:
        """Convenience accessor for `pipelinekit.compiler.compile_stage_nodes`."""

        return self.stage_configs.instances

    @staticmethod
    def from_root_dict(
        cfg: Mapping[str, Any],
        *,
        run_mode: RunMode,
        generation_dir: str | None,
    ) -> tuple["PromptPipelineConfig", list[str]]:
        """Parse prompt pipeline config from the root config mapping.

        Args:
            cfg: Root YAML mapping loaded via `foundation.config_io.load_config()`.
            run_mode: Parsed run mode (needed for conditional requirements).
            generation_dir: Normalized generation output directory (used only for
                loud defaulting of `prompt.titles_manifest_path` in full mode).

        Returns:
            (PromptPipelineConfig, warnings)

        Raises:
            ValueError: If required keys are missing/invalid.
        """

        if not isinstance(cfg, Mapping):
            raise ValueError("Config must be a mapping")

        warnings: list[str] = []

        strict_unknown_keys = False
        if "strict" in cfg:
            strict_unknown_keys = parse_bool(cfg.get("strict"), "strict")

        raw_prompt: Any = cfg.get("prompt")
        if raw_prompt is None:
            prompt_cfg: Mapping[str, Any] = {}
        elif not isinstance(raw_prompt, Mapping):
            raise ValueError("Invalid config type for prompt: expected mapping")
        else:
            prompt_cfg = raw_prompt

        # Explicitly reject legacy prompt feature blocks (no silent ignore).
        removed_blocks = ("scoring", "blackbox_refine", "concepts")
        present_removed = [key for key in removed_blocks if key in prompt_cfg]
        if present_removed:
            raise ValueError(
                "Removed prompt config blocks present: "
                + ", ".join(f"prompt.{key}" for key in present_removed)
                + ". Move stage knobs under prompt.stage_configs.{defaults,instances}."
            )

        repo_root: str | None = None

        def normalize_path(value: str, *, path: str) -> str:
            """Normalize a (possibly relative) path, anchored at the repo root."""

            if not isinstance(value, str):
                raise TypeError(f"{path} must be a string (type={type(value).__name__})")
            text = value.strip()
            if not text:
                raise ValueError(f"{path} must be a non-empty string")
            expanded = os.path.expandvars(os.path.expanduser(text))
            if not os.path.isabs(expanded):
                nonlocal repo_root
                if repo_root is None:
                    repo_root = find_repo_root()
                expanded = os.path.join(repo_root, expanded)
            return os.path.abspath(expanded)

        def optional_path(value: Any, *, path: str) -> str | None:
            """Normalize an optional path; return None when missing/empty."""

            if value is None:
                return None
            if not isinstance(value, str):
                raise TypeError(f"{path} must be a string or null (type={type(value).__name__})")
            text = value.strip()
            if not text:
                return None
            return normalize_path(text, path=path)

        def optional_int(value: Any, *, path: str) -> int | None:
            """Parse an optional int from int/string inputs."""

            if value is None:
                return None
            if isinstance(value, bool):
                raise TypeError(f"{path} must be an int (type=bool)")
            if isinstance(value, int):
                return int(value)
            if isinstance(value, str):
                if not value.strip():
                    return None
                try:
                    return int(value.strip())
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"{path} must be an int (got {value!r})") from exc
            raise TypeError(f"{path} must be an int (type={type(value).__name__})")

        def parse_string_list(value: Any, *, path: str) -> tuple[str, ...]:
            """Parse a list[str] into a normalized tuple[str, ...]."""

            if value is None:
                return ()
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{path} must be a list[str] (type={type(value).__name__})")
            items: list[str] = []
            for idx, item in enumerate(value):
                if not isinstance(item, str):
                    raise TypeError(
                        f"{path}[{idx}] must be a string (type={type(item).__name__})"
                    )
                trimmed = item.strip()
                if not trimmed:
                    raise ValueError(f"{path}[{idx}] cannot be empty")
                items.append(trimmed)
            return tuple(items)

        def parse_stage_sequence(value: Any, *, path: str) -> tuple[Any, ...]:
            """Parse a stage sequence list into a normalized tuple of string/spec mappings."""

            if value is None:
                return ()
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"{path} must be a list[str|mapping] (type={type(value).__name__})"
                )

            entries: list[Any] = []
            for idx, item in enumerate(value):
                if isinstance(item, str):
                    trimmed = item.strip()
                    if not trimmed:
                        raise ValueError(f"{path}[{idx}] cannot be empty")
                    entries.append(trimmed)
                    continue

                if isinstance(item, Mapping):
                    allowed = {"stage", "name"}
                    for key in item.keys():
                        if key not in allowed:
                            raise ValueError(f"Unknown config key {path}[{idx}].{key}")
                    raw_stage = item.get("stage")
                    raw_name = item.get("name")
                    if not isinstance(raw_stage, str) or not raw_stage.strip():
                        raise ValueError(f"{path}[{idx}].stage missing/invalid")
                    if not isinstance(raw_name, str) or not raw_name.strip():
                        raise ValueError(f"{path}[{idx}].name missing/invalid")
                    entries.append({"stage": raw_stage.strip(), "name": raw_name.strip()})
                    continue

                raise TypeError(
                    f"{path}[{idx}] must be a string or mapping (type={type(item).__name__})"
                )

            return tuple(entries)

        def parse_stage_config_mapping(value: Any, *, path: str) -> dict[str, dict[str, Any]]:
            """Parse a stage config mapping into a dict[str, dict[str, Any]]."""

            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise TypeError(f"{path} must be a mapping (type={type(value).__name__})")
            out: dict[str, dict[str, Any]] = {}
            for raw_key, raw_cfg in value.items():
                if not isinstance(raw_key, str) or not raw_key.strip():
                    raise ValueError(f"{path} keys must be non-empty strings")
                key = raw_key.strip()
                if raw_cfg is None:
                    out[key] = {}
                    continue
                if not isinstance(raw_cfg, Mapping):
                    raise TypeError(
                        f"{path}.{key} must be a mapping (type={type(raw_cfg).__name__})"
                    )
                out[key] = dict(raw_cfg)
            return out

        # Unknown-key validation for prompt section (stage knobs remain opaque).
        allowed_prompt_keys = {
            "categories_path",
            "profile_path",
            "generations_path",
            "titles_manifest_path",
            "random_seed",
            "plan",
            "refine_only",
            "stages",
            "stage_configs",
            "output",
            "extensions",
        }
        unknown_prompt_keys = sorted(
            key for key in prompt_cfg.keys() if isinstance(key, str) and key not in allowed_prompt_keys
        )
        if unknown_prompt_keys:
            message = "Unknown prompt config keys: " + ", ".join(f"prompt.{k}" for k in unknown_prompt_keys)
            if strict_unknown_keys:
                raise ValueError(message)
            warnings.append(message)

        categories_path_raw = prompt_cfg.get("categories_path")
        categories_path = normalize_path(categories_path_raw, path="prompt.categories_path")

        profile_path_raw = prompt_cfg.get("profile_path")
        profile_path = normalize_path(profile_path_raw, path="prompt.profile_path")

        generations_csv_path_raw = prompt_cfg.get("generations_path")
        if run_mode == "full" and not (isinstance(generations_csv_path_raw, str) and generations_csv_path_raw.strip()):
            raise ValueError("Missing required config: prompt.generations_path")
        generations_csv_path = optional_path(generations_csv_path_raw, path="prompt.generations_path")

        titles_manifest_raw = prompt_cfg.get("titles_manifest_path")
        titles_manifest_path = optional_path(titles_manifest_raw, path="prompt.titles_manifest_path")
        if run_mode == "full" and not titles_manifest_path:
            if not generation_dir:
                raise ValueError(
                    "prompt.titles_manifest_path is not set and image.generation_path is unavailable"
                )
            titles_manifest_path = os.path.abspath(os.path.join(generation_dir, "titles_manifest.csv"))
            warnings.append(
                "Config key prompt.titles_manifest_path is not set; defaulting to "
                f"{titles_manifest_path}"
            )

        random_seed = optional_int(prompt_cfg.get("random_seed"), path="prompt.random_seed")

        raw_plan = prompt_cfg.get("plan")
        if raw_plan is None:
            raise ValueError("Missing required config: prompt.plan")
        if not isinstance(raw_plan, str):
            raise TypeError(f"prompt.plan must be a string (type={type(raw_plan).__name__})")
        plan = raw_plan.strip().lower()
        if not plan:
            raise ValueError("Invalid config value for prompt.plan: must be a non-empty string")

        refine_only_raw = prompt_cfg.get("refine_only")
        if refine_only_raw is None:
            refine_only_cfg: Mapping[str, Any] = {}
        elif not isinstance(refine_only_raw, Mapping):
            raise TypeError(
                f"prompt.refine_only must be a mapping (type={type(refine_only_raw).__name__})"
            )
        else:
            refine_only_cfg = refine_only_raw

        raw_draft = refine_only_cfg.get("draft")
        if raw_draft is None:
            refine_draft: str | None = None
        elif not isinstance(raw_draft, str):
            raise TypeError(
                f"prompt.refine_only.draft must be a string (type={type(raw_draft).__name__})"
            )
        else:
            refine_draft = raw_draft

        draft_path = optional_path(refine_only_cfg.get("draft_path"), path="prompt.refine_only.draft_path")
        if refine_draft and draft_path:
            raise ValueError(
                "Provide only one of prompt.refine_only.draft or prompt.refine_only.draft_path"
            )

        stages_raw = prompt_cfg.get("stages")
        if stages_raw is None:
            stages_cfg: Mapping[str, Any] = {}
        elif not isinstance(stages_raw, Mapping):
            raise TypeError(f"prompt.stages must be a mapping (type={type(stages_raw).__name__})")
        else:
            stages_cfg = stages_raw

        allowed_stage_keys = {"include", "exclude", "sequence", "overrides"}
        unknown_stage_keys = sorted(
            key for key in stages_cfg.keys() if isinstance(key, str) and key not in allowed_stage_keys
        )
        if unknown_stage_keys:
            raise ValueError(
                "Unknown prompt.stages keys: "
                + ", ".join(f"prompt.stages.{k}" for k in unknown_stage_keys)
            )

        include = parse_string_list(stages_cfg.get("include"), path="prompt.stages.include")
        exclude = parse_string_list(stages_cfg.get("exclude"), path="prompt.stages.exclude")
        sequence = parse_stage_sequence(stages_cfg.get("sequence"), path="prompt.stages.sequence")

        raw_overrides = stages_cfg.get("overrides")
        if raw_overrides is None:
            overrides_cfg: Mapping[str, Any] = {}
        elif not isinstance(raw_overrides, Mapping):
            raise TypeError(
                f"prompt.stages.overrides must be a mapping (type={type(raw_overrides).__name__})"
            )
        else:
            overrides_cfg = raw_overrides

        stage_overrides: dict[str, StageOverride] = {}
        for raw_stage_id, raw_override in overrides_cfg.items():
            if not isinstance(raw_stage_id, str) or not raw_stage_id.strip():
                raise ValueError("prompt.stages.overrides keys must be non-empty strings")
            stage_id = raw_stage_id.strip()
            if not isinstance(raw_override, Mapping):
                raise TypeError(
                    f"prompt.stages.overrides.{stage_id} must be a mapping (type={type(raw_override).__name__})"
                )

            allowed_keys = {"temperature", "params"}
            unknown_keys = sorted(
                key for key in raw_override.keys() if isinstance(key, str) and key not in allowed_keys
            )
            if unknown_keys:
                raise ValueError(
                    "Unknown prompt.stages.overrides keys: "
                    + ", ".join(f"prompt.stages.overrides.{stage_id}.{k}" for k in unknown_keys)
                )

            temperature_raw = raw_override.get("temperature")
            temperature: float | None = None
            if temperature_raw is not None:
                temperature = parse_float(temperature_raw, f"prompt.stages.overrides.{stage_id}.temperature")

            params_raw = raw_override.get("params")
            params: dict[str, Any] | None = None
            if params_raw is not None:
                if not isinstance(params_raw, Mapping):
                    raise TypeError(
                        f"prompt.stages.overrides.{stage_id}.params must be a mapping (type={type(params_raw).__name__})"
                    )
                if "temperature" in params_raw:
                    raise ValueError(
                        f"prompt.stages.overrides.{stage_id}.params sets temperature; use .temperature instead"
                    )
                params = dict(params_raw)

            stage_overrides[stage_id] = StageOverride(temperature=temperature, params=params)

        output_raw = prompt_cfg.get("output")
        if output_raw is None:
            output_cfg: Mapping[str, Any] = {}
        elif not isinstance(output_raw, Mapping):
            raise TypeError(f"prompt.output must be a mapping (type={type(output_raw).__name__})")
        else:
            output_cfg = output_raw

        allowed_output_keys = {"capture_stage"}
        unknown_output_keys = sorted(
            key for key in output_cfg.keys() if isinstance(key, str) and key not in allowed_output_keys
        )
        if unknown_output_keys:
            raise ValueError(
                "Unknown prompt.output keys: "
                + ", ".join(f"prompt.output.{k}" for k in unknown_output_keys)
            )

        capture_stage_raw = output_cfg.get("capture_stage") if output_cfg else None
        capture_stage: str | None
        if capture_stage_raw is None:
            capture_stage = None
        elif not isinstance(capture_stage_raw, str):
            raise TypeError(
                "prompt.output.capture_stage must be a string or null "
                f"(type={type(capture_stage_raw).__name__})"
            )
        else:
            capture_stage = capture_stage_raw.strip() or None
            if capture_stage_raw is not None and capture_stage is None:
                raise ValueError(
                    "prompt.output.capture_stage must be a non-empty string or null"
                )

        stage_configs_raw = prompt_cfg.get("stage_configs")
        if stage_configs_raw is None:
            stage_configs_cfg: Mapping[str, Any] = {}
        elif not isinstance(stage_configs_raw, Mapping):
            raise TypeError(
                f"prompt.stage_configs must be a mapping (type={type(stage_configs_raw).__name__})"
            )
        else:
            stage_configs_cfg = stage_configs_raw

        allowed_stage_configs_keys = {"defaults", "instances"}
        unknown_stage_configs_keys = sorted(
            key
            for key in stage_configs_cfg.keys()
            if isinstance(key, str) and key not in allowed_stage_configs_keys
        )
        if unknown_stage_configs_keys:
            raise ValueError(
                "Unknown prompt.stage_configs keys: "
                + ", ".join(f"prompt.stage_configs.{k}" for k in unknown_stage_configs_keys)
            )

        stage_configs_defaults = parse_stage_config_mapping(
            stage_configs_cfg.get("defaults"), path="prompt.stage_configs.defaults"
        )
        stage_configs_instances = parse_stage_config_mapping(
            stage_configs_cfg.get("instances"), path="prompt.stage_configs.instances"
        )

        return (
            PromptPipelineConfig(
                categories_path=categories_path,
                profile_path=profile_path,
                generations_csv_path=generations_csv_path,
                titles_manifest_path=titles_manifest_path,
                random_seed=random_seed,
                plan=plan,
                refine_only=RefineOnlyInputs(draft=refine_draft, draft_path=draft_path),
                stages=PipelineOverrides(
                    include=include,
                    exclude=exclude,
                    sequence=sequence,
                    overrides=stage_overrides,
                    capture_stage=capture_stage,
                ),
                stage_configs=StageConfigRouting(
                    defaults=stage_configs_defaults,
                    instances=stage_configs_instances,
                ),
            ),
            warnings,
        )
