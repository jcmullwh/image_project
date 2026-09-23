from __future__ import annotations

"""Framework-level configuration parsing (infra only).

This module intentionally parses only operational/infra configuration:
- run mode
- output/log directories
- context injection infra
- rclone + upscaling infra
- experiment metadata

Prompt pipeline selection, stage routing, and stage-owned knobs live outside this
module (see `image_project.framework.prompt_pipeline`).
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from image_project.foundation.config_io import find_repo_root
from image_project.framework.context import ContextManager

RunMode = Literal["full", "prompt_only"]
ContextInjectionLocation = Literal["system", "prompt", "both"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Optional experiment metadata recorded in transcripts/artifacts."""

    id: str | None = None
    variant: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunReviewConfig:
    """Configuration for optional run-review report generation."""

    enabled: bool = False
    review_path: str | None = None


def parse_bool(value: Any, path: str) -> bool:
    """Parse a boolean strictly to avoid `bool("false")` footguns.

    Accepts:
      - True/False
      - 0/1 (ints)
      - strings: true/false/1/0/yes/no (case-insensitive, surrounding whitespace ignored)

    Raises:
      ValueError: for anything else, with the provided config key path.
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
    """Parse a float from int/float/string inputs with clear errors."""

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
    """Parse an int from int/string inputs with clear errors."""

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
    """Parse an aspect ratio (width/height) from common formats.

    Accepts:
      - string: "1.5", "3/2", or "16:9"
      - 2-item list/tuple: [3, 2]
    """

    ratio: float
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Invalid aspect ratio for {path}: {value!r}")
        if "/" in text:
            raw_parts = text.split("/")
        elif ":" in text:
            raw_parts = text.split(":")
        else:
            raw_parts = [text]
        parts = [p.strip() for p in raw_parts if p.strip()]
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
class RunConfig:
    """Operational config used by the framework runtime."""

    generation_dir: str | None
    upscale_dir: str | None
    log_dir: str
    caption_font_path: str | None

    run_mode: RunMode
    experiment: ExperimentConfig
    run_review: RunReviewConfig

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
        """Parse and validate configuration, returning (RunConfig, warnings).

        Raises:
            ValueError: If required keys are missing or invalid.
        """

        if not isinstance(cfg, Mapping):
            raise ValueError("Config must be a mapping")

        warnings: list[str] = []
        repo_root: str | None = None

        strict_unknown_keys = False
        if "strict" in cfg:
            strict_unknown_keys = parse_bool(cfg.get("strict"), "strict")

        raw_image_cfg: Any = cfg.get("image")
        if isinstance(raw_image_cfg, Mapping) and "save_path" in raw_image_cfg:
            raise ValueError(
                "Config key image.save_path has been removed; use image.generation_path instead."
            )

        def collect_unknown_keys(
            mapping: Any,
            schema: Mapping[str, Any],
            *,
            prefix: str,
            exempt_keys: set[str] | None = None,
        ) -> list[str]:
            """Collect unknown keys under `mapping` given a nested `schema`."""
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

        run_schema: Mapping[str, Any] = {"mode": None}

        image_schema: Mapping[str, Any] = {
            "generation_path": None,
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

        run_review_schema: Mapping[str, Any] = {"enabled": None, "review_path": None}

        unknown_keys: list[str] = []
        unknown_keys.extend(collect_unknown_keys(cfg.get("run"), run_schema, prefix="run"))
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
            """Normalize a (possibly relative) path, anchored at the repo root."""
            expanded = os.path.expandvars(os.path.expanduser(value.strip()))
            if not os.path.isabs(expanded):
                nonlocal repo_root
                if repo_root is None:
                    repo_root = find_repo_root()
                expanded = os.path.join(repo_root, expanded)
            return os.path.abspath(expanded)

        def normalize_optional_path(value: str | None) -> str | None:
            """Normalize an optional path, returning None when empty/unset."""
            if value is None:
                return None
            normalized = value.strip()
            if not normalized:
                return None
            return normalize_path(normalized)

        def get_mapping(path: str) -> Mapping[str, Any]:
            """Best-effort lookup of a nested mapping at `path` (dot-separated)."""
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping):
                    return {}
                cur = cur.get(part)
            return cur if isinstance(cur, Mapping) else {}

        def require_str(path: str, *, reason: str | None = None) -> str:
            """Get a required non-empty string at `path` (dot-separated)."""
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
            """Get an optional string at `path`, returning None when missing/empty."""
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
            """Get an optional int at `path`, accepting int or int-like strings."""
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
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"Invalid config value for {path}: must be an int") from exc
            raise ValueError(f"Invalid config type for {path}: expected int")

        def optional_bool(path: str, *, default: bool) -> bool:
            """Get a boolean at `path` or return the provided default when missing."""
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
        if run_mode == "full" and not generation_dir_raw:
            raise ValueError("Missing required config: image.generation_path")

        log_dir_raw = require_str("image.log_path")
        if run_mode == "full" and upscale_enabled:
            upscale_dir_raw = require_str(
                "image.upscale_path", reason="required when upscale.enabled=true"
            )
        else:
            upscale_dir_raw = optional_str("image.upscale_path")

        caption_font_raw = optional_str("image.caption_font_path")

        raw_experiment_cfg: Any = cfg.get("experiment")
        experiment_cfg: Mapping[str, Any]
        if raw_experiment_cfg is None:
            experiment_cfg = {}
        elif not isinstance(raw_experiment_cfg, Mapping):
            raise ValueError("Invalid config type for experiment: expected mapping")
        else:
            experiment_cfg = raw_experiment_cfg

        def normalize_experiment_text(value: Any, path: str) -> str | None:
            """Normalize optional experiment metadata values into `str | None`."""
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

        context_cfg_raw = get_mapping("context")
        context_enabled = optional_bool("context.enabled", default=False)
        raw_injection_location: Any = context_cfg_raw.get("injection_location", "system")
        if raw_injection_location is None:
            raise ValueError("Invalid config value for context.injection_location: None")
        if not isinstance(raw_injection_location, str):
            raise ValueError("Invalid config type for context.injection_location: expected string")
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
                raise ValueError("Missing required config: rclone.remote (required when rclone.enabled=true)")
            if not rclone_album:
                raise ValueError("Missing required config: rclone.album (required when rclone.enabled=true)")

        upscale_cfg = get_mapping("upscale")
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
            upscale_target_aspect_ratio = parse_aspect_ratio(raw_aspect_ratio, "upscale.target_aspect_ratio")

        if upscale_target_aspect_ratio is not None and upscale_target_width_px is not None:
            raise ValueError(
                "Provide either upscale.target_width_px/target_height_px or upscale.target_aspect_ratio, not both"
            )

        upscale_engine = (
            str(upscale_cfg.get("engine", "realesrgan-ncnn-vulkan")) if upscale_cfg else "realesrgan-ncnn-vulkan"
        )
        upscale_realesrgan_binary = optional_str("upscale.realesrgan_binary")
        upscale_model_name = (
            str(upscale_cfg.get("model_name", "realesrgan-x4plus")) if upscale_cfg else "realesrgan-x4plus"
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
                caption_font_path=normalize_path(caption_font_raw) if caption_font_raw else None,
                run_mode=run_mode,  # type: ignore[arg-type]
                experiment=experiment,
                run_review=run_review,
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
                upscale_realesrgan_binary=normalize_path(upscale_realesrgan_binary)
                if upscale_realesrgan_binary
                else None,
                upscale_model_name=upscale_model_name,
                upscale_model_path=normalize_path(upscale_model_path) if upscale_model_path else None,
                upscale_tile_size=upscale_tile_size,
                upscale_tta=upscale_tta,
                upscale_allow_fallback_resize=upscale_allow_fallback_resize,
            ),
            warnings,
        )
