from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from context_injectors import ContextManager


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


@dataclass(frozen=True)
class RunConfig:
    generation_dir: str
    upscale_dir: str
    log_dir: str

    categories_path: str
    profile_path: str
    generations_csv_path: str

    titles_manifest_path: str
    caption_font_path: str | None

    random_seed: int | None

    context_enabled: bool
    context_injectors: tuple[str, ...]
    context_cfg: Mapping[str, Any]

    rclone_enabled: bool
    rclone_remote: str | None
    rclone_album: str | None

    upscale_enabled: bool
    upscale_target_long_edge_px: int
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
        project_root = os.path.dirname(os.path.abspath(__file__))

        def normalize_path(value: str) -> str:
            expanded = os.path.expandvars(os.path.expanduser(value.strip()))
            if not os.path.isabs(expanded):
                expanded = os.path.join(project_root, expanded)
            return os.path.abspath(expanded)

        def get_mapping(path: str) -> Mapping[str, Any]:
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping):
                    return {}
                cur = cur.get(part)
            return cur if isinstance(cur, Mapping) else {}

        def require_str(path: str) -> str:
            cur: Any = cfg
            for part in path.split("."):
                if not isinstance(cur, Mapping) or part not in cur:
                    raise ValueError(f"Missing required config: {path}")
                cur = cur[part]
            if cur is None:
                raise ValueError(f"Missing required config: {path}")
            if not isinstance(cur, str):
                raise ValueError(f"Invalid config type for {path}: expected string")
            if not cur.strip():
                raise ValueError(f"Missing required config: {path}")
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

        image_cfg = get_mapping("image")
        if not image_cfg:
            raise ValueError("Missing required config: image")

        generation_dir_raw = optional_str("image.generation_path")
        save_dir_raw = optional_str("image.save_path")
        if generation_dir_raw and save_dir_raw:
            warnings.append(
                "Both image.generation_path and image.save_path are set; using image.generation_path."
            )
        if not generation_dir_raw:
            if save_dir_raw:
                warnings.append(
                    "Config key image.save_path is deprecated; use image.generation_path instead."
                )
                generation_dir_raw = save_dir_raw
            else:
                raise ValueError("Missing required config: image.generation_path")

        upscale_dir_raw = require_str("image.upscale_path")
        log_dir_raw = require_str("image.log_path")

        prompt_cfg = get_mapping("prompt")
        if not prompt_cfg:
            raise ValueError("Missing required config: prompt")

        categories_path_raw = require_str("prompt.categories_path")
        profile_path_raw = require_str("prompt.profile_path")
        generations_csv_raw = require_str("prompt.generations_path")

        titles_manifest_raw = optional_str("prompt.titles_manifest_path")
        if not titles_manifest_raw:
            titles_manifest_raw = os.path.join(generation_dir_raw, "titles_manifest.csv")
            warnings.append(
                "Config key prompt.titles_manifest_path is not set; defaulting to "
                f"{titles_manifest_raw}"
            )

        caption_font_raw = optional_str("image.caption_font_path")

        random_seed = optional_int("prompt.random_seed")

        context_cfg_raw = get_mapping("context")
        context_enabled = optional_bool("context.enabled", default=False)
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

        rclone_enabled = optional_bool("rclone.enabled", default=False)
        rclone_remote = optional_str("rclone.remote")
        rclone_album = optional_str("rclone.album")
        if rclone_enabled and (not rclone_remote or not rclone_album):
            raise ValueError("rclone.enabled=true but rclone.remote/album missing")

        upscale_cfg = get_mapping("upscale")
        upscale_enabled = optional_bool("upscale.enabled", default=False)
        upscale_target_long_edge_px = (
            int(upscale_cfg.get("target_long_edge_px", 3840)) if upscale_cfg else 3840
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
                generation_dir=normalize_path(generation_dir_raw),
                upscale_dir=normalize_path(upscale_dir_raw),
                log_dir=normalize_path(log_dir_raw),
                categories_path=normalize_path(categories_path_raw),
                profile_path=normalize_path(profile_path_raw),
                generations_csv_path=normalize_path(generations_csv_raw),
                titles_manifest_path=normalize_path(titles_manifest_raw),
                caption_font_path=normalize_path(caption_font_raw) if caption_font_raw else None,
                random_seed=random_seed,
                context_enabled=context_enabled,
                context_injectors=context_injectors,
                context_cfg=context_cfg,
                rclone_enabled=rclone_enabled,
                rclone_remote=rclone_remote,
                rclone_album=rclone_album,
                upscale_enabled=upscale_enabled,
                upscale_target_long_edge_px=upscale_target_long_edge_px,
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
