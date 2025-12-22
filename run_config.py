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
    generator_profile_abstraction: bool
    novelty: PromptNoveltyConfig


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
    prompt_scoring: PromptScoringConfig

    context_enabled: bool
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
            generator_profile_abstraction=generator_profile_abstraction,
            novelty=PromptNoveltyConfig(enabled=novelty_enabled, window=novelty_window),
        )

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
                generation_dir=normalize_path(generation_dir_raw),
                upscale_dir=normalize_path(upscale_dir_raw),
                log_dir=normalize_path(log_dir_raw),
                categories_path=normalize_path(categories_path_raw),
                profile_path=normalize_path(profile_path_raw),
                generations_csv_path=normalize_path(generations_csv_raw),
                titles_manifest_path=normalize_path(titles_manifest_raw),
                caption_font_path=normalize_path(caption_font_raw) if caption_font_raw else None,
                random_seed=random_seed,
                prompt_scoring=prompt_scoring,
                context_enabled=context_enabled,
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
