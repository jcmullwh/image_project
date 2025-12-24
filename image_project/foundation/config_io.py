from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml


def find_repo_root(start: str | os.PathLike[str] | None = None) -> str:
    start_path = Path(start or os.getcwd()).resolve()
    if start_path.is_file():
        start_path = start_path.parent

    markers = ("pyproject.toml", ".git")
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return str(candidate)
        if (candidate / ".git").exists():
            return str(candidate)

    raise FileNotFoundError(
        "Cannot locate repo root: searched from "
        f"{start_path} for {', '.join(markers)}"
    )


def _load_yaml_mapping(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except FileNotFoundError:
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return dict(payload)


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


def load_config(**kwargs):
    """
    Load configuration settings from a YAML file.

    Backwards-compatible signature used by existing tests and scripts.
    """

    config_path_override = kwargs.get("config_path")
    env_var = kwargs.get("env_var", "IMAGE_PROJECT_CONFIG")
    config_file = kwargs.get("config_name", "config")
    config_filetype = kwargs.get("config_type", ".yaml")
    config_relative_path = kwargs.get("config_rel_path", "config")

    # Env override (or explicit config_path) loads a single file (no local overlay).
    explicit_path = None
    if config_path_override is not None:
        explicit_path = str(config_path_override).strip() or None
    elif env_var:
        raw_env = os.environ.get(str(env_var), "")
        explicit_path = raw_env.strip() or None

    if explicit_path:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(explicit_path)))
        cfg = _load_yaml_mapping(expanded)
        meta = {
            "mode": "env" if config_path_override is None else "explicit",
            "paths": [expanded],
            "env_var": env_var,
            "repo_root": None,
        }
        return cfg, meta

    if os.path.isabs(str(config_relative_path)):
        config_directory = str(config_relative_path)
        repo_root = None
    else:
        repo_root = find_repo_root(kwargs.get("start_dir"))
        config_directory = os.path.join(repo_root, str(config_relative_path))
    base_config_path = os.path.join(config_directory, config_file + config_filetype)
    local_overlay_path = os.path.join(config_directory, "config.local.yaml")

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Missing base config file: {base_config_path}")

    cfg = _load_yaml_mapping(base_config_path)
    loaded_paths = [os.path.abspath(base_config_path)]
    mode = "base"

    if os.path.exists(local_overlay_path):
        overlay = _load_yaml_mapping(local_overlay_path)
        cfg = _deep_merge(cfg, overlay, path="")
        loaded_paths.append(os.path.abspath(local_overlay_path))
        mode = "base+local"

    meta = {"mode": mode, "paths": loaded_paths, "env_var": env_var, "repo_root": repo_root}
    return cfg, meta
