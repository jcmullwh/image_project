import pytest

from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline.pipeline_overrides import PromptPipelineConfig


def test_prompt_only_allows_omitting_media_paths(tmp_path):
    cfg_dict = {
        "run": {"mode": "prompt_only"},
        "prompt": {
            "plan": "simple",
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
        },
        "image": {"log_path": str(tmp_path / "logs")},
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    cfg, warnings = RunConfig.from_dict(cfg_dict)
    prompt_cfg, prompt_warnings = PromptPipelineConfig.from_root_dict(
        cfg_dict, run_mode=cfg.run_mode, generation_dir=cfg.generation_dir
    )
    assert cfg.run_mode == "prompt_only"
    assert cfg.generation_dir is None
    assert cfg.upscale_dir is None
    assert prompt_cfg.generations_csv_path is None
    assert prompt_cfg.titles_manifest_path is None
    assert warnings == []
    assert prompt_warnings == []


def test_full_mode_requires_image_generation_path(tmp_path):
    cfg_dict = {
        "run": {"mode": "full"},
        "prompt": {
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
            "generations_path": str(tmp_path / "generations.csv"),
        },
        "image": {"log_path": str(tmp_path / "logs")},
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    assert "image.generation_path" in str(excinfo.value)


def test_full_mode_requires_prompt_generations_path(tmp_path):
    cfg_dict = {
        "run": {"mode": "full"},
        "prompt": {
            "plan": "standard",
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
        },
        "image": {
            "generation_path": str(tmp_path / "generated"),
            "log_path": str(tmp_path / "logs"),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    with pytest.raises(ValueError) as excinfo:
        PromptPipelineConfig.from_root_dict(cfg_dict, run_mode=cfg.run_mode, generation_dir=cfg.generation_dir)

    assert "prompt.generations_path" in str(excinfo.value)


def test_upscale_enabled_requires_image_upscale_path(tmp_path):
    cfg_dict = {
        "run": {"mode": "full"},
        "prompt": {
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
            "generations_path": str(tmp_path / "generations.csv"),
        },
        "image": {
            "generation_path": str(tmp_path / "generated"),
            "log_path": str(tmp_path / "logs"),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": True},
    }

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    msg = str(excinfo.value)
    assert "image.upscale_path" in msg
    assert "upscale.enabled" in msg
