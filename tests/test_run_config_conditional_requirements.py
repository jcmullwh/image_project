import pytest

from run_config import RunConfig


def test_prompt_only_allows_omitting_media_paths(tmp_path):
    cfg_dict = {
        "run": {"mode": "prompt_only"},
        "prompt": {
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
        },
        "image": {"log_path": str(tmp_path / "logs")},
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    cfg, warnings = RunConfig.from_dict(cfg_dict)
    assert cfg.run_mode == "prompt_only"
    assert cfg.generation_dir is None
    assert cfg.upscale_dir is None
    assert cfg.generations_csv_path is None
    assert cfg.titles_manifest_path is None
    assert warnings == []


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

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

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

