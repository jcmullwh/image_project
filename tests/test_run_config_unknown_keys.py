import pytest

from run_config import RunConfig


def _base_cfg_dict(tmp_path) -> dict:
    return {
        "prompt": {
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
            "generations_path": str(tmp_path / "generations.csv"),
            "random_seed": 123,
        },
        "image": {
            "generation_path": str(tmp_path / "generated"),
            "upscale_path": str(tmp_path / "upscaled"),
            "log_path": str(tmp_path / "logs"),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }


def test_unknown_config_keys_warn_by_default(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["unknown_prompt_key"] = 123
    cfg_dict["prompt"]["extensions"] = {"plugin_x": {"enabled": True}}
    cfg_dict["image"]["unknown_image_key"] = True
    cfg_dict["context"] = {"enabled": False, "unknown_context_key": "x", "season": {"foo": 1}}

    _cfg, warnings = RunConfig.from_dict(cfg_dict)
    assert any("Unknown config key: prompt.unknown_prompt_key" in w for w in warnings)
    assert any("Unknown config key: image.unknown_image_key" in w for w in warnings)
    assert any("Unknown config key: context.unknown_context_key" in w for w in warnings)
    assert not any("prompt.extensions.plugin_x" in w for w in warnings)
    assert not any(w.startswith("Unknown config key: context.season") for w in warnings)


def test_unknown_config_keys_strict_mode_raises(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["strict"] = True
    cfg_dict["prompt"]["unknown_prompt_key"] = 123

    with pytest.raises(ValueError, match=r"Unknown config keys: prompt\.unknown_prompt_key"):
        RunConfig.from_dict(cfg_dict)

