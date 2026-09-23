import pytest

from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline.pipeline_overrides import PromptPipelineConfig


def _base_cfg_dict(tmp_path) -> dict:
    return {
        "prompt": {
            "plan": "standard",
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

    cfg, run_warnings = RunConfig.from_dict(cfg_dict)
    _prompt_cfg, prompt_warnings = PromptPipelineConfig.from_root_dict(
        cfg_dict, run_mode=cfg.run_mode, generation_dir=cfg.generation_dir
    )

    warnings = [*run_warnings, *prompt_warnings]
    assert any("Unknown prompt config keys: prompt.unknown_prompt_key" in w for w in warnings)
    assert any("Unknown config key: image.unknown_image_key" in w for w in warnings)
    assert any("Unknown config key: context.unknown_context_key" in w for w in warnings)
    assert not any("prompt.extensions.plugin_x" in w for w in warnings)
    assert not any(w.startswith("Unknown config key: context.season") for w in warnings)


def test_unknown_config_keys_strict_mode_raises(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["strict"] = True
    cfg_dict["prompt"]["unknown_prompt_key"] = 123

    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    with pytest.raises(ValueError, match=r"Unknown prompt config keys: prompt\.unknown_prompt_key"):
        PromptPipelineConfig.from_root_dict(cfg_dict, run_mode=cfg.run_mode, generation_dir=cfg.generation_dir)


def test_stage_configs_allowed_in_strict_unknown_key_mode(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["strict"] = True
    cfg_dict["prompt"]["stage_configs"] = {
        "instances": {"standard.initial_prompt": {"typo_key": True}}
    }

    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    prompt_cfg, _prompt_warnings = PromptPipelineConfig.from_root_dict(
        cfg_dict, run_mode=cfg.run_mode, generation_dir=cfg.generation_dir
    )
    assert prompt_cfg.stage_configs_instances["standard.initial_prompt"]["typo_key"] is True


@pytest.mark.parametrize("strict", [False, True])
def test_image_save_path_is_rejected(tmp_path, strict):
    cfg_dict = _base_cfg_dict(tmp_path)
    if strict:
        cfg_dict["strict"] = True
    cfg_dict["image"]["save_path"] = str(tmp_path / "legacy")

    with pytest.raises(ValueError, match=r"Config key image\.save_path has been removed"):
        RunConfig.from_dict(cfg_dict)
