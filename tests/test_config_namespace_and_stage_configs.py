import random

import pandas as pd
import pytest

from image_project.framework.config import RunConfig
from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, compile_stage_nodes
from image_project.framework.prompt_pipeline.pipeline_overrides import PipelineOverrides
from image_project.impl.current.plans import PromptPlanManager
from image_project.stages.registry import get_stage_registry


def _base_cfg_dict(tmp_path) -> dict:
    return {
        "prompt": {
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
            "generations_path": str(tmp_path / "generations.csv"),
        },
        "image": {
            "generation_path": str(tmp_path / "generated"),
            "upscale_path": str(tmp_path / "upscaled"),
            "log_path": str(tmp_path / "logs"),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }


def _make_inputs(cfg: RunConfig) -> PlanInputs:
    return PlanInputs(
        cfg=cfg,
        pipeline=PipelineOverrides(include=(), exclude=(), sequence=(), overrides={}, capture_stage=None),
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame({"Likes": ["x"], "Dislikes": [None]}),
        preferences_guidance="Likes:\n- x",
        context_guidance=None,
        rng=random.Random(0),
    )


def test_config_namespace_get_bool_is_strict():
    ns = ConfigNamespace({"enabled": "false"}, path="prompt.stage_configs.test")
    with pytest.raises(TypeError, match=r"must be a boolean"):
        ns.get_bool("enabled")


def test_config_namespace_get_int_is_strict_and_validates_constraints():
    ns = ConfigNamespace({"count": 2.0}, path="prompt.stage_configs.test")
    with pytest.raises(TypeError, match=r"must be an int"):
        ns.get_int("count")

    ns2 = ConfigNamespace({"count": 2}, path="prompt.stage_configs.test")
    with pytest.raises(ValueError, match=r"must be >= 3"):
        ns2.get_int("count", min_value=3)
    with pytest.raises(ValueError, match=r"must be <= 1"):
        ns2.get_int("count", max_value=1)
    with pytest.raises(ValueError, match=r"must be one of: 1, 3"):
        ns2.get_int("count", choices=(1, 3))


def test_config_namespace_unknown_key_enforcement_includes_path_and_consumed_keys():
    ns = ConfigNamespace({"known": True, "typo": 1}, path="prompt.stage_configs.test")
    assert ns.get_bool("known") is True
    with pytest.raises(ValueError, match=r"Unknown config keys under prompt\.stage_configs\.test: typo"):
        ns.assert_consumed()


def test_stage_config_defaults_unknown_kind_fails_fast(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)
    stage_nodes = PromptPlanManager.get("standard").stage_nodes(inputs)

    with pytest.raises(ValueError, match=r"Unknown stage kind id in prompt\.stage_configs\.defaults"):
        compile_stage_nodes(
            stage_nodes,
            plan_name="standard",
            include=(),
            exclude=(),
            overrides={},
            stage_configs_defaults={"does_not_exist": {"x": 1}},
            stage_configs_instances={},
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )


def test_stage_config_instances_unknown_stage_fails_fast(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)
    stage_nodes = PromptPlanManager.get("standard").stage_nodes(inputs)

    with pytest.raises(ValueError, match=r"prompt\.stage_configs\.instances"):
        compile_stage_nodes(
            stage_nodes,
            plan_name="standard",
            include=("select_concepts", "initial_prompt"),
            exclude=(),
            overrides={},
            stage_configs_defaults={},
            stage_configs_instances={"does_not_exist": {"x": 1}},
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )


def test_stage_config_unknown_keys_fail_fast(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)
    stage_nodes = PromptPlanManager.get("standard").stage_nodes(inputs)

    with pytest.raises(ValueError, match=r"Unknown config keys under prompt\.stage_configs\.resolved\.standard\.initial_prompt: typo_key"):
        compile_stage_nodes(
            stage_nodes,
            plan_name="standard",
            include=("select_concepts", "initial_prompt"),
            exclude=(),
            overrides={},
            stage_configs_defaults={},
            stage_configs_instances={"standard.initial_prompt": {"typo_key": True}},
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )


def test_stage_io_missing_required_outputs_fails_before_execution(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)
    stage_nodes = PromptPlanManager.get("standard").stage_nodes(inputs)

    with pytest.raises(ValueError, match=r"Stage IO validation failed: stage=standard\.initial_prompt"):
        compile_stage_nodes(
            stage_nodes,
            plan_name="standard",
            include=("initial_prompt",),
            exclude=(),
            overrides={},
            stage_configs_defaults={},
            stage_configs_instances={},
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )
