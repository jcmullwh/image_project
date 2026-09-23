import random

import pandas as pd
import pytest

from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import PlanInputs
from image_project.framework.prompt_pipeline.pipeline_overrides import PipelineOverrides
from image_project.stages.blackbox.prepare import STAGE as BLACKBOX_PREPARE
from pipelinekit.config_namespace import ConfigNamespace


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
        user_profile=pd.DataFrame(),
        preferences_guidance="",
        context_guidance=None,
        rng=random.Random(0),
    )


def test_unknown_novelty_method_raises(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)
    cfg_ns = ConfigNamespace(
        {"novelty": {"enabled": True, "method": "nope"}},
        path="prompt.stage_configs.resolved.blackbox.prepare",
    )
    with pytest.raises(ValueError, match=r"blackbox\.prepare\.novelty\.method"):
        BLACKBOX_PREPARE.build(inputs, instance_id="blackbox.prepare", cfg=cfg_ns)


def test_invalid_novelty_df_min_raises(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)
    cfg_ns = ConfigNamespace(
        {"novelty": {"enabled": True, "method": "df_overlap_v1", "df_min": 0}},
        path="prompt.stage_configs.resolved.blackbox.prepare",
    )
    with pytest.raises(ValueError, match=r"blackbox\.prepare\.novelty\.df_min"):
        BLACKBOX_PREPARE.build(inputs, instance_id="blackbox.prepare", cfg=cfg_ns)
