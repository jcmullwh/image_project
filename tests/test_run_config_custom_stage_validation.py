import random

import pandas as pd
import pytest

from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import PlanInputs
from image_project.impl.current.plans import PromptPlanManager


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


def test_custom_plan_unknown_stage_ids_fail_fast_with_suggestions(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "custom"
    cfg_dict["prompt"]["stages"] = {"sequence": ["ab.scene_draf"]}

    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        preferences_guidance="",
        context_guidance=None,
        rng=random.Random(0),
    )

    with pytest.raises(ValueError) as excinfo:
        PromptPlanManager.get("custom").stage_nodes(inputs)

    msg = str(excinfo.value)
    assert "prompt.stages.sequence[0]" in msg
    assert "ab.scene_draf" in msg
    assert "did you mean" in msg
    assert "ab.scene_draft" in msg
