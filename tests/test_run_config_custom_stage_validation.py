import pytest

from image_project.framework.config import RunConfig


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

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    msg = str(excinfo.value)
    assert "prompt.stages.sequence[0]" in msg
    assert "ab.scene_draf" in msg
    assert "did you mean" in msg
    assert "ab.scene_draft" in msg

