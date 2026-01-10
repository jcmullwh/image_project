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


def test_unknown_novelty_method_raises(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["scoring"] = {"enabled": True, "novelty": {"enabled": True, "method": "nope"}}
    with pytest.raises(ValueError, match=r"prompt\.scoring\.novelty\.method"):
        RunConfig.from_dict(cfg_dict)


def test_invalid_novelty_df_min_raises(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["scoring"] = {
        "enabled": True,
        "novelty": {"enabled": True, "method": "df_overlap_v1", "df_min": 0},
    }
    with pytest.raises(ValueError, match=r"prompt\.scoring\.novelty\.df_min"):
        RunConfig.from_dict(cfg_dict)

