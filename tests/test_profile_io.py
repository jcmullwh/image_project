import random

import pandas as pd

from image_project.framework.config import RunConfig
from image_project.framework.inputs import extract_dislikes
from image_project.framework.profile_io import load_user_profile
from image_project.framework.prompt_pipeline import PlanInputs
from image_project.prompts.preprompt import build_preferences_guidance
from image_project.impl.current.plans import PromptPlanManager
from pipelinekit.stage_types import StageInstance


def test_load_user_profile_row_based_includes_notes(tmp_path):
    path = tmp_path / "user_profile_v5_like_dislike.csv"
    pd.DataFrame(
        {
            "category": ["like", "dislike"],
            "item": ["Bright palettes", "Doomsday themes"],
            "notes": ["vivid and cohesive", "avoid before bed"],
        }
    ).to_csv(path, index=False)

    df = load_user_profile(str(path))
    assert list(df.columns) == ["Likes", "Dislikes"]
    assert df["Likes"].dropna().tolist() == ["Bright palettes — vivid and cohesive"]
    assert df["Dislikes"].dropna().tolist() == ["Doomsday themes — avoid before bed"]


def test_load_user_profile_row_based_love_like_dislike_hate(tmp_path):
    path = tmp_path / "user_profile_v5_love_like_dislike_hate.csv"
    pd.DataFrame(
        {
            "category": ["love", "like", "dislike", "hate"],
            "item": ["Symmetry", "Gardens", "Boring frames", "Horror"],
            "notes": ["strong preference", "", "", "no jump scares"],
        }
    ).to_csv(path, index=False)

    df = load_user_profile(str(path))
    assert list(df.columns) == ["Loves", "Likes", "Dislikes", "Hates"]
    assert df["Loves"].dropna().tolist() == ["Symmetry — strong preference"]
    assert df["Likes"].dropna().tolist() == ["Gardens"]
    assert df["Dislikes"].dropna().tolist() == ["Boring frames"]
    assert df["Hates"].dropna().tolist() == ["Horror — no jump scares"]


def test_load_user_profile_row_based_strength_conditionals(tmp_path):
    path = tmp_path / "user_profile_v5_strength_conditionals.csv"
    pd.DataFrame(
        {
            "strength": ["loves", "likes", "dislikes", "hates"],
            "item": ["Symmetry", "Gardens", "Boring frames", "Horror"],
            "conditionals": ["strong preference", "", "", "no jump scares"],
        }
    ).to_csv(path, index=False)

    df = load_user_profile(str(path))
    assert list(df.columns) == ["Loves", "Likes", "Dislikes", "Hates"]
    assert df["Loves"].dropna().tolist() == ["Symmetry — strong preference"]
    assert df["Likes"].dropna().tolist() == ["Gardens"]
    assert df["Dislikes"].dropna().tolist() == ["Boring frames"]
    assert df["Hates"].dropna().tolist() == ["Horror — no jump scares"]


def test_extract_dislikes_includes_hates():
    profile = pd.DataFrame({"Dislikes": ["x", None], "Hates": ["y", "x"]})
    assert extract_dislikes(profile) == ["x", "y"]


def test_build_preferences_guidance_distinguishes_love_like_dislike_hate():
    profile = pd.DataFrame(
        {
            "Loves": ["A"],
            "Likes": ["B"],
            "Dislikes": ["C"],
            "Hates": ["D"],
        }
    )

    guidance = build_preferences_guidance(profile)
    assert "Preference strength legend:" in guidance
    assert "- Loves:" in guidance
    assert "- Likes:" in guidance
    assert "- Dislikes:" in guidance
    assert "- Hates:" in guidance


def test_blackbox_plan_uses_profile_hints_load_when_configured(tmp_path):
    categories_path = tmp_path / "categories.csv"
    profile_path = tmp_path / "profile.csv"
    hints_path = tmp_path / "profile_hints.txt"

    pd.DataFrame({"Subject Matter": ["Cat"], "Narrative": ["Quest"]}).to_csv(
        categories_path, index=False
    )
    pd.DataFrame({"Likes": ["colorful"], "Dislikes": ["boring"]}).to_csv(profile_path, index=False)
    hints_path.write_text("generator-safe hints", encoding="utf-8")

    cfg_dict = {
        "run": {"mode": "prompt_only"},
        "prompt": {
            "plan": "blackbox",
            "categories_path": str(categories_path),
            "profile_path": str(profile_path),
            "scoring": {
                "enabled": True,
                "generator_profile_abstraction": True,
                "generator_profile_hints_path": str(hints_path),
                "novelty": {"enabled": False, "window": 0},
            },
        },
        "image": {"log_path": str(tmp_path / "logs")},
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    resolved = PromptPlanManager.resolve(cfg)

    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame({"Likes": ["x"], "Dislikes": ["y"]}),
        preferences_guidance="Likes:\n- x\n\nDislikes:\n- y",
        context_guidance=None,
        rng=random.Random(0),
    )

    stage_ids = [
        node.instance_id if isinstance(node, StageInstance) else str(node.name)
        for node in resolved.plan.stage_nodes(inputs)
    ]
    assert "blackbox.profile_hints_load" in stage_ids
    assert "blackbox.profile_abstraction" not in stage_ids


def test_blackbox_refine_plan_adds_final_refinement_stage(tmp_path):
    categories_path = tmp_path / "categories.csv"
    profile_path = tmp_path / "profile.csv"

    pd.DataFrame({"Subject Matter": ["Cat"], "Narrative": ["Quest"]}).to_csv(
        categories_path, index=False
    )
    pd.DataFrame({"Likes": ["colorful"], "Dislikes": ["boring"]}).to_csv(profile_path, index=False)

    cfg_dict = {
        "run": {"mode": "prompt_only"},
        "prompt": {
            "plan": "blackbox_refine",
            "categories_path": str(categories_path),
            "profile_path": str(profile_path),
            "scoring": {"enabled": True, "novelty": {"enabled": False, "window": 0}},
        },
        "image": {"log_path": str(tmp_path / "logs")},
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    resolved = PromptPlanManager.resolve(cfg)

    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame({"Likes": ["x"], "Dislikes": ["y"]}),
        preferences_guidance="Likes:\n- x\n\nDislikes:\n- y",
        context_guidance=None,
        rng=random.Random(0),
    )

    stage_ids = [
        node.instance_id if isinstance(node, StageInstance) else str(node.name)
        for node in resolved.plan.stage_nodes(inputs)
    ]
    assert "blackbox_refine.init_state" in stage_ids
    assert stage_ids[-1] == "postprompt.openai_format"
