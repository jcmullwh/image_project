import random

import pandas as pd

from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import PlanInputs
from image_project.impl.current.plans import PromptPlanManager
from pipelinekit.stage_types import StageInstance


def test_all_registered_plans_return_stage_instances_only(tmp_path):
    categories_path = tmp_path / "categories.csv"
    categories_path.write_text(
        "Subject Matter,Narrative,Mood,Composition,Perspective,Style,Time Period_Context,Color Scheme\n"
        "Cat,Quest,Moody,Wide,Top-down,Baroque,Renaissance,Vibrant\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.csv"
    profile_path.write_text("Likes,Dislikes\ncolorful,boring\n", encoding="utf-8")

    generation_dir = tmp_path / "generated"
    upscale_dir = tmp_path / "upscaled"
    log_dir = tmp_path / "logs"
    generations_csv = tmp_path / "generations.csv"

    cfg_dict = {
        "run": {"mode": "prompt_only"},
        "prompt": {
            "plan": "standard",
            "categories_path": str(categories_path),
            "profile_path": str(profile_path),
            "generations_path": str(generations_csv),
            "random_seed": 123,
            "concepts": {"filters": {"enabled": False}},
            "stages": {"sequence": ["standard.initial_prompt"]},
            "scoring": {
                "enabled": True,
                "num_ideas": 2,
                "exploration_rate": 0.0,
                "judge_temperature": 0.0,
                "generator_profile_abstraction": False,
                "novelty": {"enabled": False, "window": 0},
            },
            "blackbox_refine": {
                "enabled": True,
                "iterations": 1,
                "algorithm": "hillclimb",
                "branching_factor": 2,
                "include_parents_as_candidates": False,
                "generator_temperature": 0.9,
                "variation_prompt": {"template": "v1", "include_profile": False},
                "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
            },
        },
        "image": {
            "generation_path": str(generation_dir),
            "upscale_path": str(upscale_dir),
            "log_path": str(log_dir),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        preferences_guidance="",
        context_guidance=None,
        rng=random.Random(0),
        draft_prompt="seed",
    )

    for plan_name in PromptPlanManager.available():
        plan = PromptPlanManager.get(plan_name)
        nodes = plan.stage_nodes(inputs)
        assert isinstance(nodes, list), plan_name
        assert nodes, plan_name
        assert all(isinstance(node, StageInstance) for node in nodes), plan_name
