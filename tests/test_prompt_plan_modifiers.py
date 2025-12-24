import json
import logging
import random

import pandas as pd
import pytest

import main
from message_handling import MessageHandler
from pipeline import ChatRunner, RunContext
from prompt_plans import PlanInputs, PromptPlanManager, resolve_stage_specs
from run_config import RunConfig


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


def _make_ctx(cfg: RunConfig, *, generation_id: str = "gen_test") -> RunContext:
    logger = logging.getLogger(f"test.prompt_plans.{generation_id}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return RunContext(
        generation_id=generation_id,
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
        selected_concepts=["A", "B", "C"],
    )


def _make_standard_stage_specs(cfg: RunConfig) -> tuple[list, PlanInputs]:
    prompt_data = pd.DataFrame()
    user_profile = pd.DataFrame({"Likes": ["x"], "Dislikes": [None]})
    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=prompt_data,
        user_profile=user_profile,
        preferences_guidance="Likes:\n- x",
        context_guidance=None,
        rng=random.Random(0),
    )
    plan = PromptPlanManager.get("standard")
    return plan.stage_specs(inputs), inputs


def test_unknown_plan_fails_fast(tmp_path, monkeypatch):
    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            pass

        def text_chat(self, messages, **kwargs):
            return "resp"

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "does_not_exist"

    # Avoid importing a real backend in tests.
    monkeypatch.setattr(main, "TextAI", FakeTextAI)

    # Minimal CSV inputs (dislikes are empty so no concept rewrite call is attempted).
    pd.DataFrame(
        {
            "Subject Matter": ["Cat"],
            "Narrative": ["Quest"],
            "Mood": ["Moody"],
            "Composition": ["Wide"],
            "Perspective": ["Top-down"],
            "Style": ["Baroque"],
            "Time Period_Context": ["Renaissance"],
            "Color Scheme": ["Vibrant"],
        }
    ).to_csv(cfg_dict["prompt"]["categories_path"], index=False)
    pd.DataFrame({"Likes": ["x"], "Dislikes": [None]}).to_csv(cfg_dict["prompt"]["profile_path"], index=False)

    with pytest.raises(ValueError, match="Unknown prompt plan: does_not_exist"):
        main.run_generation(cfg_dict, generation_id="unit_test_unknown_plan")


def test_unknown_stage_id_in_include_fails_fast(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    stage_specs, _inputs = _make_standard_stage_specs(cfg)

    with pytest.raises(ValueError, match="prompt\\.stages\\.include"):
        resolve_stage_specs(
            stage_specs,
            plan_name="standard",
            include=("nope",),
            exclude=(),
            overrides={},
            capture_stage=None,
        )


def test_capture_stage_not_in_resolved_stages_fails_fast(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    stage_specs, _inputs = _make_standard_stage_specs(cfg)

    with pytest.raises(ValueError, match="prompt\\.output\\.capture_stage"):
        resolve_stage_specs(
            stage_specs,
            plan_name="standard",
            include=("initial_prompt",),
            exclude=(),
            overrides={},
            capture_stage="image_prompt_creation",
        )


def test_refinement_policy_none_produces_no_tot_enclave_steps(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["refinement"] = {"policy": "none"}
    cfg_dict["prompt"]["stages"] = {"include": ["initial_prompt"]}
    cfg_dict["prompt"]["output"] = {"capture_stage": "initial_prompt"}
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    plan = PromptPlanManager.get("standard")
    stage_specs, inputs = _make_standard_stage_specs(cfg)

    resolved = resolve_stage_specs(
        stage_specs,
        plan_name=plan.name,
        include=cfg.prompt_stages_include,
        exclude=cfg.prompt_stages_exclude,
        overrides=cfg.prompt_stages_overrides,
        capture_stage=cfg.prompt_output_capture_stage,
    )

    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp_none"

    ctx = _make_ctx(cfg, generation_id="none")
    runner = ChatRunner(ai_text=FakeTextAI())
    plan.execute(ctx, runner, resolved, inputs)

    assert all("tot_enclave" not in (step.get("path") or "") for step in ctx.steps)


def test_refinement_policy_tot_produces_tot_enclave_steps(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["refinement"] = {"policy": "tot"}
    cfg_dict["prompt"]["stages"] = {"include": ["initial_prompt"]}
    cfg_dict["prompt"]["output"] = {"capture_stage": "initial_prompt"}
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    plan = PromptPlanManager.get("standard")
    stage_specs, inputs = _make_standard_stage_specs(cfg)

    resolved = resolve_stage_specs(
        stage_specs,
        plan_name=plan.name,
        include=cfg.prompt_stages_include,
        exclude=cfg.prompt_stages_exclude,
        overrides=cfg.prompt_stages_overrides,
        capture_stage=cfg.prompt_output_capture_stage,
    )

    class FakeTextAI:
        def __init__(self):
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            self.calls += 1
            return f"resp_tot_{self.calls}"

    ctx = _make_ctx(cfg, generation_id="tot")
    runner = ChatRunner(ai_text=FakeTextAI())
    plan.execute(ctx, runner, resolved, inputs)

    assert any("tot_enclave" in (step.get("path") or "") for step in ctx.steps)


def test_per_stage_refinement_override_works(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["refinement"] = {"policy": "tot"}
    cfg_dict["prompt"]["stages"] = {
        "include": ["initial_prompt", "section_2_choice"],
        "overrides": {"initial_prompt": {"refinement_policy": "none"}},
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    plan = PromptPlanManager.get("standard")
    stage_specs, inputs = _make_standard_stage_specs(cfg)

    resolved = resolve_stage_specs(
        stage_specs,
        plan_name=plan.name,
        include=cfg.prompt_stages_include,
        exclude=cfg.prompt_stages_exclude,
        overrides=cfg.prompt_stages_overrides,
        capture_stage=cfg.prompt_output_capture_stage,
    )

    class FakeTextAI:
        def __init__(self):
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            self.calls += 1
            return f"resp_{self.calls}"

    ctx = _make_ctx(cfg, generation_id="override")
    runner = ChatRunner(ai_text=FakeTextAI())
    plan.execute(ctx, runner, resolved, inputs)

    initial_paths = [
        step.get("path") or ""
        for step in ctx.steps
        if (step.get("path") or "").startswith("pipeline/standard.initial_prompt/")
    ]
    section2_paths = [
        step.get("path") or ""
        for step in ctx.steps
        if (step.get("path") or "").startswith("pipeline/standard.section_2_choice/")
    ]

    assert initial_paths
    assert all("tot_enclave" not in path for path in initial_paths)

    assert section2_paths
    assert any("tot_enclave" in path for path in section2_paths)


def test_baseline_capture_first_stage_output(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["refinement"] = {"policy": "none"}
    cfg_dict["prompt"]["stages"] = {"include": ["initial_prompt"]}
    cfg_dict["prompt"]["output"] = {"capture_stage": "initial_prompt"}
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    plan = PromptPlanManager.get("standard")
    stage_specs, inputs = _make_standard_stage_specs(cfg)
    resolved = resolve_stage_specs(
        stage_specs,
        plan_name=plan.name,
        include=cfg.prompt_stages_include,
        exclude=cfg.prompt_stages_exclude,
        overrides=cfg.prompt_stages_overrides,
        capture_stage=cfg.prompt_output_capture_stage,
    )

    class FakeTextAI:
        def __init__(self):
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            self.calls += 1
            return "FIRST_STAGE_RESPONSE" if self.calls == 1 else f"resp_{self.calls}"

    ctx = _make_ctx(cfg, generation_id="baseline")
    runner = ChatRunner(ai_text=FakeTextAI())
    plan.execute(ctx, runner, resolved, inputs)

    assert ctx.outputs["image_prompt"] == "FIRST_STAGE_RESPONSE"

