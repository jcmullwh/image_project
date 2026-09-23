import json
import logging
import random

import pandas as pd
import pytest

from image_project.app import generate as app_generate
from pipelinekit.engine.messages import MessageHandler
from pipelinekit.engine.pipeline import Block, ChatRunner, ChatStep
from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import (
    PlanInputs,
    compile_stage_nodes,
    make_pipeline_root_block,
    resolve_stage_blocks,
)
from image_project.framework.prompt_pipeline.pipeline_overrides import (
    PipelineOverrides,
    PromptPipelineConfig,
)
from image_project.framework.runtime import RunContext
from image_project.impl.current.plans import PromptPlanManager
from image_project.stages.registry import get_stage_registry


def _base_cfg_dict(tmp_path) -> dict:
    return {
        "prompt": {
            "plan": "standard",
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


def _make_standard_stage_nodes(cfg: RunConfig) -> tuple[list, PlanInputs]:
    prompt_data = pd.DataFrame(
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
    )
    user_profile = pd.DataFrame({"Likes": ["x"], "Dislikes": [None]})
    inputs = PlanInputs(
        cfg=cfg,
        pipeline=PipelineOverrides(include=(), exclude=(), sequence=(), overrides={}, capture_stage=None),
        ai_text=None,
        prompt_data=prompt_data,
        user_profile=user_profile,
        preferences_guidance="Likes:\n- x",
        context_guidance=None,
        rng=random.Random(0),
    )
    plan = PromptPlanManager.get("standard")
    return plan.stage_nodes(inputs), inputs


def test_unknown_plan_fails_fast(tmp_path, monkeypatch):
    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            pass

        def text_chat(self, messages, **kwargs):
            return "resp"

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "does_not_exist"

    # Avoid importing a real backend in tests.
    monkeypatch.setattr(app_generate, "TextAI", FakeTextAI)

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
        app_generate.run_generation(cfg_dict, generation_id="unit_test_unknown_plan")


def test_unknown_stage_id_in_include_fails_fast(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    stage_nodes, inputs = _make_standard_stage_nodes(cfg)

    with pytest.raises(ValueError, match="prompt\\.stages\\.include"):
        compile_stage_nodes(
            stage_nodes,
            plan_name="standard",
            include=("nope",),
            exclude=(),
            overrides={},
            stage_configs_defaults={},
            stage_configs_instances={},
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )


def test_capture_stage_not_in_resolved_stages_fails_fast(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    stage_nodes, inputs = _make_standard_stage_nodes(cfg)

    with pytest.raises(ValueError, match="prompt\\.output\\.capture_stage"):
        compiled = compile_stage_nodes(
            stage_nodes,
            plan_name="standard",
            include=("select_concepts", "filter_concepts", "initial_prompt"),
            exclude=(),
            overrides={},
            stage_configs_defaults={},
            stage_configs_instances={},
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )
        resolve_stage_blocks(
            list(compiled.blocks),
            plan_name="standard",
            include=(),
            exclude=(),
            overrides=compiled.overrides,
            capture_stage="image_prompt_creation",
        )


def test_refine_tot_enclave_stage_produces_tot_enclave_steps(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    plan = PromptPlanManager.get("standard")
    stage_nodes, inputs = _make_standard_stage_nodes(cfg)

    compiled = compile_stage_nodes(
        stage_nodes,
        plan_name=plan.name,
        include=("select_concepts", "initial_prompt", "refine.tot_enclave"),
        exclude=(),
        overrides={},
        stage_configs_defaults={},
        stage_configs_instances={},
        stage_registry=get_stage_registry(),
        inputs=inputs,
    )
    resolved = resolve_stage_blocks(
        list(compiled.blocks),
        plan_name=plan.name,
        include=(),
        exclude=(),
        overrides=compiled.overrides,
        capture_stage=None,
    )

    class FakeTextAI:
        def __init__(self):
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            self.calls += 1
            return f"resp_{self.calls}"

    ctx = _make_ctx(cfg, generation_id="with_refine")
    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, make_pipeline_root_block(resolved))

    assert any(
        (step.get("path") or "")
        == "pipeline/refine.tot_enclave/tot_enclave/reduce/consensus/select/final_consensus"
        for step in ctx.steps
    )


def test_excluding_refine_tot_enclave_removes_tot_enclave_steps(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    plan = PromptPlanManager.get("standard")
    stage_nodes, inputs = _make_standard_stage_nodes(cfg)

    compiled = compile_stage_nodes(
        stage_nodes,
        plan_name=plan.name,
        include=("select_concepts", "initial_prompt", "refine.tot_enclave"),
        exclude=("refine.tot_enclave",),
        overrides={},
        stage_configs_defaults={},
        stage_configs_instances={},
        stage_registry=get_stage_registry(),
        inputs=inputs,
    )
    resolved = resolve_stage_blocks(
        list(compiled.blocks),
        plan_name=plan.name,
        include=(),
        exclude=(),
        overrides=compiled.overrides,
        capture_stage=None,
    )

    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    ctx = _make_ctx(cfg, generation_id="exclude_refine")
    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, make_pipeline_root_block(resolved))

    assert all("pipeline/refine.tot_enclave/" not in (step.get("path") or "") for step in ctx.steps)


def test_baseline_capture_first_stage_output(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["stages"] = {"include": ["initial_prompt"]}
    cfg_dict["prompt"]["output"] = {"capture_stage": "initial_prompt"}
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    prompt_cfg, _prompt_warnings = PromptPipelineConfig.from_root_dict(
        cfg_dict, run_mode=cfg.run_mode, generation_dir=cfg.generation_dir
    )

    plan = PromptPlanManager.get("standard")
    stage_nodes, inputs = _make_standard_stage_nodes(cfg)
    inputs = PlanInputs(
        cfg=cfg,
        pipeline=prompt_cfg.stages,
        ai_text=inputs.ai_text,
        prompt_data=inputs.prompt_data,
        user_profile=inputs.user_profile,
        preferences_guidance=inputs.preferences_guidance,
        context_guidance=inputs.context_guidance,
        rng=inputs.rng,
    )
    compiled = compile_stage_nodes(
        stage_nodes,
        plan_name=plan.name,
        include=prompt_cfg.stages.include,
        exclude=prompt_cfg.stages.exclude,
        overrides=prompt_cfg.stages.overrides,
        stage_configs_defaults=prompt_cfg.stage_configs_defaults,
        stage_configs_instances=prompt_cfg.stage_configs_instances,
        initial_outputs=("selected_concepts",),
        stage_registry=get_stage_registry(),
        inputs=inputs,
    )
    resolved = resolve_stage_blocks(
        list(compiled.blocks),
        plan_name=plan.name,
        include=(),
        exclude=(),
        overrides=compiled.overrides,
        capture_stage=prompt_cfg.stages.capture_stage,
    )

    class FakeTextAI:
        def __init__(self):
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            self.calls += 1
            return "FIRST_STAGE_RESPONSE" if self.calls == 1 else f"resp_{self.calls}"

    ctx = _make_ctx(cfg, generation_id="baseline")
    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, make_pipeline_root_block(resolved))

    assert ctx.outputs["image_prompt"] == "FIRST_STAGE_RESPONSE"


def test_capture_stage_must_produce_assistant_output(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))

    stage_blocks = [
        Block(
            name="stage_a",
            merge="none",
            nodes=[ChatStep(name="cand", prompt="p", temperature=0.0, merge="none")],
        )
    ]

    with pytest.raises(ValueError, match=r"capture_stage='stage_a'.*produces no assistant output"):
        resolve_stage_blocks(
            stage_blocks,
            plan_name="test",
            include=(),
            exclude=(),
            overrides={},
            capture_stage="stage_a",
        )


def test_default_capture_skips_nonproducing_chat_stages(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))

    stage_blocks = [
        Block(
            name="stage_a",
            merge="none",
            nodes=[ChatStep(name="draft", prompt="p1", temperature=0.0)],
        ),
        Block(
            name="stage_b",
            merge="none",
            nodes=[ChatStep(name="cand", prompt="p2", temperature=0.0, merge="none")],
        ),
    ]

    resolved = resolve_stage_blocks(
        stage_blocks,
        plan_name="test",
        include=(),
        exclude=(),
        overrides={},
        capture_stage=None,
    )
    assert resolved.capture_stage == "stage_a"

    class FakeTextAI:
        def __init__(self):
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            self.calls += 1
            return "A" if self.calls == 1 else "B"

    ctx = _make_ctx(cfg, generation_id="capture_skip_nonproducing")
    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, make_pipeline_root_block(resolved))
    assert ctx.outputs["image_prompt"] == "A"
