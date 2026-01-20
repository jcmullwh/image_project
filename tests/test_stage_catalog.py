import logging
import random

import pandas as pd
import pytest

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.messages import MessageHandler
from pipelinekit.engine.pipeline import Block, ChatStep
from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import PlanInputs
from image_project.framework.prompt_pipeline.pipeline_overrides import PipelineOverrides
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as blackbox_prompts
from image_project.impl.current.plans import StandardPromptPlan
from image_project.stages.registry import get_stage_registry
from pipelinekit.stage_registry import StageRegistry
from pipelinekit.stage_types import StageRef


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
        draft_prompt="Draft prompt",
    )


def test_stage_registry_rejects_duplicate_stage_ids():
    def _builder(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace) -> Block:
        cfg.assert_consumed()
        return Block(name=instance_id, merge="none", nodes=[])

    dupe = StageRef(id="unit_test.dupe", builder=_builder)

    with pytest.raises(ValueError, match=r"Duplicate stage kind id: unit_test\.dupe"):

        StageRegistry.from_refs([dupe, dupe])


def test_stage_registry_unknown_stage_fails_fast():
    registry = get_stage_registry()
    with pytest.raises(ValueError, match=r"Unknown stage kind id: does_not_exist"):
        registry.resolve("does_not_exist")


def test_stage_registry_suffix_lookup_resolves_unambiguous_ids(tmp_path):
    registry = get_stage_registry()
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    ref = registry.resolve("initial_prompt")
    block = ref.build(
        inputs,
        instance_id=ref.id,
        cfg=ConfigNamespace.empty(path=f"prompt.stage_configs.resolved.{ref.id}"),
    )
    assert block.name == "standard.initial_prompt"
    assert block.meta.get("doc") == "Generate candidate themes/stories."
    assert block.meta.get("source") == "prompts.standard.generate_first_prompt"
    assert "standard" in (block.meta.get("tags") or [])


def test_stage_registry_custom_name_records_stage_kind(tmp_path):
    registry = get_stage_registry()
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    ref = registry.resolve("standard.initial_prompt")
    block = ref.build(
        inputs,
        instance_id="my_initial_prompt",
        cfg=ConfigNamespace.empty(path="prompt.stage_configs.resolved.my_initial_prompt"),
    )
    assert block.name == "my_initial_prompt"
    assert block.meta.get("stage_kind") == "standard.initial_prompt"
    assert block.meta.get("stage_instance") == "my_initial_prompt"


def test_standard_plan_stage_nodes_come_from_stage_refs(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    plan = StandardPromptPlan()
    stage_nodes = plan.stage_nodes(inputs)
    assert stage_nodes[0].instance_id == "preprompt.select_concepts"
    assert stage_nodes[1].instance_id == "preprompt.filter_concepts"
    assert stage_nodes[2].instance_id == "standard.initial_prompt"
    assert stage_nodes[-2].instance_id == "standard.image_prompt_creation"
    assert stage_nodes[-2].stage.source == "prompts.standard.generate_image_prompt"
    assert stage_nodes[-1].instance_id == "refine.tot_enclave"


def _make_ctx(cfg: RunConfig, logger_name: str) -> RunContext:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return RunContext(
        generation_id="gen_stage_catalog",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("raw", "RAW"),
        ("generator_hints", "HINTS"),
        (
            "generator_hints_plus_dislikes",
            "Profile extraction (generator-safe hints):\nHINTS\n\nDislikes:\n- gore",
        ),
    ],
)
def test_blackbox_judge_profile_source_routing(tmp_path, monkeypatch, source, expected):
    registry = get_stage_registry()

    def fake_prompt(*, raw_profile: str, **_kwargs) -> str:
        return f"PROFILE={raw_profile}"

    monkeypatch.setattr(blackbox_prompts, "idea_cards_judge_prompt", fake_prompt)

    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    ref = registry.resolve("blackbox.idea_cards_judge_score")
    block = ref.build(
        inputs,
        instance_id=ref.id,
        cfg=ConfigNamespace(
            {"judge_profile_source": source},
            path=f"prompt.stage_configs.resolved.{ref.id}",
        ),
    )
    draft = next(node for node in block.nodes if isinstance(node, ChatStep) and node.name == "draft")

    ctx = _make_ctx(cfg, "test.blackbox.judge_profile_source")
    ctx.outputs["preferences_guidance"] = "RAW"
    ctx.outputs["generator_profile_hints"] = "HINTS"
    ctx.outputs["idea_cards_json"] = "{}"
    ctx.outputs["dislikes"] = ["gore"]

    assert draft.render_prompt(ctx) == f"PROFILE={expected}"


def test_blackbox_judge_profile_source_requires_generator_hints(tmp_path, monkeypatch):
    registry = get_stage_registry()

    def fake_prompt(*, raw_profile: str, **_kwargs) -> str:
        return f"PROFILE={raw_profile}"

    monkeypatch.setattr(blackbox_prompts, "idea_cards_judge_prompt", fake_prompt)

    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    ref = registry.resolve("blackbox.idea_cards_judge_score")
    block = ref.build(
        inputs,
        instance_id=ref.id,
        cfg=ConfigNamespace(
            {"judge_profile_source": "generator_hints"},
            path=f"prompt.stage_configs.resolved.{ref.id}",
        ),
    )
    draft = next(node for node in block.nodes if isinstance(node, ChatStep) and node.name == "draft")

    ctx = _make_ctx(cfg, "test.blackbox.judge_profile_source_missing")
    ctx.outputs["preferences_guidance"] = "RAW"
    ctx.outputs["idea_cards_json"] = "{}"

    with pytest.raises(ValueError, match=r"blackbox\.idea_cards_judge_score requires generator_profile_hints"):
        draft.render_prompt(ctx)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("raw", "RAW"), ("generator_hints", "HINTS"), ("none", "")],
)
def test_blackbox_idea_profile_source_routing(tmp_path, monkeypatch, source, expected):
    registry = get_stage_registry()

    def fake_prompt(*, generator_profile_hints: str, **_kwargs) -> str:
        return f"HINTS={generator_profile_hints}"

    monkeypatch.setattr(blackbox_prompts, "idea_cards_generate_prompt", fake_prompt)

    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    ref = registry.resolve("blackbox.idea_cards_generate")
    block = ref.build(
        inputs,
        instance_id=ref.id,
        cfg=ConfigNamespace(
            {"idea_profile_source": source},
            path=f"prompt.stage_configs.resolved.{ref.id}",
        ),
    )
    draft = next(node for node in block.nodes if isinstance(node, ChatStep) and node.name == "draft")

    ctx = _make_ctx(cfg, "test.blackbox.idea_profile_source")
    ctx.outputs["preferences_guidance"] = "RAW"
    ctx.outputs["generator_profile_hints"] = "HINTS"

    assert draft.render_prompt(ctx) == f"HINTS={expected}"


@pytest.mark.parametrize(
    ("source", "expected"),
    [("raw", "RAW"), ("generator_hints", "HINTS")],
)
def test_blackbox_final_profile_source_routing(tmp_path, monkeypatch, source, expected):
    registry = get_stage_registry()

    def fake_prompt(*, raw_profile: str, **_kwargs) -> str:
        return f"PROFILE={raw_profile}"

    monkeypatch.setattr(blackbox_prompts, "final_prompt_from_selected_idea_prompt", fake_prompt)

    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    ref = registry.resolve("blackbox.image_prompt_creation")
    block = ref.build(
        inputs,
        instance_id=ref.id,
        cfg=ConfigNamespace(
            {"final_profile_source": source},
            path=f"prompt.stage_configs.resolved.{ref.id}",
        ),
    )
    draft = next(node for node in block.nodes if isinstance(node, ChatStep) and node.name == "draft")

    ctx = _make_ctx(cfg, "test.blackbox.final_profile_source")
    ctx.outputs["preferences_guidance"] = "RAW"
    ctx.outputs["generator_profile_hints"] = "HINTS"
    ctx.outputs["selected_idea_card"] = {"id": "A"}

    assert draft.render_prompt(ctx) == f"PROFILE={expected}"
