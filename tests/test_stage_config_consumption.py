import logging
import random

import pandas as pd
import pytest

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.messages import MessageHandler
from pipelinekit.engine.pipeline import Block, ChatRunner, ChatStep
from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import (
    PlanInputs,
    compile_stage_nodes,
    make_action_stage_block,
    make_chat_stage_block,
)
from image_project.framework.prompt_pipeline.pipeline_overrides import PipelineOverrides
from image_project.framework.runtime import RunContext
from image_project.stages.blackbox_refine.loop import STAGE as BLACKBOX_REFINE_LOOP
from image_project.stages.refine.tot_enclave import STAGE as REFINE_TOT_ENCLAVE
from image_project.stages.registry import get_stage_registry
from pipelinekit.stage_types import StageIO, StageRef


def _build_test_seed_stage(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace) -> Block:
    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt="seed",
        temperature=0.0,
        merge="last_response",
    )


TEST_SEED_STAGE = StageRef(
    id="test.seed",
    builder=_build_test_seed_stage,
    doc="Test-only seed stage (emits a deterministic draft).",
    source="tests.test_stage_config_consumption._build_test_seed_stage",
    tags=("test",),
    kind="chat",
)


def _build_test_bbref_seed_stage(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace) -> Block:
    """Build a test-only stage that provides bbref.seed_prompt via an action capture."""

    def _action(_ctx: RunContext) -> str:
        return "SEED"

    cfg.assert_consumed()
    return make_action_stage_block(
        instance_id,
        fn=_action,
        merge="none",
        step_capture_key="bbref.seed_prompt",
        doc="Test-only seed provider for blackbox_refine.loop.",
        source="tests.test_stage_config_consumption._build_test_bbref_seed_stage",
        tags=("test",),
    )


TEST_BBREF_SEED_STAGE = StageRef(
    id="test.bbref_seed",
    builder=_build_test_bbref_seed_stage,
    doc="Test-only action stage that provides bbref.seed_prompt.",
    source="tests.test_stage_config_consumption._build_test_bbref_seed_stage",
    tags=("test",),
    kind="action",
    io=StageIO(provides=("bbref.seed_prompt",), captures=("bbref.seed_prompt",)),
)


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


def _make_ctx(cfg: RunConfig, *, generation_id: str) -> RunContext:
    logger = logging.getLogger(f"test.stage_config.{generation_id}")
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
        selected_concepts=["A", "B"],
    )


def test_tot_enclave_stage_config_selects_critics_and_records_effective(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    stage_configs_defaults: dict[str, dict[str, object]] = {}
    stage_configs_instances = {"refine.tot_enclave": {"critics": ["hemingway", "munch"]}}

    stage_nodes = [
        TEST_SEED_STAGE.instance("seed"),
        REFINE_TOT_ENCLAVE.instance(),
    ]

    compiled = compile_stage_nodes(
        stage_nodes,
        plan_name="unit",
        include=(),
        exclude=(),
        overrides={},
        stage_configs_defaults=stage_configs_defaults,
        stage_configs_instances=stage_configs_instances,
        stage_registry=get_stage_registry(),
        inputs=inputs,
    )

    effective = compiled.metadata.get("stage_configs_effective") or {}
    assert effective["refine.tot_enclave"]["critics"] == ["hemingway", "munch"]

    class FakeTextAI:
        def __init__(self) -> None:
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            self.calls += 1
            return f"resp_{self.calls}"

    ctx = _make_ctx(cfg, generation_id="tot_critics")
    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, Block(name="pipeline", merge="all_messages", nodes=list(compiled.blocks)))

    paths = [step.get("path") for step in ctx.steps]
    assert "pipeline/refine.tot_enclave/tot_enclave/fanout/hemingway" in paths
    assert "pipeline/refine.tot_enclave/tot_enclave/fanout/munch" in paths
    assert "pipeline/refine.tot_enclave/tot_enclave/fanout/da_vinci" not in paths
    assert "pipeline/refine.tot_enclave/tot_enclave/fanout/representative" not in paths
    assert "pipeline/refine.tot_enclave/tot_enclave/fanout/chameleon" not in paths


def test_tot_enclave_stage_config_unknown_key_fails_fast(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    stage_configs_defaults: dict[str, dict[str, object]] = {}
    stage_configs_instances = {"refine.tot_enclave": {"typo_key": True}}

    stage_nodes = [
        TEST_SEED_STAGE.instance("seed"),
        REFINE_TOT_ENCLAVE.instance(),
    ]

    with pytest.raises(
        ValueError,
        match=r"Unknown config keys under prompt\.stage_configs\.resolved\.refine\.tot_enclave: typo_key",
    ):
        compile_stage_nodes(
            stage_nodes,
            plan_name="unit",
            include=(),
            exclude=(),
            overrides={},
            stage_configs_defaults=stage_configs_defaults,
            stage_configs_instances=stage_configs_instances,
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )


def test_blackbox_refine_loop_stage_config_controls_judges_and_candidates_and_records_effective(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    stage_configs_defaults: dict[str, dict[str, object]] = {}
    stage_configs_instances = {
        "blackbox_refine.loop": {
            "iterations": 1,
            "algorithm": "hillclimb",
            "branching_factor": 1,
            "include_parents_as_candidates": False,
            "exploration_rate": 0.0,
            "variation_prompt": {
                "template": "v1",
                "include_profile": False,
                "score_feedback": "none",
            },
            "judging": {"judges": [{"id": "j1"}, {"id": "j2"}], "aggregation": "mean"},
        }
    }

    stage_nodes = [
        TEST_BBREF_SEED_STAGE.instance("seed"),
        BLACKBOX_REFINE_LOOP.instance(),
    ]
    compiled = compile_stage_nodes(
        stage_nodes,
        plan_name="unit",
        include=(),
        exclude=(),
        overrides={},
        stage_configs_defaults=stage_configs_defaults,
        stage_configs_instances=stage_configs_instances,
        stage_registry=get_stage_registry(),
        inputs=inputs,
    )

    effective = compiled.metadata.get("stage_configs_effective") or {}
    assert effective["blackbox_refine.loop"]["branching_factor"] == 1
    judge_ids_effective = [row.get("id") for row in effective["blackbox_refine.loop"]["judging"]["judges"]]
    assert judge_ids_effective == ["j1", "j2"]

    loop_block = next(block for block in compiled.blocks if block.name == "blackbox_refine.loop")
    iter_block = next(
        node for node in loop_block.nodes if isinstance(node, Block) and node.name == "blackbox_refine.iter_01"
    )
    judge_block = next(node for node in iter_block.nodes if isinstance(node, Block) and node.name == "judge")
    judge_ids = [node.name for node in judge_block.nodes if isinstance(node, ChatStep)]
    assert judge_ids == ["j1", "j2"]

    beam_block = next(node for node in iter_block.nodes if isinstance(node, Block) and node.name == "beam_01")
    candidates = [
        node
        for node in beam_block.nodes
        if isinstance(node, ChatStep) and (node.name or "").startswith("cand_")
    ]
    assert len(candidates) == 1


def test_blackbox_refine_loop_stage_config_unknown_key_fails_fast(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    stage_configs_defaults: dict[str, dict[str, object]] = {}
    stage_configs_instances = {
        "blackbox_refine.loop": {
            "iterations": 1,
            "algorithm": "hillclimb",
            "branching_factor": 1,
            "include_parents_as_candidates": False,
            "exploration_rate": 0.0,
            "variation_prompt": {
                "template": "v1",
                "include_profile": False,
                "score_feedback": "none",
            },
            "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
            "typo_key": True,
        }
    }

    stage_nodes = [
        TEST_BBREF_SEED_STAGE.instance("seed"),
        BLACKBOX_REFINE_LOOP.instance(),
    ]

    with pytest.raises(
        ValueError,
        match=r"Unknown config keys under prompt\.stage_configs\.resolved\.blackbox_refine\.loop: typo_key",
    ):
        compile_stage_nodes(
            stage_nodes,
            plan_name="unit",
            include=(),
            exclude=(),
            overrides={},
            stage_configs_defaults=stage_configs_defaults,
            stage_configs_instances=stage_configs_instances,
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )
