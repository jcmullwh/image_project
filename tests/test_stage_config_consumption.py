import logging
import random

import pandas as pd
import pytest

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.messages import MessageHandler
from pipelinekit.engine.pipeline import Block, ChatRunner, ChatStep
from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import PlanInputs, compile_stage_nodes, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.stages.blackbox_refine.loop import (
    BLACKBOX_REFINE_INIT_STATE_STAGE,
    BLACKBOX_REFINE_ITER_STAGE,
)
from image_project.stages.refine.tot_enclave import STAGE as REFINE_TOT_ENCLAVE
from image_project.stages.registry import get_stage_registry
from pipelinekit.stage_types import StageRef


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
    cfg_dict["prompt"]["stage_configs"] = {
        "instances": {
            "refine.tot_enclave": {
                "critics": ["hemingway", "munch"],
            }
        }
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

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
        stage_configs_defaults=cfg.prompt_stage_configs_defaults,
        stage_configs_instances=cfg.prompt_stage_configs_instances,
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
    cfg_dict["prompt"]["stage_configs"] = {
        "instances": {
            "refine.tot_enclave": {
                "typo_key": True,
            }
        }
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

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
            stage_configs_defaults=cfg.prompt_stage_configs_defaults,
            stage_configs_instances=cfg.prompt_stage_configs_instances,
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )


def test_blackbox_iter_stage_config_controls_judges_and_candidates_and_records_effective(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["blackbox_refine"] = {
        "enabled": True,
        "iterations": 1,
        "algorithm": "hillclimb",
        "branching_factor": 3,
        "include_parents_as_candidates": False,
        "generator_temperature": 0.9,
        "variation_prompt": {
            "template": "v1",
            "include_profile": False,
            "include_context_guidance": False,
            "include_novelty_summary": False,
            "include_mutation_directive": False,
            "include_scoring_rubric": False,
        },
        "judging": {"judges": [{"id": "j1"}, {"id": "j2"}], "aggregation": "mean"},
    }
    cfg_dict["prompt"]["stage_configs"] = {
        "instances": {
            "blackbox_refine.iter_01": {
                "judges": ["j2"],
                "candidates_per_iter": 1,
            }
        }
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    stage_nodes = [
        BLACKBOX_REFINE_INIT_STATE_STAGE.instance(),
        BLACKBOX_REFINE_ITER_STAGE.instance("blackbox_refine.iter_01"),
    ]
    compiled = compile_stage_nodes(
        stage_nodes,
        plan_name="unit",
        include=(),
        exclude=(),
        overrides={},
        stage_configs_defaults=cfg.prompt_stage_configs_defaults,
        stage_configs_instances=cfg.prompt_stage_configs_instances,
        stage_registry=get_stage_registry(),
        inputs=inputs,
    )

    effective = compiled.metadata.get("stage_configs_effective") or {}
    assert effective["blackbox_refine.iter_01"]["judges"] == ["j2"]
    assert effective["blackbox_refine.iter_01"]["candidates_per_iter"] == 1

    block = next(block for block in compiled.blocks if block.name == "blackbox_refine.iter_01")
    judge_block = next(node for node in block.nodes if isinstance(node, Block) and node.name == "judge")
    judge_ids = [node.name for node in judge_block.nodes if isinstance(node, ChatStep)]
    assert judge_ids == ["j2"]

    beam_block = next(node for node in block.nodes if isinstance(node, Block) and node.name == "beam_01")
    candidates = [node for node in beam_block.nodes if isinstance(node, ChatStep) and (node.name or "").startswith("cand_")]
    assert len(candidates) == 1


def test_blackbox_iter_stage_config_invalid_judge_fails_loudly(tmp_path):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["blackbox_refine"] = {
        "enabled": True,
        "iterations": 1,
        "algorithm": "hillclimb",
        "branching_factor": 2,
        "include_parents_as_candidates": False,
        "generator_temperature": 0.9,
        "variation_prompt": {
            "template": "v1",
            "include_profile": False,
        },
        "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
    }
    cfg_dict["prompt"]["stage_configs"] = {
        "instances": {
            "blackbox_refine.iter_01": {
                "judges": ["nope"],
            }
        }
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    stage_nodes = [
        BLACKBOX_REFINE_INIT_STATE_STAGE.instance(),
        BLACKBOX_REFINE_ITER_STAGE.instance("blackbox_refine.iter_01"),
    ]
    with pytest.raises(ValueError, match=r"Invalid judge id\(s\).*allowed:"):
        compile_stage_nodes(
            stage_nodes,
            plan_name="unit",
            include=(),
            exclude=(),
            overrides={},
            stage_configs_defaults=cfg.prompt_stage_configs_defaults,
            stage_configs_instances=cfg.prompt_stage_configs_instances,
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )
