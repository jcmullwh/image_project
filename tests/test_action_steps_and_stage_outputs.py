import logging
import random

import pytest

from pipelinekit.engine.messages import MessageHandler
from pipelinekit.engine.pipeline import ActionStep, Block, ChatRunner
from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import (
    make_chat_stage_block,
    make_pipeline_root_block,
    resolve_stage_blocks,
)
from image_project.framework.runtime import RunContext


def _make_ctx(tmp_path) -> RunContext:
    cfg_dict = {
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
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.action_steps")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return RunContext(
        generation_id="gen",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )


def test_action_step_records_and_captures_output(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            raise AssertionError("text_chat should not be called for action-only pipelines")

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    def _action(inner_ctx: RunContext) -> dict[str, str]:
        inner_ctx.outputs["foo"] = "bar"
        return {"ok": "yes"}

    root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            ActionStep(name="act", fn=_action, capture_key="action_result"),
        ],
    )

    runner.run(ctx, root)

    assert ctx.messages.messages == [{"role": "system", "content": "system"}]
    assert ctx.outputs["foo"] == "bar"
    assert ctx.outputs["action_result"] == {"ok": "yes"}

    assert len(ctx.steps) == 1
    assert ctx.steps[0]["type"] == "action"
    assert ctx.steps[0]["path"] == "pipeline/act"
    assert ctx.steps[0]["result"] == {"ok": "yes"}


def test_stage_output_key_captures_intermediate_outputs(tmp_path):
    class FakeTextAI:
        def __init__(self) -> None:
            self.responses = ["A_OUT", "B_OUT"]

        def text_chat(self, messages, **kwargs):
            return self.responses.pop(0)

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    stage_blocks = [
        make_chat_stage_block(
            "stage_a",
            prompt="prompt_a",
            temperature=0.0,
            merge="none",
            step_capture_key="a_out",
        ),
        make_chat_stage_block("stage_b", prompt="prompt_b", temperature=0.0),
    ]

    resolved = resolve_stage_blocks(
        stage_blocks,
        plan_name="test",
        include=(),
        exclude=(),
        overrides={},
        capture_stage="stage_b",
        capture_key="final_out",
    )

    pipeline_root = make_pipeline_root_block(resolved)
    runner.run(ctx, pipeline_root)

    assert ctx.outputs["a_out"] == "A_OUT"
    assert ctx.outputs["final_out"] == "B_OUT"


def test_capture_stage_and_step_capture_key_can_coexist(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "OK"

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    stage_blocks = [
        make_chat_stage_block(
            "stage_a",
            prompt="prompt_a",
            temperature=0.0,
            step_capture_key="other_key",
        ),
    ]
    resolved = resolve_stage_blocks(
        stage_blocks,
        plan_name="test",
        include=(),
        exclude=(),
        overrides={},
        capture_stage="stage_a",
        capture_key="image_prompt",
    )

    runner.run(ctx, make_pipeline_root_block(resolved))

    assert ctx.outputs["other_key"] == "OK"
    assert ctx.outputs["image_prompt"] == "OK"
