import json
import logging
import random

import pytest

from pipelinekit.engine.messages import MessageHandler
from pipelinekit.engine.pipeline import Block, ChatRunner, ChatStep, NullStepRecorder
from image_project.framework.config import RunConfig
from image_project.framework.runtime import RunContext
from image_project.framework.artifacts import write_transcript
from image_project.stages.refine.tot_enclave_prompts import make_tot_enclave_block


def _make_ctx(tmp_path, logger_name: str = "test.step_recorder") -> RunContext:
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

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return RunContext(
        generation_id="gen_recorder",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )


def test_default_recorder_appends_step_records(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp_default"

    ctx = _make_ctx(tmp_path, "test.default_recorder")
    runner = ChatRunner(ai_text=FakeTextAI())

    runner.run_step(ctx, ChatStep(name="step_a", prompt="hello", temperature=0.1))

    assert len(ctx.steps) == 1
    record = ctx.steps[0]
    for key in [
        "name",
        "path",
        "prompt",
        "response",
        "params",
        "prompt_chars",
        "response_chars",
        "context_chars",
        "input_chars",
        "context_messages",
        "input_messages",
        "created_at",
    ]:
        assert key in record
    assert "step_a" in record["path"]


def test_null_step_recorder_skips_transcript_append(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp_null"

    ctx = _make_ctx(tmp_path, "test.null_recorder")
    runner = ChatRunner(ai_text=FakeTextAI(), recorder=NullStepRecorder())

    runner.run_step(ctx, ChatStep(name="silent_step", prompt="hi", temperature=0.1))

    assert ctx.steps == []


def test_custom_recorder_receives_paths_and_metrics(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp_custom"

    class RecordingRecorder:
        def __init__(self) -> None:
            self.start = None
            self.end = None
            self.errors: list[tuple[str, str, Exception]] = []

        def on_step_start(self, ctx, path, **metrics):
            self.start = (path, metrics)

        def on_step_end(self, ctx, record):
            self.end = record

        def on_step_error(self, ctx, path, step_name, exc):
            self.errors.append((path, step_name, exc))

    recorder = RecordingRecorder()
    ctx = _make_ctx(tmp_path, "test.custom_recorder")
    runner = ChatRunner(ai_text=FakeTextAI(), recorder=recorder)

    runner.run_step(ctx, ChatStep(name="custom_step", prompt="hello", temperature=0.1))

    assert recorder.start is not None
    path, metrics = recorder.start
    assert path == "custom_step"
    assert metrics["prompt_chars"] > 0
    assert metrics["context_messages"] == 1  # system prompt only

    assert recorder.end is not None
    assert recorder.end["path"] == "custom_step"
    assert not recorder.errors


def test_api_surface_exports_expected_symbols():
    from pipelinekit.engine import pipeline  # noqa: PLC0415
    from image_project.stages.refine import tot_enclave_prompts  # noqa: PLC0415

    assert hasattr(pipeline, "ChatRunner")
    assert hasattr(pipeline, "NullStepRecorder")
    assert hasattr(tot_enclave_prompts, "make_tot_enclave_block")


def test_tot_enclave_block_shape():
    enclave = make_tot_enclave_block("refine.tot_enclave")
    assert isinstance(enclave, Block)
    assert enclave.name == "tot_enclave"
    assert enclave.merge == "all_messages"
    assert any(isinstance(node, Block) and node.name == "fanout" for node in enclave.nodes)
    assert any(isinstance(node, Block) and node.name == "reduce" for node in enclave.nodes)

    def _walk(node):
        if isinstance(node, Block):
            for child in node.nodes:
                yield from _walk(child)
            return
        yield node

    steps = [node for node in _walk(enclave) if isinstance(node, ChatStep)]
    assert any(step.name == "hemingway" for step in steps)
    assert any(step.name == "final_consensus" for step in steps)


def test_tot_enclave_stage_runs_and_writes_transcript(tmp_path):
    class FakeTextAI:
        def __init__(self) -> None:
            self.call_index = 0

        def text_chat(self, messages, **kwargs):
            value = f"resp_{self.call_index}"
            self.call_index += 1
            return value

    ctx = _make_ctx(tmp_path, "test.tot_integration")
    pipeline_root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            Block(
                name="refine.tot_enclave",
                merge="last_response",
                nodes=[make_tot_enclave_block("refine.tot_enclave")],
                capture_key="final_output",
            )
        ],
    )

    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, pipeline_root)

    assert ctx.outputs["final_output"]
    transcript_path = tmp_path / "transcript.json"
    write_transcript(str(transcript_path), ctx)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert any("tot_enclave" in step["path"] for step in transcript["steps"])


def test_tot_enclave_steps_have_own_provenance(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    ctx = _make_ctx(tmp_path, "test.tot_provenance")
    pipeline_root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            Block(
                name="refine.tot_enclave",
                merge="last_response",
                nodes=[make_tot_enclave_block("refine.tot_enclave")],
                capture_key="final_output",
            )
        ],
    )

    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, pipeline_root)

    hemingway = next(
        step
        for step in ctx.steps
        if step.get("path") == "pipeline/refine.tot_enclave/tot_enclave/fanout/hemingway"
    )
    meta = hemingway.get("meta") or {}
    assert meta.get("source") == "refinement_enclave.enclave_thread_prompt"
    assert meta.get("doc") == "ToT enclave thread (Hemingway): critique + edits."


def test_tot_enclave_stage_params_propagate_to_enclave_steps(tmp_path):
    class RecordingTextAI:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def text_chat(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            return "resp"

    fake_ai = RecordingTextAI()
    ctx = _make_ctx(tmp_path, "test.tot_params")
    enclave = make_tot_enclave_block("refine.tot_enclave", params={"model": "model-x"})
    pipeline_root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            Block(
                name="refine.tot_enclave",
                merge="last_response",
                nodes=[enclave],
                capture_key="final_output",
            )
        ],
    )

    runner = ChatRunner(ai_text=fake_ai)
    runner.run(ctx, pipeline_root)

    assert fake_ai.calls
    assert all(call.get("model") == "model-x" for call in fake_ai.calls)


def test_recorder_exception_preserves_pipeline_path(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp_error"

    class BoomRecorder:
        def __init__(self) -> None:
            self.error_calls = 0

        def on_step_start(self, ctx, path, **kwargs):
            return

        def on_step_end(self, ctx, record):
            raise RuntimeError("recorder boom")

        def on_step_error(self, ctx, path, step_name, exc):
            self.error_calls += 1

    recorder = BoomRecorder()
    ctx = _make_ctx(tmp_path, "test.recorder_exception")
    runner = ChatRunner(ai_text=FakeTextAI(), recorder=recorder)

    with pytest.raises(RuntimeError) as excinfo:
        runner.run_step(ctx, ChatStep(name="fail_step", prompt="prompt", temperature=0.1))

    assert getattr(excinfo.value, "pipeline_path", None) == "fail_step"
    assert recorder.error_calls == 1
