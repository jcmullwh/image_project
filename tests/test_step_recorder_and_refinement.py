import json
import logging
import random

import pytest

from message_handling import MessageHandler
from pipeline import (
    Block,
    ChatRunner,
    ChatStep,
    RunContext,
    NullStepRecorder,
)
from refinement import NoRefinement, TotEnclaveRefinement
from run_config import RunConfig
from transcript import write_transcript


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
    import pipeline  # noqa: PLC0415
    import refinement  # noqa: PLC0415

    assert hasattr(pipeline, "ChatRunner")
    assert hasattr(pipeline, "NullStepRecorder")
    assert hasattr(refinement, "NoRefinement")
    assert hasattr(refinement, "TotEnclaveRefinement")


def test_refinement_policy_stage_shapes():
    no_refinement = NoRefinement()
    stage = no_refinement.stage("stage_one", prompt="text", temperature=0.1)
    assert isinstance(stage, Block)
    assert stage.merge == "last_response"
    assert stage.capture_key is None
    assert isinstance(stage.nodes[0], ChatStep)
    assert stage.nodes[0].name == "draft"

    tot_refinement = TotEnclaveRefinement()
    stage_tot = tot_refinement.stage("stage_two", prompt="text", temperature=0.1, capture_key="cap")
    assert isinstance(stage_tot, Block)
    assert stage_tot.capture_key == "cap"
    assert stage_tot.merge == "last_response"
    assert isinstance(stage_tot.nodes[0], ChatStep)
    assert stage_tot.nodes[0].name == "draft"
    assert len(stage_tot.nodes) >= 2
    assert isinstance(stage_tot.nodes[1], Block)


def test_tot_refinement_pipeline_runs_and_writes_transcript(tmp_path):
    class FakeTextAI:
        def __init__(self) -> None:
            self.call_index = 0

        def text_chat(self, messages, **kwargs):
            value = f"resp_{self.call_index}"
            self.call_index += 1
            return value

    ctx = _make_ctx(tmp_path, "test.tot_integration")
    refinement = TotEnclaveRefinement()
    pipeline_root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            refinement.stage(
                "tot_stage",
                prompt="draft prompt",
                temperature=0.1,
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
    assert any("draft" in step["path"] for step in transcript["steps"])
    assert any("tot_enclave" in step["path"] for step in transcript["steps"])


def test_tot_refinement_enclave_steps_have_own_provenance(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    ctx = _make_ctx(tmp_path, "test.tot_provenance")
    refinement = TotEnclaveRefinement()
    pipeline_root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            refinement.stage(
                "tot_stage",
                prompt="draft prompt",
                temperature=0.1,
                meta={"source": "parent.source", "doc": "Parent doc"},
                capture_key="final_output",
            )
        ],
    )

    runner = ChatRunner(ai_text=FakeTextAI())
    runner.run(ctx, pipeline_root)

    hemingway = next(
        step for step in ctx.steps if step.get("path") == "pipeline/tot_stage/tot_enclave/hemingway"
    )
    meta = hemingway.get("meta") or {}
    assert meta.get("source") == "refinement_enclave.enclave_thread_prompt"
    assert meta.get("doc") == "ToT enclave thread (Hemingway): critique + edits."


def test_tot_refinement_stage_params_propagate_to_enclave_steps(tmp_path):
    class RecordingTextAI:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def text_chat(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            return "resp"

    fake_ai = RecordingTextAI()
    ctx = _make_ctx(tmp_path, "test.tot_params")
    refinement = TotEnclaveRefinement()
    pipeline_root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            refinement.stage(
                "tot_stage",
                prompt="draft prompt",
                temperature=0.1,
                params={"model": "model-x"},
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
