import logging
import random

import pytest

from message_handling import MessageHandler
from pipeline import Block, ChatRunner, ChatStep, RunContext
from run_config import RunConfig


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

    logger = logging.getLogger("test.step_block_structure")
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


def test_merge_none_leaves_parent_unchanged_and_records_transcript(tmp_path):
    class FakeTextAI:
        def __init__(self):
            self._responses = ["A", "B"]

        def text_chat(self, messages, **kwargs):
            return self._responses.pop(0)

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    inner = Block(
        name="inner",
        merge="none",
        nodes=[
            ChatStep(name="s1", prompt="p1", temperature=0.0),
            ChatStep(name="s2", prompt="p2", temperature=0.0),
        ],
    )
    root = Block(name="pipeline", merge="all_messages", nodes=[inner])

    runner.run(ctx, root)

    assert ctx.messages.messages == [{"role": "system", "content": "system"}]
    assert [step["name"] for step in ctx.steps] == ["s1", "s2"]
    assert [step["path"] for step in ctx.steps] == ["pipeline/inner/s1", "pipeline/inner/s2"]


def test_merge_last_response_appends_exactly_one_assistant_message(tmp_path):
    class FakeTextAI:
        def __init__(self):
            self._responses = ["A", "B"]

        def text_chat(self, messages, **kwargs):
            return self._responses.pop(0)

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    block = Block(
        name="inner",
        merge="last_response",
        nodes=[
            ChatStep(name="step1", prompt="p1", temperature=0.0),
            ChatStep(name="step2", prompt="p2", temperature=0.0),
        ],
    )
    root = Block(name="pipeline", merge="all_messages", nodes=[block])

    runner.run(ctx, root)

    assert ctx.messages.messages == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "B"},
    ]


def test_merge_all_messages_includes_user_and_assistant_pair(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    block = Block(
        name="inner",
        merge="all_messages",
        nodes=[ChatStep(name="s", prompt="hello", temperature=0.0)],
    )

    runner.run(ctx, block)

    assert ctx.messages.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "resp"},
    ]


def test_merge_last_response_raises_if_no_assistant_output_exists(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    empty = Block(name="empty", merge="last_response", nodes=[])
    with pytest.raises(ValueError, match="last_response requested but no assistant output exists"):
        runner.run(ctx, empty)

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    hidden = Block(
        name="hidden",
        merge="none",
        nodes=[ChatStep(name="s", prompt="p", temperature=0.0)],
    )
    parent = Block(name="parent", merge="last_response", nodes=[hidden])
    with pytest.raises(ValueError, match="last_response requested but no assistant output exists"):
        runner.run(ctx, parent)


def test_nested_paths_are_unique_for_reused_tot_blocks(tmp_path):
    class FakeTextAI:
        def __init__(self):
            self._responses = ["r1", "r2", "r3", "r4", "r5", "r6"]

        def text_chat(self, messages, **kwargs):
            return self._responses.pop(0)

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    tot = Block(
        name="tot_enclave",
        merge="all_messages",
        nodes=[
            ChatStep(name="enclave_opinion", prompt="opinion", temperature=0.0),
            ChatStep(name="enclave_consensus", prompt="consensus", temperature=0.0),
        ],
    )

    stage1 = Block(
        name="stage1",
        merge="last_response",
        nodes=[ChatStep(name="stage1_step", prompt="stage1", temperature=0.0), tot],
    )
    stage2 = Block(
        name="stage2",
        merge="last_response",
        nodes=[ChatStep(name="stage2_step", prompt="stage2", temperature=0.0), tot],
    )

    root = Block(name="pipeline", merge="all_messages", nodes=[stage1, stage2])
    runner.run(ctx, root)

    consensus_paths = [step["path"] for step in ctx.steps if step["name"] == "enclave_consensus"]
    assert consensus_paths == [
        "pipeline/stage1/tot_enclave/enclave_consensus",
        "pipeline/stage2/tot_enclave/enclave_consensus",
    ]


def test_no_dangling_messages_when_model_raises(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            last_user = messages[-1]["content"]
            if last_user == "fail":
                raise RuntimeError("boom")
            return "ok"

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            ChatStep(name="a", prompt="ok", temperature=0.0),
            ChatStep(name="b", prompt="fail", temperature=0.0),
        ],
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(ctx, root)

    assert ctx.messages.messages == [{"role": "system", "content": "system"}]
    assert [step["name"] for step in ctx.steps] == ["a"]


def test_tot_wrapping_reduces_persisted_context_but_keeps_full_transcript(tmp_path):
    class FakeTextAI:
        def __init__(self):
            self._responses = [
                "stage1_out",
                "stage1_opinion",
                "stage1_consensus",
                "stage2_out",
                "stage2_opinion",
                "stage2_consensus",
            ]

        def text_chat(self, messages, **kwargs):
            return self._responses.pop(0)

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    tot = Block(
        name="tot_enclave",
        merge="all_messages",
        nodes=[
            ChatStep(name="enclave_opinion", prompt="opinion", temperature=0.0),
            ChatStep(name="enclave_consensus", prompt="consensus", temperature=0.0),
        ],
    )

    stage1 = Block(
        name="stage1",
        merge="last_response",
        nodes=[ChatStep(name="stage1_step", prompt="stage1", temperature=0.0), tot],
    )
    stage2 = Block(
        name="stage2",
        merge="last_response",
        nodes=[ChatStep(name="stage2_step", prompt="stage2", temperature=0.0), tot],
    )

    root = Block(name="pipeline", merge="all_messages", nodes=[stage1, stage2])
    runner.run(ctx, root)

    assert ctx.messages.messages == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "stage1_consensus"},
        {"role": "assistant", "content": "stage2_consensus"},
    ]

    assert len(ctx.steps) == 6
    assert [step["path"] for step in ctx.steps] == [
        "pipeline/stage1/stage1_step",
        "pipeline/stage1/tot_enclave/enclave_opinion",
        "pipeline/stage1/tot_enclave/enclave_consensus",
        "pipeline/stage2/stage2_step",
        "pipeline/stage2/tot_enclave/enclave_opinion",
        "pipeline/stage2/tot_enclave/enclave_consensus",
    ]


def test_step_merge_none_does_not_affect_subsequent_steps(tmp_path):
    class FakeTextAI:
        def __init__(self):
            self.calls: list[list[dict[str, str]]] = []
            self._responses = ["A", "B"]

        def text_chat(self, messages, **kwargs):
            self.calls.append(messages)
            return self._responses.pop(0)

    ctx = _make_ctx(tmp_path)
    fake = FakeTextAI()
    runner = ChatRunner(ai_text=fake)

    root = Block(
        name="pipeline",
        merge="none",
        nodes=[
            ChatStep(name="s1", prompt="p1", temperature=0.0, merge="none"),
            ChatStep(name="s2", prompt="p2", temperature=0.0, merge="none"),
        ],
    )

    runner.run(ctx, root)

    assert [len(call) for call in fake.calls] == [2, 2]
    assert [call[-1]["content"] for call in fake.calls] == ["p1", "p2"]
    assert all(message["content"] != "p1" for message in fake.calls[1])
    assert all(message["content"] != "A" for message in fake.calls[1])

    assert ctx.messages.messages == [{"role": "system", "content": "system"}]
    assert [step["name"] for step in ctx.steps] == ["s1", "s2"]


def test_step_merge_last_response_appends_only_assistant_message(tmp_path):
    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    ctx = _make_ctx(tmp_path)
    runner = ChatRunner(ai_text=FakeTextAI())

    root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[ChatStep(name="s1", prompt="p1", temperature=0.0, merge="last_response")],
    )

    runner.run(ctx, root)

    assert ctx.messages.messages == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "resp"},
    ]


def test_step_invalid_merge_mode_raises():
    with pytest.raises(ValueError, match=r"Invalid step merge mode"):
        ChatStep(name="bad", prompt="p", temperature=0.0, merge="bad")  # type: ignore[arg-type]


def test_threaded_enclave_steps_are_independent_and_consensus_uses_captures(tmp_path):
    class FakeTextAI:
        def __init__(self):
            self.calls: list[list[dict[str, str]]] = []
            self._responses = ["DRAFT_OUT", "A1_OUT", "A2_OUT", "REFINED_OUT"]

        def text_chat(self, messages, **kwargs):
            self.calls.append(messages)
            return self._responses.pop(0)

    ctx = _make_ctx(tmp_path)
    fake = FakeTextAI()
    runner = ChatRunner(ai_text=fake)

    def consensus_prompt(inner_ctx: RunContext) -> str:
        a1 = inner_ctx.outputs.get("enclave.stage.a1")
        if not isinstance(a1, str) or not a1.strip():
            raise ValueError("Missing enclave.stage.a1")
        a2 = inner_ctx.outputs.get("enclave.stage.a2")
        if not isinstance(a2, str) or not a2.strip():
            raise ValueError("Missing enclave.stage.a2")
        return (
            "Combine these independent notes to refine the last assistant response.\n"
            f"A1:\n{a1}\n\nA2:\n{a2}\n\n"
            "Return ONLY the revised response."
        )

    enclave = Block(
        name="tot_enclave",
        merge="all_messages",
        nodes=[
            ChatStep(
                name="a1",
                prompt="artist1",
                temperature=0.0,
                merge="none",
                capture_key="enclave.stage.a1",
            ),
            ChatStep(
                name="a2",
                prompt="artist2",
                temperature=0.0,
                merge="none",
                capture_key="enclave.stage.a2",
            ),
            ChatStep(name="consensus", prompt=consensus_prompt, temperature=0.0),
        ],
    )

    stage = Block(
        name="stage",
        merge="last_response",
        nodes=[ChatStep(name="draft", prompt="draft", temperature=0.0), enclave],
    )
    root = Block(name="pipeline", merge="all_messages", nodes=[stage])

    runner.run(ctx, root)

    assert ctx.outputs["enclave.stage.a1"] == "A1_OUT"
    assert ctx.outputs["enclave.stage.a2"] == "A2_OUT"

    artist2_messages = fake.calls[2]
    assert all(message["content"] != "artist1" for message in artist2_messages)
    assert all(message["content"] != "A1_OUT" for message in artist2_messages)

    consensus_user_prompt = fake.calls[3][-1]["content"]
    assert "A1_OUT" in consensus_user_prompt
    assert "A2_OUT" in consensus_user_prompt

    assert ctx.messages.messages == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "REFINED_OUT"},
    ]
