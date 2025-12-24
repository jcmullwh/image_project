import io
import logging
import random

import pytest
from PIL import Image

from message_handling import MessageHandler
from pipeline import Block, ChatRunner, ChatStep, RunContext
from run_config import RunConfig, parse_bool
from utils import save_image


def test_parse_bool_accepts_valid_values():
    assert parse_bool(True, "x") is True
    assert parse_bool(False, "x") is False
    assert parse_bool(1, "x") is True
    assert parse_bool(0, "x") is False
    assert parse_bool("true", "x") is True
    assert parse_bool("FALSE", "x") is False
    assert parse_bool(" 1 ", "x") is True
    assert parse_bool("0", "x") is False
    assert parse_bool("yes", "x") is True
    assert parse_bool("No", "x") is False


def test_parse_bool_rejects_invalid_values():
    with pytest.raises(ValueError, match=r"Invalid boolean for rclone\.enabled"):
        parse_bool("maybe", "rclone.enabled")
    with pytest.raises(ValueError, match=r"Invalid boolean for upscale\.enabled"):
        parse_bool(2, "upscale.enabled")
    with pytest.raises(ValueError, match=r"Invalid boolean for upscale\.enabled"):
        parse_bool(None, "upscale.enabled")


def test_step_temperature_conflict_raises():
    with pytest.raises(ValueError, match=r"params\['temperature'\]"):
        ChatStep(
            name="a",
            prompt="hello",
            temperature=0.8,
            params={"temperature": 0.1},
        )


def test_invalid_caption_font_path_raises_or_warns(tmp_path):
    im = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")

    with pytest.raises(ValueError, match="Failed to load caption font"):
        save_image(
            buf.getvalue(),
            str(tmp_path / "out.jpg"),
            caption_text="test caption",
            caption_font_path=str(tmp_path / "missing_font.ttf"),
        )


def test_logs_contain_full_pipeline_path(tmp_path, caplog):
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

    logger = logging.getLogger("test.logs.full_path")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = True

    ctx = RunContext(
        generation_id="gen",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    runner = ChatRunner(ai_text=FakeTextAI())
    root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            Block(
                name="inner",
                merge="all_messages",
                nodes=[ChatStep(name="s1", prompt="p1", temperature=0.0)],
            )
        ],
    )

    with caplog.at_level(logging.INFO):
        runner.run(ctx, root)

    assert any("Step: pipeline/inner/s1" in record.getMessage() for record in caplog.records)


def test_logs_include_stage_provenance_tokens(tmp_path, caplog):
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

    logger = logging.getLogger("test.logs.provenance")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = True

    ctx = RunContext(
        generation_id="gen",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    class FakeTextAI:
        def text_chat(self, messages, **kwargs):
            return "resp"

    runner = ChatRunner(ai_text=FakeTextAI())
    root = Block(
        name="pipeline",
        merge="all_messages",
        nodes=[
            Block(
                name="inner",
                merge="all_messages",
                nodes=[
                    ChatStep(
                        name="s1",
                        prompt="p1",
                        temperature=0.0,
                        meta={"source": "test.source", "doc": "Test doc"},
                    )
                ],
            )
        ],
    )

    with caplog.at_level(logging.INFO):
        runner.run(ctx, root)

    start_line = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Step: pipeline/inner/s1")
    )
    assert "stage_id=inner" in start_line
    assert "source=test.source" in start_line
    assert 'doc="Test doc"' in start_line

    end_line = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Received response for pipeline/inner/s1")
    )
    assert "stage_id=inner" in end_line
    assert "source=test.source" in end_line
