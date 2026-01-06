import base64
import csv
import io
import json
import logging
import random
import types
from unittest.mock import Mock

import pandas as pd
import pytest
from PIL import Image

import main
from message_handling import MessageHandler
from pipeline import ChatRunner, ChatStep, RunContext
from records import append_generation_row
from run_config import RunConfig
from transcript import write_transcript


def test_config_validation_missing_prompt_categories_path_raises():
    cfg_dict = {
        "prompt": {
            "profile_path": "x.csv",
            "generations_path": "g.csv",
        },
        "image": {
            "generation_path": "out",
            "upscale_path": "up",
            "log_path": "logs",
        },
    }

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    assert "prompt.categories_path" in str(excinfo.value)


def test_config_validation_missing_image_generation_path_raises():
    cfg_dict = {
        "prompt": {
            "categories_path": "x.csv",
            "profile_path": "p.csv",
            "generations_path": "g.csv",
        },
        "image": {
            "upscale_path": "up",
            "log_path": "logs",
        },
    }

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    assert "image.generation_path" in str(excinfo.value)


def test_config_validation_empty_strings_are_missing():
    cfg_dict = {
        "prompt": {
            "categories_path": "   ",
            "profile_path": "p.csv",
            "generations_path": "g.csv",
        },
        "image": {
            "generation_path": "out",
            "upscale_path": "up",
            "log_path": "logs",
        },
    }

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    assert "prompt.categories_path" in str(excinfo.value)


def test_config_validation_requires_both_upscale_dimensions(tmp_path):
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
        "upscale": {"enabled": True, "target_width_px": 2000},
    }

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    assert "target_width_px" in str(excinfo.value)


def test_config_validation_rejects_conflicting_upscale_size_and_aspect(tmp_path):
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
        "upscale": {
            "enabled": True,
            "target_width_px": 2000,
            "target_height_px": 1200,
            "target_aspect_ratio": "16:9",
        },
    }

    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_dict(cfg_dict)

    assert "target_width_px/target_height_px" in str(excinfo.value)


def test_config_parses_target_aspect_ratio(tmp_path):
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
        "upscale": {"enabled": True, "target_aspect_ratio": "21:9"},
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    assert pytest.approx(cfg.upscale_target_aspect_ratio, rel=1e-6) == 21 / 9


def test_seeded_randomness_is_deterministic_for_selected_concepts():
    categories = pd.DataFrame(
        {
            "Subject Matter": ["Cat", "Dog"],
            "Narrative": ["Quest", "Heist"],
            "Mood": ["Moody", "Joyful"],
            "Composition": ["Wide", "Closeup"],
            "Perspective": ["Top-down", "Eye-level"],
            "Style": ["Baroque", "Minimalist"],
            "Time Period_Context": ["Renaissance", "Futuristic"],
            "Color Scheme": ["Vibrant", "Monochrome"],
        }
    )
    profile = pd.DataFrame(
        {
            "Likes": ["colorful", None],
            "Dislikes": ["boring", "cliche"],
        }
    )

    rng1 = random.Random(123)
    rng2 = random.Random(123)

    _, selected1 = main.generate_first_prompt(categories, profile, rng1)
    _, selected2 = main.generate_first_prompt(categories, profile, rng2)

    assert selected1 == selected2


def test_pipeline_step_execution_order_and_capture(tmp_path):
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

    logger = logging.getLogger("test.pipeline")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    mock_ai = Mock()
    mock_ai.text_chat.side_effect = ["resp:a", "resp:b", "resp:c"]

    ctx = RunContext(
        generation_id="gen1",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    steps = [
        ChatStep(name="a", prompt="p1", temperature=0.8),
        ChatStep(name="b", prompt="p2", temperature=0.8),
        ChatStep(name="c", prompt="p3", temperature=0.8, capture_key="dalle_prompt"),
    ]

    runner = ChatRunner(ai_text=mock_ai)
    runner.run_steps(ctx, steps)

    assert [s["name"] for s in ctx.steps] == ["a", "b", "c"]
    assert ctx.outputs["dalle_prompt"] == "resp:c"


def test_pipeline_prompt_factory_none_raises(tmp_path):
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
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.pipeline.prompt.none")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen_prompt_none",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    mock_ai = Mock()
    runner = ChatRunner(ai_text=mock_ai)

    steps = [ChatStep(name="bad", prompt=lambda _ctx: None, temperature=0.8)]
    with pytest.raises(ValueError, match="Step bad produced None prompt"):
        runner.run_steps(ctx, steps)

    mock_ai.text_chat.assert_not_called()


def test_pipeline_prompt_factory_whitespace_raises(tmp_path):
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
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.pipeline.prompt.empty")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen_prompt_empty",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    mock_ai = Mock()
    runner = ChatRunner(ai_text=mock_ai)

    steps = [ChatStep(name="bad", prompt=lambda _ctx: "   ", temperature=0.8)]
    with pytest.raises(ValueError, match="Step bad produced empty prompt"):
        runner.run_steps(ctx, steps)

    mock_ai.text_chat.assert_not_called()


def test_pipeline_prompt_factory_non_string_raises(tmp_path):
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
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.pipeline.prompt.type")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen_prompt_type",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    mock_ai = Mock()
    runner = ChatRunner(ai_text=mock_ai)

    steps = [ChatStep(name="bad", prompt=lambda _ctx: 123, temperature=0.8)]
    with pytest.raises(TypeError, match="Step bad produced non-string prompt"):
        runner.run_steps(ctx, steps)

    mock_ai.text_chat.assert_not_called()


def test_pipeline_response_none_raises(tmp_path):
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
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.pipeline.response.none")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen_resp_none",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    mock_ai = Mock()
    mock_ai.text_chat.return_value = None
    runner = ChatRunner(ai_text=mock_ai)

    steps = [ChatStep(name="a", prompt="hello", temperature=0.8)]
    with pytest.raises(ValueError, match="Step a produced None response"):
        runner.run_steps(ctx, steps)


def test_pipeline_response_empty_raises(tmp_path):
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
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.pipeline.response.empty")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen_resp_empty",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )

    mock_ai = Mock()
    mock_ai.text_chat.return_value = ""
    runner = ChatRunner(ai_text=mock_ai)

    steps = [ChatStep(name="a", prompt="hello", temperature=0.8)]
    with pytest.raises(ValueError, match="Step a produced empty response"):
        runner.run_steps(ctx, steps)


def test_transcript_json_is_valid(tmp_path):
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
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.transcript")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen2",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=42,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )
    ctx.selected_concepts = ["A", "B"]
    ctx.outputs["concept_filter_log"] = {
        "input": ["A", "B"],
        "output": ["A", "B"],
        "filters": [],
    }
    ctx.steps.append(
        {
            "name": "step1",
            "prompt": "hello",
            "response": "world",
            "params": {"temperature": 0.8},
            "created_at": "2025-01-01T00:00:01Z",
        }
    )
    ctx.image_path = "C:\\tmp\\image.jpg"

    out_path = tmp_path / "transcript.json"
    write_transcript(str(out_path), ctx)

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ["generation_id", "seed", "selected_concepts", "steps", "image_path", "created_at"]:
        assert key in loaded
    assert loaded["concept_filter_log"]["input"] == ["A", "B"]


def test_csv_writer_creates_header_and_appends_row(tmp_path):
    csv_path = str(tmp_path / "gens.csv")
    fieldnames = ["generation_id", "image_path"]

    append_generation_row(csv_path, {"generation_id": "g1", "image_path": "p1"}, fieldnames)
    append_generation_row(csv_path, {"generation_id": "g2", "image_path": "p2"}, fieldnames)

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["generation_id"] == "g1"
    assert rows[1]["image_path"] == "p2"


def test_csv_writer_raises_when_required_keys_missing(tmp_path):
    csv_path = str(tmp_path / "gens.csv")
    with pytest.raises(KeyError):
        append_generation_row(csv_path, {"generation_id": "g1"}, ["generation_id", "image_path"])


def test_integration_offline_run_generation_writes_artifacts(tmp_path, monkeypatch):
    image_prompt_request = "__TEST_IMAGE_PROMPT_REQUEST__"

    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            pass

        def text_chat(self, messages, **kwargs):
            last_user = (messages[-1].get("content", "") if messages else "") or ""
            if last_user == image_prompt_request:
                return "A test image prompt"
            return "resp"

    class FakeImageAI:
        def __init__(self, *args, **kwargs):
            pass

        def generate_image(self, prompt, **kwargs):
            im = Image.new("RGB", (4, 4), color=(255, 0, 0))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"image": b64, "seed": "test-seed"}

    categories_path = tmp_path / "categories.csv"
    profile_path = tmp_path / "profile.csv"

    categories = pd.DataFrame(
        {
            "Subject Matter": ["Cat", "Dog"],
            "Narrative": ["Quest", "Heist"],
            "Mood": ["Moody", "Joyful"],
            "Composition": ["Wide", "Closeup"],
            "Perspective": ["Top-down", "Eye-level"],
            "Style": ["Baroque", "Minimalist"],
            "Time Period_Context": ["Renaissance", "Futuristic"],
            "Color Scheme": ["Vibrant", "Monochrome"],
        }
    )
    categories.to_csv(categories_path, index=False)

    profile = pd.DataFrame({"Likes": ["colorful"], "Dislikes": ["boring"]})
    profile.to_csv(profile_path, index=False)

    generation_dir = tmp_path / "generated"
    upscale_dir = tmp_path / "upscaled"
    log_dir = tmp_path / "logs"
    generations_csv = tmp_path / "generations.csv"

    cfg_dict = {
        "prompt": {
            "categories_path": str(categories_path),
            "profile_path": str(profile_path),
            "generations_path": str(generations_csv),
            "random_seed": 123,
        },
        "image": {
            "generation_path": str(generation_dir),
            "upscale_path": str(upscale_dir),
            "log_path": str(log_dir),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    generation_id = "unit_test_generation"

    monkeypatch.setattr(main, "TextAI", FakeTextAI)
    monkeypatch.setattr(main, "ImageAI", FakeImageAI)
    monkeypatch.setattr(main, "generate_image_prompt", lambda: image_prompt_request)
    monkeypatch.setattr(
        main,
        "generate_title",
        lambda **_kwargs: types.SimpleNamespace(
            title="Test Title", title_source="test", title_raw="Test Title"
        ),
    )

    main.run_generation(cfg_dict, generation_id=generation_id)

    image_path = generation_dir / f"{generation_id}_image.jpg"
    transcript_path = log_dir / f"{generation_id}_transcript.json"

    assert image_path.exists()
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["generation_id"] == generation_id

    with open(generations_csv, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows
    assert rows[-1]["generation_id"] == generation_id
    assert rows[-1]["image_path"]


def test_transcript_written_on_pipeline_failure(tmp_path, monkeypatch):
    fail_sentinel = "__TEST_FAIL_STEP__"

    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            pass

        def text_chat(self, messages, **kwargs):
            last_user = (messages[-1].get("content", "") if messages else "") or ""
            if last_user == fail_sentinel:
                raise RuntimeError("boom")
            return "resp"

    categories_path = tmp_path / "categories.csv"
    profile_path = tmp_path / "profile.csv"

    categories = pd.DataFrame(
        {
            "Subject Matter": ["Cat", "Dog"],
            "Narrative": ["Quest", "Heist"],
            "Mood": ["Moody", "Joyful"],
            "Composition": ["Wide", "Closeup"],
            "Perspective": ["Top-down", "Eye-level"],
            "Style": ["Baroque", "Minimalist"],
            "Time Period_Context": ["Renaissance", "Futuristic"],
            "Color Scheme": ["Vibrant", "Monochrome"],
        }
    )
    categories.to_csv(categories_path, index=False)

    profile = pd.DataFrame({"Likes": ["colorful"], "Dislikes": ["boring"]})
    profile.to_csv(profile_path, index=False)

    generation_dir = tmp_path / "generated"
    upscale_dir = tmp_path / "upscaled"
    log_dir = tmp_path / "logs"
    generations_csv = tmp_path / "generations.csv"

    cfg_dict = {
        "prompt": {
            "categories_path": str(categories_path),
            "profile_path": str(profile_path),
            "generations_path": str(generations_csv),
            "random_seed": 123,
        },
        "image": {
            "generation_path": str(generation_dir),
            "upscale_path": str(upscale_dir),
            "log_path": str(log_dir),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    generation_id = "unit_test_failure"

    monkeypatch.setattr(main, "TextAI", FakeTextAI)
    monkeypatch.setattr(main, "generate_second_prompt", lambda: fail_sentinel)

    with pytest.raises(RuntimeError, match="boom"):
        main.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = log_dir / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["generation_id"] == generation_id
    assert "error" in transcript
    assert transcript["error"]["type"] == "RuntimeError"
    assert transcript["error"]["phase"] == "pipeline"
    assert transcript["error"]["step"] == "draft"
    assert transcript["error"]["path"] == "pipeline/section_2_choice/draft"

    recorded_step_paths = [step.get("path") for step in transcript.get("steps", [])]
    assert "pipeline/initial_prompt/draft" in recorded_step_paths
    assert "pipeline/section_2_choice/draft" not in recorded_step_paths
