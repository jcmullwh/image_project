import json
import logging
import random

from message_handling import MessageHandler
from pipeline import RunContext
from run_config import RunConfig
from transcript import write_transcript


def test_write_transcript_includes_experiment_metadata(tmp_path):
    cfg_dict = {
        "run": {"mode": "prompt_only"},
        "prompt": {
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
        },
        "image": {"log_path": str(tmp_path / "logs")},
        "experiment": {"id": "exp_unit", "variant": "A", "tags": ["tag1"]},
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.transcript.experiment")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen123",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=123,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )
    ctx.outputs["image_prompt"] = "Prompt"

    out_path = tmp_path / "transcript.json"
    write_transcript(str(out_path), ctx)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["experiment"]["id"] == "exp_unit"
    assert payload["experiment"]["variant"] == "A"
    assert payload["experiment"]["tags"] == ["tag1"]

