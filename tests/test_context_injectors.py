from __future__ import annotations

import json
import logging
import random
import re
from datetime import date

import pandas as pd
import pytest

from image_project.framework.context import ContextManager
from pipelinekit.engine.messages import MessageHandler
from image_project.framework.runtime import RunContext
from image_project.impl.current import context_plugins as _context_plugins
from image_project.prompts.preprompt import DEFAULT_SYSTEM_PROMPT
from image_project.prompts.standard import generate_first_prompt
from image_project.framework.config import RunConfig
from image_project.framework.artifacts import write_transcript

_context_plugins.discover()


CALENDAR_NOT_IMPLEMENTED = (
    "Calendar context injector is not implemented yet. Set context.calendar.enabled=false (default) "
    "or remove 'calendar' from context.injectors."
)


def _minimal_cfg(tmp_path, *, context: dict | None = None) -> dict:
    cfg = {
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
    if context is not None:
        cfg["context"] = context
    return cfg


def test_calendar_enabled_fails_loudly(tmp_path):
    cfg_dict = _minimal_cfg(
        tmp_path,
        context={
            "enabled": True,
            "calendar": {"enabled": True},
        },
    )

    with pytest.raises(ValueError, match=re.escape(CALENDAR_NOT_IMPLEMENTED)):
        RunConfig.from_dict(cfg_dict)


def test_context_disabled_is_no_op():
    guidance, metadata = ContextManager.build(
        enabled=False,
        injectors=("season", "holiday"),
        context_cfg={},
        seed=123,
        today=date(2025, 1, 1),
        preferences_guidance="",
        logger=None,
    )

    assert guidance == ""
    assert metadata == {}

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if guidance:
        system_prompt = DEFAULT_SYSTEM_PROMPT + "\n\n" + guidance
    assert system_prompt == DEFAULT_SYSTEM_PROMPT


def test_holiday_probability_is_deterministic():
    today = date(2025, 12, 20)
    cfg = {"holiday": {"lookahead_days": 14, "base_probability": 0.15, "max_probability": 0.55}}

    guidance_1, meta_1 = ContextManager.build(
        enabled=True,
        injectors=("holiday",),
        context_cfg=cfg,
        seed=999,
        today=today,
        preferences_guidance="",
        logger=None,
    )
    guidance_2, meta_2 = ContextManager.build(
        enabled=True,
        injectors=("holiday",),
        context_cfg=cfg,
        seed=999,
        today=today,
        preferences_guidance="",
        logger=None,
    )

    assert guidance_1 == guidance_2
    assert meta_1 == meta_2
    assert meta_1["holiday"]["date_used"] == today.isoformat()


def test_context_injection_does_not_change_concept_selection():
    seed = 123
    today = date(2025, 12, 20)

    prompt_data = pd.DataFrame(
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
    user_profile = pd.DataFrame({"Likes": ["colorful"], "Dislikes": ["no horror"]})

    rng_a = random.Random(seed)
    _prompt_a, concepts_a = generate_first_prompt(prompt_data, user_profile, rng_a)

    guidance, _meta = ContextManager.build(
        enabled=True,
        injectors=("holiday", "season"),
        context_cfg={"holiday": {"lookahead_days": 14}},
        seed=seed,
        today=today,
        preferences_guidance="Dislikes:\n- horror\n",
        logger=None,
    )

    rng_b = random.Random(seed)
    prompt_b, concepts_b = generate_first_prompt(
        prompt_data, user_profile, rng_b, context_guidance=guidance
    )

    assert concepts_a == concepts_b
    assert "Context guidance" in prompt_b


def test_transcript_includes_context_when_present(tmp_path):
    cfg_dict = _minimal_cfg(tmp_path)
    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    logger = logging.getLogger("test.context.transcript")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    ctx = RunContext(
        generation_id="gen_ctx",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )
    ctx.outputs["context"] = {"season": {"season": "winter"}}

    out_path = tmp_path / "transcript.json"
    write_transcript(str(out_path), ctx)

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["context"] == {"season": {"season": "winter"}}
