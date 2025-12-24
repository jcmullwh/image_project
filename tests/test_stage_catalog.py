import logging
import random

import pandas as pd
import pytest

import prompts
from message_handling import MessageHandler
from pipeline import RunContext
from prompt_plans import PlanInputs, StageSpec, StandardPromptPlan
from run_config import RunConfig
from stage_catalog import StageCatalog


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
        user_profile=pd.DataFrame({"Likes": ["x"], "Dislikes": [None]}),
        preferences_guidance="Likes:\n- x",
        context_guidance=None,
        rng=random.Random(0),
        draft_prompt="Draft prompt",
    )


def test_stage_catalog_rejects_duplicate_stage_ids():
    with pytest.raises(ValueError, match=r"Duplicate stage id: standard\.initial_prompt"):

        @StageCatalog.register("standard.initial_prompt")
        def dup(_inputs: PlanInputs) -> StageSpec:  # pragma: no cover
            return StageSpec(stage_id="standard.initial_prompt", prompt="x", temperature=0.0)


def test_stage_catalog_unknown_stage_fails_fast(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)
    with pytest.raises(ValueError, match=r"Unknown stage id: does_not_exist"):
        StageCatalog.build("does_not_exist", inputs)


def test_stage_catalog_suffix_lookup_resolves_unambiguous_ids(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    spec = StageCatalog.build("initial_prompt", inputs)
    assert spec.stage_id == "standard.initial_prompt"
    assert spec.doc == "Generate candidate themes/stories."
    assert spec.source == "prompts.generate_first_prompt"
    assert "standard" in spec.tags


def test_standard_plan_stage_specs_come_from_catalog(tmp_path):
    cfg, _warnings = RunConfig.from_dict(_base_cfg_dict(tmp_path))
    inputs = _make_inputs(cfg)

    plan = StandardPromptPlan()
    stage_specs = plan.stage_specs(inputs)
    assert stage_specs[0].stage_id == "preprompt.select_concepts"
    assert stage_specs[1].stage_id == "preprompt.filter_concepts"
    assert stage_specs[2].stage_id == "standard.initial_prompt"
    assert stage_specs[-1].stage_id == "standard.image_prompt_creation"
    assert stage_specs[-1].source == "prompts.generate_image_prompt"


def _make_ctx(cfg: RunConfig, logger_name: str) -> RunContext:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return RunContext(
        generation_id="gen_stage_catalog",
        cfg=cfg,
        logger=logger,
        rng=random.Random(0),
        seed=0,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [("raw", "RAW"), ("generator_hints", "HINTS")],
)
def test_blackbox_judge_profile_source_routing(tmp_path, monkeypatch, source, expected):
    def fake_prompt(*, raw_profile: str, **_kwargs) -> str:
        return f"PROFILE={raw_profile}"

    monkeypatch.setattr(prompts, "idea_cards_judge_prompt", fake_prompt)

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["scoring"] = {"judge_profile_source": source}
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    spec = StageCatalog.build("blackbox.idea_cards_judge_score", inputs)
    assert isinstance(spec, StageSpec)

    ctx = _make_ctx(cfg, "test.blackbox.judge_profile_source")
    ctx.outputs["preferences_guidance"] = "RAW"
    ctx.outputs["generator_profile_hints"] = "HINTS"
    ctx.outputs["idea_cards_json"] = "{}"

    assert spec.prompt(ctx) == f"PROFILE={expected}"


def test_blackbox_judge_profile_source_requires_generator_hints(tmp_path, monkeypatch):
    def fake_prompt(*, raw_profile: str, **_kwargs) -> str:
        return f"PROFILE={raw_profile}"

    monkeypatch.setattr(prompts, "idea_cards_judge_prompt", fake_prompt)

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["scoring"] = {"judge_profile_source": "generator_hints"}
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    spec = StageCatalog.build("blackbox.idea_cards_judge_score", inputs)
    assert isinstance(spec, StageSpec)

    ctx = _make_ctx(cfg, "test.blackbox.judge_profile_source_missing")
    ctx.outputs["preferences_guidance"] = "RAW"
    ctx.outputs["idea_cards_json"] = "{}"

    with pytest.raises(ValueError, match=r"blackbox\.idea_cards_judge_score requires generator_profile_hints"):
        spec.prompt(ctx)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("raw", "RAW"), ("generator_hints", "HINTS")],
)
def test_blackbox_final_profile_source_routing(tmp_path, monkeypatch, source, expected):
    def fake_prompt(*, raw_profile: str, **_kwargs) -> str:
        return f"PROFILE={raw_profile}"

    monkeypatch.setattr(prompts, "final_prompt_from_selected_idea_prompt", fake_prompt)

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["scoring"] = {"final_profile_source": source}
    cfg, _warnings = RunConfig.from_dict(cfg_dict)
    inputs = _make_inputs(cfg)

    spec = StageCatalog.build("blackbox.image_prompt_creation", inputs)
    assert isinstance(spec, StageSpec)

    ctx = _make_ctx(cfg, "test.blackbox.final_profile_source")
    ctx.outputs["preferences_guidance"] = "RAW"
    ctx.outputs["generator_profile_hints"] = "HINTS"
    ctx.outputs["selected_idea_card"] = {"id": "A"}

    assert spec.prompt(ctx) == f"PROFILE={expected}"
