import json
import logging
import random

import pandas as pd
import pytest

from image_project.app import generate as app_generate
from image_project.foundation.messages import MessageHandler
from image_project.framework.blackbox_refine_loop import _aggregate_scores, build_blackbox_refine_loop_specs
from image_project.framework.config import PromptBlackboxRefineJudgeConfig, RunConfig
from image_project.framework.prompting import PlanInputs
from image_project.framework.runtime import RunContext
from image_project.impl.current import prompting as prompts


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _base_cfg_dict(tmp_path) -> dict:
    return {
        "run": {"mode": "prompt_only"},
        "image": {"log_path": str(tmp_path / "logs")},
        "prompt": {
            "categories_path": str(tmp_path / "categories.csv"),
            "profile_path": str(tmp_path / "profile.csv"),
            "concepts": {"filters": {"enabled": False}},
            "scoring": {
                "enabled": True,
                "exploration_rate": 0.0,
                "judge_temperature": 0.0,
                "generator_profile_abstraction": False,
                "novelty": {"enabled": False, "window": 0},
            },
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }


def test_directive_determinism(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "categories.csv").write_text(
        "Subject Matter,Narrative,Mood,Composition,Perspective,Style,Time Period_Context,Color Scheme\n"
        "Cat,Quest,Moody,Wide,Top-down,Baroque,Renaissance,Vibrant\n",
        encoding="utf-8",
    )
    (tmp_path / "profile.csv").write_text("Likes,Dislikes\ncolorful,boring\n", encoding="utf-8")

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
            "include_mutation_directive": True,
            "include_scoring_rubric": False,
        },
        "mutation_directives": {
            "mode": "random",
            "directives": ["D1", "D2", "D3"],
        },
        "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
    }

    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        preferences_guidance="Likes:\n- colorful",
        context_guidance=None,
        rng=random.Random(0),
        draft_prompt="seed",
    )

    def build_map() -> dict[str, str]:
        loop_specs = build_blackbox_refine_loop_specs(inputs, seed_output_key="bbref.seed_prompt")
        init_stage = loop_specs[0]
        assert init_stage.stage_id == "blackbox_refine.init_state"

        ctx = RunContext(
            generation_id="unit_test",
            cfg=cfg,
            logger=_make_logger("test.bbref.directive"),
            rng=random.Random(0),
            seed=123,
            created_at="2025-01-01T00:00:00Z",
            messages=MessageHandler("system"),
            selected_concepts=["X"],
        )
        ctx.outputs["bbref.seed_prompt"] = "SEED"
        init_stage.fn(ctx)  # type: ignore[attr-defined]

        mapping: dict[str, str] = {}
        for spec in loop_specs:
            if getattr(spec, "stage_id", "").startswith("blackbox_refine.iter_01") and "cand_" in getattr(
                spec, "stage_id", ""
            ):
                prompt_text = spec.prompt(ctx)  # type: ignore[attr-defined]
                directive = "<missing>"
                lines = prompt_text.splitlines()
                for idx, line in enumerate(lines):
                    if line.strip() == "Mutation directive:" and idx + 1 < len(lines):
                        directive = lines[idx + 1].strip()
                        break
                mapping[spec.stage_id] = directive  # type: ignore[attr-defined]
        return mapping

    assert build_map() == build_map()


def test_judge_aggregation_methods():
    judges = (
        PromptBlackboxRefineJudgeConfig(
            id="j1",
            model=None,
            temperature=None,
            weight=1.0,
            rubric="default",  # type: ignore[arg-type]
        ),
        PromptBlackboxRefineJudgeConfig(
            id="j2",
            model=None,
            temperature=None,
            weight=1.0,
            rubric="default",  # type: ignore[arg-type]
        ),
    )
    candidates = ["A", "B"]
    judge_scores_by_id = {"j1": {"A": 0, "B": 100}, "j2": {"A": 100, "B": 0}}

    assert _aggregate_scores(
        candidates=candidates,
        judges=judges,
        judge_scores_by_id=judge_scores_by_id,
        method="mean",
        trimmed_mean_drop=0,
    ) == {"A": 50.0, "B": 50.0}

    assert _aggregate_scores(
        candidates=candidates,
        judges=judges,
        judge_scores_by_id=judge_scores_by_id,
        method="median",
        trimmed_mean_drop=0,
    ) == {"A": 50.0, "B": 50.0}

    assert _aggregate_scores(
        candidates=candidates,
        judges=judges,
        judge_scores_by_id=judge_scores_by_id,
        method="min",
        trimmed_mean_drop=0,
    ) == {"A": 0.0, "B": 0.0}

    assert _aggregate_scores(
        candidates=candidates,
        judges=judges,
        judge_scores_by_id=judge_scores_by_id,
        method="max",
        trimmed_mean_drop=0,
    ) == {"A": 100.0, "B": 100.0}


def test_novelty_penalty_applied_in_selection(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "categories.csv").write_text(
        "Subject Matter,Narrative,Mood,Composition,Perspective,Style,Time Period_Context,Color Scheme\n"
        "Cat,Quest,Moody,Wide,Top-down,Baroque,Renaissance,Vibrant\n",
        encoding="utf-8",
    )
    (tmp_path / "profile.csv").write_text("Likes,Dislikes\ncolorful,boring\n", encoding="utf-8")

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["scoring"]["novelty"] = {"enabled": True, "window": 25, "method": "df_overlap_v1"}
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
            "include_context_guidance": False,
            "include_novelty_summary": False,
            "include_mutation_directive": False,
            "include_scoring_rubric": False,
        },
        "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
    }

    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        preferences_guidance="Likes:\n- colorful",
        context_guidance=None,
        rng=random.Random(0),
        draft_prompt="seed",
    )
    loop_specs = build_blackbox_refine_loop_specs(inputs, seed_output_key="bbref.seed_prompt")

    init_stage = loop_specs[0]
    select_stage = next(
        spec for spec in loop_specs if getattr(spec, "stage_id", "") == "blackbox_refine.iter_01.select"
    )

    ctx = RunContext(
        generation_id="unit_test",
        cfg=cfg,
        logger=_make_logger("test.bbref.novelty"),
        rng=random.Random(0),
        seed=123,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
        selected_concepts=["X"],
    )
    ctx.outputs["bbref.seed_prompt"] = "SEED"
    ctx.blackbox_scoring = {
        "enabled": True,
        "config_snapshot": {},
        "novelty_summary": {
            "enabled": True,
            "method": "df_overlap_v1",
            "motifs": [
                {"token": "ocean", "df": 10, "w": 10},
                {"token": "moon", "df": 4, "w": 4},
                {"token": "sunset", "df": 3, "w": 3},
            ],
            "total_weight": 17,
        },
    }
    init_stage.fn(ctx)  # type: ignore[attr-defined]

    ctx.outputs["bbref.iter_01.beam_01.cand_A.prompt"] = "A prompt with sunset"
    ctx.outputs["bbref.iter_01.beam_01.cand_B.prompt"] = "A prompt without repetition"
    ctx.outputs["bbref.iter_01.judge_j1.scores_json"] = json.dumps(
        {"scores": [{"id": "A", "score": 10}, {"id": "B", "score": 10}]}
    )

    select_stage.fn(ctx)  # type: ignore[attr-defined]
    assert ctx.outputs["bbref.beams"][0]["candidate_id"] == "B"

    iteration_log = ctx.blackbox_scoring["prompt_refine"]["iterations"][0]
    score_table = iteration_log["selection"]["score_table"]
    penalty_a = next(row["novelty_penalty"] for row in score_table if row["id"] == "A")
    penalty_b = next(row["novelty_penalty"] for row in score_table if row["id"] == "B")
    assert penalty_a > penalty_b
    assert penalty_a < 20

    row_a = next(row for row in score_table if row["id"] == "A")
    assert row_a["novelty_detail"]["top_motifs"][0]["token"] == "sunset"
    assert iteration_log["selection"]["novelty"]["method"] == "df_overlap_v1"


def test_best_worst_score_feedback_is_injected_into_next_iteration_generation_prompt(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "categories.csv").write_text(
        "Subject Matter,Narrative,Mood,Composition,Perspective,Style,Time Period_Context,Color Scheme\n"
        "Cat,Quest,Moody,Wide,Top-down,Baroque,Renaissance,Vibrant\n",
        encoding="utf-8",
    )
    (tmp_path / "profile.csv").write_text("Likes,Dislikes\ncolorful,boring\n", encoding="utf-8")

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["blackbox_refine"] = {
        "enabled": True,
        "iterations": 2,
        "algorithm": "hillclimb",
        "branching_factor": 2,
        "include_parents_as_candidates": False,
        "generator_temperature": 0.9,
        "variation_prompt": {
            "template": "v1",
            "include_profile": False,
            "include_context_guidance": False,
            "include_novelty_summary": False,
            "include_mutation_directive": False,
            "include_scoring_rubric": False,
            "score_feedback": "best_worst",
            "score_feedback_max_chars": 2000,
        },
        "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
    }
    cfg_dict["prompt"]["scoring"].update({"exploration_rate": 0.0, "novelty": {"enabled": False, "window": 0}})

    cfg, _warnings = RunConfig.from_dict(cfg_dict)

    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        preferences_guidance="Likes:\n- colorful",
        context_guidance=None,
        rng=random.Random(0),
        draft_prompt="seed",
    )

    loop_specs = build_blackbox_refine_loop_specs(inputs, seed_output_key="bbref.seed_prompt")
    init_stage = loop_specs[0]
    select_stage = next(
        spec for spec in loop_specs if getattr(spec, "stage_id", "") == "blackbox_refine.iter_01.select"
    )
    gen_stage_iter2 = next(
        spec
        for spec in loop_specs
        if getattr(spec, "stage_id", "") == "blackbox_refine.iter_02.beam_01.cand_A"
    )

    ctx = RunContext(
        generation_id="unit_test",
        cfg=cfg,
        logger=_make_logger("test.bbref.feedback"),
        rng=random.Random(0),
        seed=123,
        created_at="2025-01-01T00:00:00Z",
        messages=MessageHandler("system"),
        selected_concepts=["X"],
    )
    ctx.outputs["bbref.seed_prompt"] = "SEED"
    init_stage.fn(ctx)  # type: ignore[attr-defined]

    ctx.outputs["bbref.iter_01.beam_01.cand_A.prompt"] = "PROMPT_A"
    ctx.outputs["bbref.iter_01.beam_01.cand_B.prompt"] = "PROMPT_B"
    ctx.outputs["bbref.iter_01.judge_j1.scores_json"] = json.dumps(
        {"scores": [{"id": "A", "score": 10}, {"id": "B", "score": 90}]}
    )

    select_stage.fn(ctx)  # type: ignore[attr-defined]

    prompt_text = gen_stage_iter2.prompt(ctx)  # type: ignore[attr-defined]
    assert "SCORE FEEDBACK EXAMPLES" in prompt_text
    assert "BEST" in prompt_text and "WORST" in prompt_text
    assert "PROMPT_A" in prompt_text
    assert "PROMPT_B" in prompt_text
    assert "raw=90.0" in prompt_text
    assert "raw=10.0" in prompt_text


def test_blackbox_refine_only_end_to_end_prompt_only(tmp_path, monkeypatch):
    categories_path = tmp_path / "categories.csv"
    categories_path.write_text(
        "Subject Matter,Narrative,Mood,Composition,Perspective,Style,Time Period_Context,Color Scheme\n"
        "Cat,Quest,Moody,Wide,Top-down,Baroque,Renaissance,Vibrant\n",
        encoding="utf-8",
    )

    profile_path = tmp_path / "profile.csv"
    profile_path.write_text("Likes,Dislikes\ncolorful,boring\n", encoding="utf-8")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "blackbox_refine_only"
    cfg_dict["prompt"]["refine_only"] = {"draft": "DRAFT_SEED"}
    cfg_dict["prompt"]["blackbox_refine"] = {
        "enabled": True,
        "iterations": 2,
        "algorithm": "hillclimb",
        "branching_factor": 2,
        "include_parents_as_candidates": False,
        "generator_temperature": 0.9,
        "variation_prompt": {"template": "v1", "include_profile": False},
        "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
    }

    generation_id = "unit_test_bbref_only"

    generation_outputs = ["PROMPT_A1", "PROMPT_B1", "PROMPT_A2", "PROMPT_B2"]
    judge_json = json.dumps({"scores": [{"id": "A", "score": 0}, {"id": "B", "score": 100}]})

    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            self.model = "fake"
            self.calls: list[dict[str, object]] = []
            self._gen_idx = 0

        def text_chat(self, messages, **kwargs):
            self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
            last_user = (messages[-1].get("content", "") if messages else "") or ""
            if "You are a strict numeric judge" in last_user:
                return judge_json
            if "Generate one improved image prompt variant" in last_user:
                out = generation_outputs[self._gen_idx]
                self._gen_idx += 1
                return out
            return "resp"

    fake_text = FakeTextAI()
    monkeypatch.setattr(app_generate, "TextAI", lambda *args, **kwargs: fake_text)

    app_generate.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = logs_dir / f"{generation_id}_transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["final_image_prompt"] == "PROMPT_B2"
    assert transcript["blackbox_scoring"]["prompt_refine"]["seed_prompt"] == "DRAFT_SEED"
    assert len(transcript["blackbox_scoring"]["prompt_refine"]["iterations"]) == 2
    assert transcript["outputs"]["prompt_pipeline"]["plan"] == "blackbox_refine_only"
    assert transcript["outputs"]["prompt_pipeline"]["capture_stage"] == "blackbox_refine.finalize"

    generation_calls = [
        call
        for call in fake_text.calls
        if isinstance(call.get("messages"), list)
        and call["messages"]
        and "Generate one improved image prompt variant" in (call["messages"][-1].get("content") or "")
    ]
    assert generation_calls
    assert all(
        all((msg.get("role") or "") != "assistant" for msg in call["messages"]) for call in generation_calls
    )


def test_blackbox_refine_seed_from_blackbox(tmp_path, monkeypatch):
    idea_sentinel_a = "__TEST_IDEA_CARD_A__"
    idea_sentinel_b = "__TEST_IDEA_CARD_B__"
    judge_sentinel = "__TEST_IDEA_JUDGE__"
    final_sentinel = "__TEST_SEED_PROMPT__"

    num_ideas = 2
    expected_ids = ["A", "B"]

    idea_prompt_by_id = {expected_ids[0]: idea_sentinel_a, expected_ids[1]: idea_sentinel_b}
    idea_output_by_prompt = {
        idea_sentinel_a: json.dumps(
            {
                "id": expected_ids[0],
                "hook": "Hook A.",
                "narrative": "Narrative A.",
                "options": {
                    "composition": ["c1", "c2"],
                    "palette": ["p1", "p2"],
                    "medium": ["m1"],
                    "mood": ["mo1"],
                },
            }
        ),
        idea_sentinel_b: json.dumps(
            {
                "id": expected_ids[1],
                "hook": "Hook B.",
                "narrative": "Narrative B.",
                "options": {
                    "composition": ["c1", "c2"],
                    "palette": ["p1", "p2"],
                    "medium": ["m1"],
                    "mood": ["mo1"],
                },
            }
        ),
    }
    idea_judge_output = json.dumps(
        {"scores": [{"id": expected_ids[0], "score": 0}, {"id": expected_ids[1], "score": 100}]}
    )

    categories_path = tmp_path / "categories.csv"
    categories_path.write_text(
        "Subject Matter,Narrative,Mood,Composition,Perspective,Style,Time Period_Context,Color Scheme\n"
        "Cat,Quest,Moody,Wide,Top-down,Baroque,Renaissance,Vibrant\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.csv"
    profile_path.write_text("Likes,Dislikes\ncolorful,boring\n", encoding="utf-8")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"].update(
        {
            "plan": "blackbox_refine",
            "scoring": {
                "enabled": True,
                "num_ideas": num_ideas,
                "exploration_rate": 0.0,
                "judge_temperature": 0.0,
                "generator_profile_abstraction": False,
                "novelty": {"enabled": False, "window": 0},
            },
            "blackbox_refine": {
                "enabled": True,
                "iterations": 1,
                "algorithm": "hillclimb",
                "branching_factor": 2,
                "include_parents_as_candidates": False,
                "generator_temperature": 0.9,
                "variation_prompt": {"template": "v1", "include_profile": False},
                "judging": {"judges": [{"id": "j1"}], "aggregation": "mean"},
            },
        }
    )

    def fake_idea_card_generate_prompt(*, idea_id: str, **_kwargs):
        return idea_prompt_by_id[idea_id]

    def fake_idea_cards_judge_prompt(**_kwargs):
        return judge_sentinel

    def fake_final_prompt_from_selected_idea_prompt(**_kwargs):
        return final_sentinel

    generation_id = "unit_test_bbref_blackbox"

    judge_json = json.dumps({"scores": [{"id": "A", "score": 0}, {"id": "B", "score": 100}]})

    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            self.model = "fake"
            self.calls = 0

        def text_chat(self, messages, **kwargs):
            last_user = (messages[-1].get("content", "") if messages else "") or ""
            if last_user in idea_output_by_prompt:
                return idea_output_by_prompt[last_user]
            if last_user == judge_sentinel:
                return idea_judge_output
            if last_user == final_sentinel:
                return "SEED_FROM_BLACKBOX"
            if "You are a strict numeric judge" in last_user:
                return judge_json
            if "Generate one improved image prompt variant" in last_user:
                self.calls += 1
                return "REFINED_B" if self.calls == 2 else "REFINED_A"
            return "resp"

    monkeypatch.setattr(app_generate, "TextAI", FakeTextAI)
    monkeypatch.setattr(prompts, "idea_card_generate_prompt", fake_idea_card_generate_prompt)
    monkeypatch.setattr(prompts, "idea_cards_judge_prompt", fake_idea_cards_judge_prompt)
    monkeypatch.setattr(prompts, "final_prompt_from_selected_idea_prompt", fake_final_prompt_from_selected_idea_prompt)

    app_generate.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = logs_dir / f"{generation_id}_transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["final_image_prompt"] == "REFINED_B"
    assert transcript["blackbox_scoring"]["prompt_refine"]["seed_prompt"] == "SEED_FROM_BLACKBOX"
    assert transcript["outputs"]["prompt_pipeline"]["plan"] == "blackbox_refine"
    assert transcript["outputs"]["prompt_pipeline"]["capture_stage"] == "blackbox_refine.finalize"
