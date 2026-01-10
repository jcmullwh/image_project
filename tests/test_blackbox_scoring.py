import base64
import csv
import io
import json
import random
import types

import pandas as pd
import pytest
from PIL import Image

from image_project.app import generate as app_generate
from image_project.framework.config import PromptNoveltyConfig
from image_project.framework import scoring as blackbox_scoring
from image_project.impl.current import prompting as prompts


def test_parse_idea_cards_json_valid():
    payload = {
        "ideas": [
            {
                "id": "A",
                "hook": "Hook A.",
                "narrative": "Narrative A.",
                "options": {
                    "composition": ["c1", "c2"],
                    "palette": ["p1", "p2"],
                    "medium": ["m1"],
                    "mood": ["mo1"],
                },
            },
            {
                "id": "B",
                "hook": "Hook B.",
                "narrative": "Narrative B.",
                "options": {
                    "composition": ["c1", "c2"],
                    "palette": ["p1", "p2"],
                    "medium": ["m1"],
                    "mood": ["mo1"],
                },
                "avoid": ["x"],
            },
        ]
    }
    ideas = blackbox_scoring.parse_idea_cards_json(json.dumps(payload), expected_num_ideas=2)
    assert [idea["id"] for idea in ideas] == ["A", "B"]
    assert ideas[0]["options"]["palette"]


def test_parse_idea_cards_json_duplicate_id_raises():
    payload = {
        "ideas": [
            {
                "id": "A",
                "hook": "Hook A.",
                "narrative": "Narrative A.",
                "options": {
                    "composition": ["c1", "c2"],
                    "palette": ["p1", "p2"],
                    "medium": ["m1"],
                    "mood": ["mo1"],
                },
            },
            {
                "id": "A",
                "hook": "Hook B.",
                "narrative": "Narrative B.",
                "options": {
                    "composition": ["c1", "c2"],
                    "palette": ["p1", "p2"],
                    "medium": ["m1"],
                    "mood": ["mo1"],
                },
            },
        ]
    }
    with pytest.raises(ValueError, match="duplicate idea id"):
        blackbox_scoring.parse_idea_cards_json(json.dumps(payload), expected_num_ideas=2)


def test_parse_judge_scores_json_valid():
    out = {"scores": [{"id": "A", "score": 0}, {"id": "B", "score": 78}]}
    scores = blackbox_scoring.parse_judge_scores_json(json.dumps(out), expected_ids=["A", "B"])
    assert scores == {"A": 0, "B": 78}


def test_parse_judge_scores_json_rejects_non_int_score():
    out = {"scores": [{"id": "A", "score": 1.5}, {"id": "B", "score": 78}]}
    with pytest.raises(ValueError, match="invalid_judge_output"):
        blackbox_scoring.parse_judge_scores_json(json.dumps(out), expected_ids=["A", "B"])


def test_selection_determinism_and_exploration_branch():
    idea_cards = [
        {"id": i, "hook": "h", "narrative": "n", "options": {"composition": ["c1", "c2"], "palette": ["p1", "p2"], "medium": ["m"], "mood": ["mo"]}, "avoid": []}
        for i in blackbox_scoring.expected_idea_ids(8)
    ]
    scores = {card["id"]: idx for idx, card in enumerate(reversed(idea_cards))}
    rng1 = random.Random(123)
    rng2 = random.Random(123)

    pick1 = blackbox_scoring.select_candidate(
        scores=scores, idea_cards=idea_cards, exploration_rate=1.0, rng=rng1
    )
    pick2 = blackbox_scoring.select_candidate(
        scores=scores, idea_cards=idea_cards, exploration_rate=1.0, rng=rng2
    )
    assert pick1.selected_id == pick2.selected_id
    assert pick1.selection_mode == "explore"


def test_extract_recent_motif_summary(tmp_path):
    csv_path = tmp_path / "generations.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation_id",
                "selected_concepts",
                "final_image_prompt",
                "image_path",
                "created_at",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow({"generation_id": "g1", "final_image_prompt": "A vivid sunset over the ocean"})
        writer.writerow({"generation_id": "g2", "final_image_prompt": "Sunset with trees and moon"})
        writer.writerow({"generation_id": "g3", "final_image_prompt": "A calm sunrise (not sunset) scene"})

    summary = blackbox_scoring.extract_recent_motif_summary(
        generations_csv_path=str(csv_path),
        novelty_cfg=PromptNoveltyConfig(enabled=True, window=25),
    )
    tokens = {item["token"] for item in summary["top_tokens"]}
    assert "sunset" in tokens


def test_extract_recent_motif_summary_df_uses_document_frequency(tmp_path):
    csv_path = tmp_path / "generations.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["generation_id", "final_image_prompt"])
        writer.writeheader()
        writer.writerow({"generation_id": "g1", "final_image_prompt": "sunset " * 10})
        writer.writerow({"generation_id": "g2", "final_image_prompt": "sunset"})

    summary = blackbox_scoring.extract_recent_motif_summary(
        generations_csv_path=str(csv_path),
        novelty_cfg=PromptNoveltyConfig(
            enabled=True,
            window=25,
            method="df_overlap_v1",  # type: ignore[arg-type]
            df_min=1,
        ),
    )
    motifs = {item["token"]: item for item in summary["motifs"]}
    assert motifs["sunset"]["df"] == 2
    assert motifs["sunset"]["w"] == 2


def test_df_overlap_penalty_normalization_prevents_trivial_saturation():
    tokens = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    motifs = [{"token": tok, "df": 10, "w": 10} for tok in tokens]
    summary = {"enabled": True, "method": "df_overlap_v1", "motifs": motifs, "total_weight": 100}
    cfg = PromptNoveltyConfig(enabled=True, window=25, method="df_overlap_v1")  # type: ignore[arg-type]

    candidates = [
        {"id": "A", "prompt": "alpha"},
        {"id": "B", "prompt": "alpha bravo charlie delta echo"},
    ]
    penalties, _breakdown = blackbox_scoring.novelty_penalties(
        candidates, cfg, summary, text_field="prompt"
    )
    assert penalties["A"] < cfg.max_penalty
    assert penalties["B"] > penalties["A"]


def test_df_overlap_scaffolding_stopwords_filtered(tmp_path):
    csv_path = tmp_path / "generations.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["generation_id", "final_image_prompt"])
        writer.writeheader()
        writer.writerow({"generation_id": "g1", "final_image_prompt": "PROMPT IMAGE lighting sunset"})
        writer.writerow({"generation_id": "g2", "final_image_prompt": "prompt image Lighting sunset"})
        writer.writerow({"generation_id": "g3", "final_image_prompt": "prompt image lighting sunrise"})

    summary = blackbox_scoring.extract_recent_motif_summary(
        generations_csv_path=str(csv_path),
        novelty_cfg=PromptNoveltyConfig(
            enabled=True,
            window=25,
            method="df_overlap_v1",  # type: ignore[arg-type]
            df_min=2,
        ),
    )
    motif_tokens = {item["token"] for item in summary["motifs"]}
    assert "sunset" in motif_tokens
    assert "prompt" not in motif_tokens
    assert "image" not in motif_tokens
    assert "lighting" not in motif_tokens


def test_legacy_v0_penalties_unchanged():
    candidates = [{"id": "A", "prompt": "sunset ocean"}, {"id": "B", "prompt": "ocean only"}]
    summary = {
        "enabled": True,
        "top_tokens": [{"token": "sunset", "count": 3}, {"token": "ocean", "count": 10}],
    }
    cfg = PromptNoveltyConfig(enabled=True, window=25, method="legacy_v0")  # type: ignore[arg-type]
    penalties, breakdown = blackbox_scoring.novelty_penalties(
        candidates, cfg, summary, text_field="prompt"
    )
    assert penalties == {"A": 8, "B": 5}
    assert breakdown["A"]["penalty"] == 8


def test_integration_scoring_enabled_isolated_from_downstream(tmp_path, monkeypatch):
    profile_sentinel = "__TEST_PROFILE_ABSTRACT__"
    idea_sentinel_a = "__TEST_IDEA_CARD_A__"
    idea_sentinel_b = "__TEST_IDEA_CARD_B__"
    judge_sentinel = "__TEST_JUDGE__"
    final_sentinel = "__TEST_FINAL_PROMPT__"

    num_ideas = 2
    expected_ids = blackbox_scoring.expected_idea_ids(num_ideas)

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
    judge_output = json.dumps(
        {"scores": [{"id": expected_ids[0], "score": 10}, {"id": expected_ids[1], "score": 90}]}
    )

    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            self.model = "base-model"
            self.calls: list[dict[str, object]] = []

        def text_chat(self, messages, **kwargs):
            self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
            last_user = (messages[-1].get("content", "") if messages else "") or ""
            if last_user == profile_sentinel:
                return "- broad taste\n- avoid cliche"
            if last_user in idea_output_by_prompt:
                return idea_output_by_prompt[last_user]
            if last_user == judge_sentinel:
                return judge_output
            if last_user == final_sentinel:
                return "FINAL_DRAFT"
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

    def fake_profile_abstraction_prompt(**_kwargs):
        return profile_sentinel

    def fake_idea_card_generate_prompt(*, idea_id: str, **_kwargs):
        return idea_prompt_by_id[idea_id]

    def fake_idea_cards_judge_prompt(**_kwargs):
        return judge_sentinel

    def fake_final_prompt_from_selected_idea_prompt(**_kwargs):
        return final_sentinel

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
            "scoring": {
                "enabled": True,
                "num_ideas": num_ideas,
                "exploration_rate": 0.0,
                "judge_temperature": 0.0,
                "judge_model": "judge-model",
                "generator_profile_abstraction": True,
                "novelty": {"enabled": False, "window": 0},
            },
        },
        "image": {
            "generation_path": str(generation_dir),
            "upscale_path": str(upscale_dir),
            "log_path": str(log_dir),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }

    generation_id = "unit_test_scoring"

    fake_text = FakeTextAI()
    monkeypatch.setattr(app_generate, "TextAI", lambda *args, **kwargs: fake_text)
    monkeypatch.setattr(app_generate, "ImageAI", FakeImageAI)
    monkeypatch.setattr(prompts, "profile_abstraction_prompt", fake_profile_abstraction_prompt)
    monkeypatch.setattr(prompts, "idea_card_generate_prompt", fake_idea_card_generate_prompt)
    monkeypatch.setattr(prompts, "idea_cards_judge_prompt", fake_idea_cards_judge_prompt)
    monkeypatch.setattr(
        prompts,
        "final_prompt_from_selected_idea_prompt",
        fake_final_prompt_from_selected_idea_prompt,
    )
    monkeypatch.setattr(
        app_generate,
        "generate_title",
        lambda **_kwargs: types.SimpleNamespace(
            title="Test Title", title_source="test", title_raw="Test Title"
        ),
    )

    app_generate.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = log_dir / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["outputs"]["prompt_pipeline"]["requested_plan"] == "auto"
    assert transcript["outputs"]["prompt_pipeline"]["plan"] == "blackbox"
    assert transcript["outputs"]["prompt_pipeline"]["refinement_policy"] == "tot"
    assert transcript["outputs"]["prompt_pipeline"]["capture_stage"] == "blackbox.image_prompt_creation"
    assert transcript["outputs"]["prompt_pipeline"]["blackbox_profile_sources"] == {
        "judge_profile_source": "raw",
        "final_profile_source": "raw",
    }
    assert transcript["outputs"]["prompt_pipeline"]["resolved_stages"] == [
        "preprompt.select_concepts",
        "preprompt.filter_concepts",
        "blackbox.prepare",
        "blackbox.profile_abstraction",
        "blackbox.idea_card_generate.A",
        "blackbox.idea_card_generate.B",
        "blackbox.idea_cards_assemble",
        "blackbox.idea_cards_judge_score",
        "blackbox.select_idea_card",
        "blackbox.image_prompt_creation",
    ]
    assert transcript["blackbox_scoring"]["selected_id"] == expected_ids[1]
    assert any(step["path"] == "pipeline/blackbox.idea_card_generate.A/draft" for step in transcript["steps"])
    assert any(step["path"] == "pipeline/blackbox.idea_card_generate.B/draft" for step in transcript["steps"])
    assert any(
        step["path"] == "pipeline/blackbox.idea_cards_judge_score/draft"
        for step in transcript["steps"]
    )

    idea_calls = [
        call["messages"]
        for call in fake_text.calls
        if isinstance(call.get("messages"), list)
        and call["messages"]
        and (call["messages"][-1].get("content") or "") in idea_output_by_prompt
    ]
    assert len(idea_calls) == num_ideas
    for call_messages in idea_calls:
        last_user = (call_messages[-1].get("content") or "")
        other_sentinels = {s for s in idea_output_by_prompt.keys() if s != last_user}
        assert all(profile_sentinel not in (msg.get("content") or "") for msg in call_messages)
        assert all("- broad taste" not in (msg.get("content") or "") for msg in call_messages)
        assert all(
            all(other not in (msg.get("content") or "") for other in other_sentinels)
            for msg in call_messages
        )

    judge_call = next(
        call
        for call in fake_text.calls
        if isinstance(call.get("messages"), list)
        and call["messages"]
        and call["messages"][-1].get("content") == judge_sentinel
    )
    assert judge_call["kwargs"].get("model") == "judge-model"

    judge_step = next(
        step
        for step in transcript["steps"]
        if step["path"] == "pipeline/blackbox.idea_cards_judge_score/draft"
    )
    assert judge_step["params"]["model"] == "judge-model"
    assert judge_step["params"]["base_model"] == "base-model"

    final_call = next(
        call["messages"]
        for call in fake_text.calls
        if isinstance(call.get("messages"), list)
        and call["messages"]
        and call["messages"][-1].get("content") == final_sentinel
    )
    assert all(judge_output not in (msg.get("content") or "") for msg in final_call)
