import base64
import csv
import io
import json
import random
import types

import pandas as pd
import pytest
from PIL import Image

import blackbox_scoring
import main


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
        generations_csv_path=str(csv_path), window=25
    )
    tokens = {item["token"] for item in summary["top_tokens"]}
    assert "sunset" in tokens


def test_integration_scoring_enabled_isolated_from_downstream(tmp_path, monkeypatch):
    profile_sentinel = "__TEST_PROFILE_ABSTRACT__"
    ideas_sentinel = "__TEST_IDEA_CARDS__"
    judge_sentinel = "__TEST_JUDGE__"
    final_sentinel = "__TEST_FINAL_PROMPT__"

    num_ideas = 2
    expected_ids = blackbox_scoring.expected_idea_ids(num_ideas)

    idea_cards_json = json.dumps(
        {
            "ideas": [
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
                },
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
                },
            ]
        }
    )
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
            if last_user == ideas_sentinel:
                return idea_cards_json
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

    def fake_idea_cards_generate_prompt(**_kwargs):
        return ideas_sentinel

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
    monkeypatch.setattr(main, "TextAI", lambda *args, **kwargs: fake_text)
    monkeypatch.setattr(main, "ImageAI", FakeImageAI)
    monkeypatch.setattr(main, "profile_abstraction_prompt", fake_profile_abstraction_prompt)
    monkeypatch.setattr(main, "idea_cards_generate_prompt", fake_idea_cards_generate_prompt)
    monkeypatch.setattr(main, "idea_cards_judge_prompt", fake_idea_cards_judge_prompt)
    monkeypatch.setattr(main, "final_prompt_from_selected_idea_prompt", fake_final_prompt_from_selected_idea_prompt)
    monkeypatch.setattr(
        main,
        "generate_title",
        lambda **_kwargs: types.SimpleNamespace(
            title="Test Title", title_source="test", title_raw="Test Title"
        ),
    )

    main.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = log_dir / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["blackbox_scoring"]["selected_id"] == expected_ids[1]
    assert any(step["path"].endswith("idea_cards_generate") for step in transcript["steps"])
    assert any(step["path"].endswith("idea_cards_judge_score") for step in transcript["steps"])

    idea_call = next(
        call["messages"]
        for call in fake_text.calls
        if isinstance(call.get("messages"), list)
        and call["messages"]
        and call["messages"][-1].get("content") == ideas_sentinel
    )
    assert all(profile_sentinel not in (msg.get("content") or "") for msg in idea_call)
    assert all("- broad taste" not in (msg.get("content") or "") for msg in idea_call)

    judge_call = next(
        call
        for call in fake_text.calls
        if isinstance(call.get("messages"), list)
        and call["messages"]
        and call["messages"][-1].get("content") == judge_sentinel
    )
    assert judge_call["kwargs"].get("model") == "judge-model"

    judge_step = next(step for step in transcript["steps"] if step["path"].endswith("idea_cards_judge_score"))
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
