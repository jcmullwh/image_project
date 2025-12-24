import base64
import io
import json
import types

import pandas as pd
from PIL import Image

from image_project.app import generate as app_generate


def _write_minimal_categories(path) -> None:
    pd.DataFrame(
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
    ).to_csv(path, index=False)


def _write_minimal_profile(path) -> None:
    pd.DataFrame({"Likes": ["colorful"], "Dislikes": [None]}).to_csv(path, index=False)


class FakeTextAI:
    def __init__(self, *args, **kwargs):
        pass

    def text_chat(self, messages, **kwargs):
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


def _base_cfg_dict(tmp_path) -> dict:
    categories_path = tmp_path / "categories.csv"
    profile_path = tmp_path / "profile.csv"
    generation_dir = tmp_path / "generated"
    upscale_dir = tmp_path / "upscaled"
    log_dir = tmp_path / "logs"
    generations_csv = tmp_path / "generations.csv"

    _write_minimal_categories(categories_path)
    _write_minimal_profile(profile_path)

    return {
        "prompt": {
            "categories_path": str(categories_path),
            "profile_path": str(profile_path),
            "generations_path": str(generations_csv),
            "random_seed": 123,
            "refinement": {"policy": "none"},
        },
        "image": {
            "generation_path": str(generation_dir),
            "upscale_path": str(upscale_dir),
            "log_path": str(log_dir),
        },
        "rclone": {"enabled": False},
        "upscale": {"enabled": False},
    }


def _patch_common(monkeypatch):
    monkeypatch.setattr(app_generate, "TextAI", FakeTextAI)
    monkeypatch.setattr(app_generate, "ImageAI", FakeImageAI)
    monkeypatch.setattr(
        app_generate,
        "generate_title",
        lambda **_kwargs: types.SimpleNamespace(
            title="Test Title", title_source="test", title_raw="Test Title"
        ),
    )


def test_simple_plan_runs_two_stages(tmp_path, monkeypatch):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "simple"

    _patch_common(monkeypatch)

    generation_id = "unit_test_simple_plan"
    app_generate.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = tmp_path / "logs" / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["outputs"]["prompt_pipeline"]["plan"] == "simple"
    assert transcript["outputs"]["prompt_pipeline"]["resolved_stages"] == [
        "preprompt.select_concepts",
        "preprompt.filter_concepts",
        "standard.initial_prompt",
        "standard.image_prompt_creation",
    ]
    assert transcript["outputs"]["prompt_pipeline"]["capture_stage"] == "standard.image_prompt_creation"

    recorded_step_paths = [step.get("path") for step in transcript.get("steps", [])]
    assert "pipeline/standard.initial_prompt/draft" in recorded_step_paths
    assert "pipeline/standard.image_prompt_creation/draft" in recorded_step_paths


def test_simple_no_concepts_plan_skips_preprompt(tmp_path, monkeypatch):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "simple_no_concepts"

    _patch_common(monkeypatch)

    generation_id = "unit_test_simple_no_concepts_plan"
    app_generate.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = tmp_path / "logs" / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["outputs"]["prompt_pipeline"]["plan"] == "simple_no_concepts"
    assert transcript["outputs"]["prompt_pipeline"]["resolved_stages"] == [
        "standard.initial_prompt",
        "standard.image_prompt_creation",
    ]
    assert transcript["outputs"]["prompt_pipeline"]["capture_stage"] == "standard.image_prompt_creation"

    recorded_step_paths = [step.get("path") for step in transcript.get("steps", [])]
    assert "pipeline/preprompt.select_concepts/action" not in recorded_step_paths
    assert "pipeline/preprompt.filter_concepts/action" not in recorded_step_paths
    assert "pipeline/standard.initial_prompt/draft" in recorded_step_paths
    assert "pipeline/standard.image_prompt_creation/draft" in recorded_step_paths


def test_profile_only_forces_context_injection_off(tmp_path, monkeypatch):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "profile_only"
    cfg_dict["context"] = {"enabled": True, "injectors": ["season"]}

    _patch_common(monkeypatch)

    generation_id = "unit_test_profile_only_plan"
    app_generate.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = tmp_path / "logs" / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["outputs"]["prompt_pipeline"]["plan"] == "profile_only"
    assert transcript["outputs"]["prompt_pipeline"]["context_injection"] == "disabled"
    assert transcript["outputs"]["prompt_pipeline"]["context_enabled"] is False
    assert "context" not in transcript


def test_standard_honors_context_injection_when_enabled(tmp_path, monkeypatch):
    cfg_dict = _base_cfg_dict(tmp_path)
    cfg_dict["prompt"]["plan"] = "standard"
    cfg_dict["context"] = {"enabled": True, "injectors": ["season"]}

    _patch_common(monkeypatch)

    generation_id = "unit_test_standard_context_enabled"
    app_generate.run_generation(cfg_dict, generation_id=generation_id)

    transcript_path = tmp_path / "logs" / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["outputs"]["prompt_pipeline"]["plan"] == "standard"
    assert transcript["outputs"]["prompt_pipeline"]["context_injection"] == "config"
    assert transcript["outputs"]["prompt_pipeline"]["context_enabled"] is True
    assert "context" in transcript
    assert "season" in transcript["context"]
