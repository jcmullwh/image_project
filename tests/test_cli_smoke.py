import json
from pathlib import Path

from image_project import cli
from image_project.app import generate as app_generate


def test_cli_list_stages_smoke(capsys):
    rc = cli.main(["list-stages"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "standard.initial_prompt" in out


def test_cli_generate_smoke_prompt_only(tmp_path, monkeypatch):
    categories_path = tmp_path / "categories.csv"
    categories_path.write_text(
        "Subject Matter,Narrative,Mood,Composition,Perspective,Style,Time Period_Context,Color Scheme\n"
        "Cat,Quest,Moody,Wide,Top-down,Baroque,Renaissance,Vibrant\n",
        encoding="utf-8",
    )

    profile_path = tmp_path / "profile.csv"
    profile_path.write_text("Likes,Dislikes\ncolorful,\n", encoding="utf-8")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  mode: prompt_only",
                "image:",
                f"  log_path: '{logs_dir.as_posix()}'",
                "prompt:",
                f"  categories_path: '{categories_path.as_posix()}'",
                f"  profile_path: '{profile_path.as_posix()}'",
                "  plan: baseline",
                "  refinement:",
                "    policy: none",
                "  concepts:",
                "    filters:",
                "      enabled: false",
                "rclone:",
                "  enabled: false",
                "upscale:",
                "  enabled: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("IMAGE_PROJECT_CONFIG", str(config_path))

    generation_id = "unit_test_cli_generate"
    monkeypatch.setattr(app_generate, "generate_unique_id", lambda: generation_id)

    class FakeTextAI:
        def __init__(self, *args, **kwargs):
            self.model = "fake"

        def text_chat(self, messages, **kwargs):
            return "OK"

    monkeypatch.setattr(app_generate, "TextAI", FakeTextAI)

    rc = cli.main(["generate"])
    assert rc == 0

    transcript_path = logs_dir / f"{generation_id}_transcript.json"
    assert transcript_path.exists()

    loaded = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert loaded["generation_id"] == generation_id

