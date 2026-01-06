import io
from pathlib import Path

import pytest
from PIL import Image

from titles import (
    TITLE_SOURCE_FALLBACK,
    append_manifest_row,
    generate_title,
    get_next_seq,
    read_manifest,
    sanitize_title,
    validate_title,
)
from utils import save_image


class _FakeTextAI:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    def text_chat(self, messages, **kwargs):
        if self._idx >= len(self._responses):
            raise AssertionError("FakeTextAI ran out of responses")
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


def _jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_next_seq_starts_at_1_when_manifest_missing(tmp_path: Path):
    manifest_path = tmp_path / "titles_manifest.csv"
    assert get_next_seq(str(manifest_path)) == 1


def test_next_seq_increments_and_ignores_malformed_rows(tmp_path: Path):
    manifest_path = tmp_path / "titles_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "seq,title,generation_id,image_prompt,image_path",
                "1,Turquoise Citadel,gen1,prompt1,path1.jpg",
                ",Bad Row,gen2,prompt2,path2.jpg",
                "not-an-int,Also Bad,gen3,prompt3,path3.jpg",
                "7,Amber Ladder,gen4,prompt4,path4.jpg",
            ]
        ),
        encoding="utf-8",
    )

    assert get_next_seq(str(manifest_path)) == 8


def test_append_manifest_row_creates_file_and_round_trips(tmp_path: Path):
    manifest_path = tmp_path / "titles_manifest.csv"

    row = {
        "seq": 1,
        "title": "Turquoise Citadel",
        "generation_id": "gen_123",
        "image_prompt": "final prompt",
        "image_path": str(tmp_path / "image.jpg"),
    }
    append_manifest_row(str(manifest_path), row)

    rows = read_manifest(str(manifest_path))
    assert len(rows) == 1
    assert rows[0]["seq"] == "1"
    assert rows[0]["title"] == "Turquoise Citadel"
    assert rows[0]["generation_id"] == "gen_123"
    assert rows[0]["image_prompt"] == "final prompt"
    assert rows[0]["image_path"] == str(tmp_path / "image.jpg")
    assert rows[0]["created_at"]


def test_title_validation_and_sanitization():
    validate_title("Turquoise Citadel")
    assert sanitize_title('"Turquoise Citadel"') == "Turquoise Citadel"
    validate_title(sanitize_title('"Turquoise Citadel"'))

    with pytest.raises(ValueError):
        validate_title("Turquoise")
    with pytest.raises(ValueError):
        validate_title("One Two Three Four Five")
    with pytest.raises(ValueError):
        validate_title("Turquoise, Citadel")
    with pytest.raises(ValueError):
        validate_title('Turquoise "Citadel"')


def test_generate_title_collision_retries_then_disambiguates():
    ai_text = _FakeTextAI(
        [
            "Turquoise Citadel",
            "Turquoise Citadel",
            "Turquoise Citadel",
        ]
    )

    result = generate_title(
        ai_text=ai_text,
        image_prompt="Some long image prompt here",
        avoid_titles=["Turquoise Citadel"],
        max_attempts=3,
    )

    assert result.title == "Turquoise Citadel II"


def test_generate_title_falls_back_instead_of_raising():
    ai_text = _FakeTextAI(
        [
            "",
            "turquoise citadel",
            "One Two Three Four Five",
        ]
    )

    result = generate_title(
        ai_text=ai_text,
        image_prompt="Some long image prompt here",
        avoid_titles=[],
        max_attempts=3,
    )

    assert result.title_source == TITLE_SOURCE_FALLBACK
    assert result.title == "One Two Three Four Five"
    assert len(result.attempts) == 3
    assert result.attempts[0]["reason"] == "sanitize_failed"


def test_caption_overlay_changes_pixels_without_resizing(tmp_path: Path):
    base_bytes = _jpeg_bytes(Image.new("RGB", (640, 360), (255, 255, 255)))

    baseline_path = tmp_path / "baseline.jpg"
    captioned_path = tmp_path / "captioned.jpg"

    save_image(base_bytes, str(baseline_path))
    save_image(base_bytes, str(captioned_path), caption_text="#001 - Turquoise Citadel")

    with Image.open(baseline_path) as baseline, Image.open(captioned_path) as captioned:
        assert baseline.size == captioned.size
        w, h = baseline.size
        baseline_px = baseline.getpixel((w // 2, h - 2))
        captioned_px = captioned.getpixel((w // 2, h - 2))
        assert baseline_px != captioned_px


def test_offline_integration_one_generation_writes_image_and_manifest(tmp_path: Path):
    manifest_path = tmp_path / "titles_manifest.csv"
    image_out = tmp_path / "out.jpg"

    ai_text = _FakeTextAI(["Turquoise Citadel"])
    prompt = "A detailed prompt for the image model."

    seq = get_next_seq(str(manifest_path))
    title = generate_title(ai_text=ai_text, image_prompt=prompt, avoid_titles=[]).title
    caption = f"#{seq:03d} - {title}"

    # "Mocked" image generation: deterministic bytes.
    image_bytes = _jpeg_bytes(Image.new("RGB", (320, 240), (200, 200, 200)))

    save_image(image_bytes, str(image_out), caption_text=caption)
    append_manifest_row(
        str(manifest_path),
        {
            "seq": seq,
            "title": title,
            "generation_id": "gen_test",
            "image_prompt": prompt,
            "image_path": str(image_out),
        },
    )

    assert image_out.exists()
    rows = read_manifest(str(manifest_path))
    assert rows and rows[0]["seq"] == "1"
    assert rows[0]["title"] == "Turquoise Citadel"
    assert rows[0]["generation_id"] == "gen_test"
    assert rows[0]["image_prompt"] == prompt
    assert rows[0]["image_path"] == str(image_out)

