from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from upscaling import UpscaleConfig, _compute_target_size, upscale_image_to_4k


def test_compute_target_size_landscape():
    new_w, new_h = _compute_target_size(1536, 1024, 3840)
    assert (new_w, new_h) == (3840, 2560)


def test_compute_target_size_portrait():
    new_w, new_h = _compute_target_size(1024, 1536, 3840)
    assert (new_w, new_h) == (2560, 3840)


def test_skip_when_already_large_enough(tmp_path: Path):
    in_path = tmp_path / "in.jpg"
    out_path = tmp_path / "out.jpg"

    Image.new("RGB", (4000, 1000)).save(in_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(target_long_edge_px=3840)
    upscale_image_to_4k(input_path=str(in_path), output_path=str(out_path), config=cfg)

    assert out_path.exists()
    with Image.open(out_path) as im:
        assert im.size == (4000, 1000)


def test_fallback_resize_when_binary_missing(tmp_path: Path):
    in_path = tmp_path / "in.jpg"
    out_path = tmp_path / "out.jpg"

    Image.new("RGB", (1536, 1024)).save(in_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(
        target_long_edge_px=3840,
        realesrgan_binary=None,
        allow_fallback_resize=True,
    )

    upscale_image_to_4k(input_path=str(in_path), output_path=str(out_path), config=cfg)

    with Image.open(out_path) as im:
        assert im.size == (3840, 2560)


def test_error_when_binary_missing_and_no_fallback(tmp_path: Path):
    in_path = tmp_path / "in.jpg"
    out_path = tmp_path / "out.jpg"

    Image.new("RGB", (1536, 1024)).save(in_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(
        target_long_edge_px=3840,
        realesrgan_binary=None,
        allow_fallback_resize=False,
    )

    with pytest.raises(FileNotFoundError):
        upscale_image_to_4k(input_path=str(in_path), output_path=str(out_path), config=cfg)


def test_error_when_default_model_dir_missing(tmp_path: Path):
    in_path = tmp_path / "in.jpg"
    out_path = tmp_path / "out.jpg"
    binary_path = tmp_path / "realesrgan-ncnn-vulkan.exe"
    binary_path.write_text("")  # dummy executable placeholder
    # Intentionally do NOT create models/ directory.

    Image.new("RGB", (1536, 1024)).save(in_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(
        target_long_edge_px=3840,
        realesrgan_binary=str(binary_path),
        allow_fallback_resize=False,
    )

    with pytest.raises(FileNotFoundError):
        upscale_image_to_4k(input_path=str(in_path), output_path=str(out_path), config=cfg)


def test_error_when_model_files_missing(tmp_path: Path):
    in_path = tmp_path / "in.jpg"
    out_path = tmp_path / "out.jpg"
    binary_path = tmp_path / "realesrgan-ncnn-vulkan.exe"
    binary_path.write_text("")  # dummy executable placeholder
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # No param/bin files written.

    Image.new("RGB", (1536, 1024)).save(in_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(
        target_long_edge_px=3840,
        realesrgan_binary=str(binary_path),
        model_path=str(model_dir),
        allow_fallback_resize=False,
    )

    with pytest.raises(FileNotFoundError):
        upscale_image_to_4k(input_path=str(in_path), output_path=str(out_path), config=cfg)


def test_error_when_model_path_missing(tmp_path: Path):
    in_path = tmp_path / "in.jpg"
    out_path = tmp_path / "out.jpg"
    binary_path = tmp_path / "realesrgan-ncnn-vulkan.exe"
    binary_path.write_text("")  # dummy executable placeholder
    missing_models = tmp_path / "models_missing"

    Image.new("RGB", (1536, 1024)).save(in_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(
        target_long_edge_px=3840,
        realesrgan_binary=str(binary_path),
        model_path=str(missing_models),
        allow_fallback_resize=False,
    )

    with pytest.raises(FileNotFoundError):
        upscale_image_to_4k(input_path=str(in_path), output_path=str(out_path), config=cfg)


def test_model_path_forwarded_to_runner(tmp_path: Path):
    in_path = tmp_path / "in.jpg"
    out_path = tmp_path / "out.jpg"
    binary_path = tmp_path / "realesrgan-ncnn-vulkan.exe"
    binary_path.write_text("")  # dummy executable placeholder
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # Create dummy model files so validation passes.
    (model_dir / "realesrgan-x4plus.param").write_text("")
    (model_dir / "realesrgan-x4plus.bin").write_text("")

    Image.new("RGB", (1536, 1024)).save(in_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(
        target_long_edge_px=3840,
        realesrgan_binary=str(binary_path),
        model_path=str(model_dir),
    )

    def fake_run(**kwargs):
        # Create the expected intermediate output so the resize step succeeds.
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100)).save(kwargs["output_path"], format="PNG")

    with patch("upscaling._run_realesrgan_ncnn_vulkan", side_effect=fake_run) as mock_run:
        upscale_image_to_4k(input_path=str(in_path), output_path=str(out_path), config=cfg)

    mock_run.assert_called_once()
    _, run_kwargs = mock_run.call_args
    assert run_kwargs["model_path"] == str(model_dir)
