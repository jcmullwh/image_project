import os
import sys
from pathlib import Path

from PIL import Image

from upscaling import UpscaleConfig, upscale_image_to_4k


def _make_fake_realesrgan(root: Path) -> Path:
    """Create a fake realesrgan executable that writes a tiny PNG and exits 0."""
    tool_lines = [
        "import sys",
        "from pathlib import Path",
        "from PIL import Image",
        "args = sys.argv[1:]",
        "try:",
        "    out_idx = args.index('-o')",
        "    out_path = Path(args[out_idx + 1])",
        "except (ValueError, IndexError):",
        "    sys.exit(2)",
        "out_path.parent.mkdir(parents=True, exist_ok=True)",
        "Image.new('RGB', (8, 8), (0, 255, 0)).save(out_path, format='PNG')",
    ]

    if os.name == "nt":
        script_path = root / "fake_realesrgan.py"
        script_path.write_text("\n".join(tool_lines))

        cmd_path = root / "realesrgan-ncnn-vulkan.cmd"
        cmd_path.write_text(
            "\r\n".join(
                [
                    "@echo off",
                    f'"{sys.executable}" "{script_path}" %*',
                ]
            )
        )
        return cmd_path

    exe_path = root / "realesrgan-ncnn-vulkan"
    exe_path.write_text("\n".join(["#!/usr/bin/env python3", *tool_lines]))
    os.chmod(exe_path, 0o755)
    return exe_path


def test_integration_uses_binary_and_writes_expected_size(tmp_path: Path):
    binary_path = _make_fake_realesrgan(tmp_path)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "realesrgan-x4plus.param").write_text("param")
    (model_dir / "realesrgan-x4plus.bin").write_text("bin")

    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "output.jpg"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(input_path, format="JPEG", quality=90)

    cfg = UpscaleConfig(
        target_long_edge_px=64,  # keep small for test speed
        realesrgan_binary=str(binary_path),
        model_path=str(model_dir),
        model_name="realesrgan-x4plus",
        tile_size=0,
        tta=False,
        allow_fallback_resize=False,
    )

    upscale_image_to_4k(
        input_path=str(input_path),
        output_path=str(output_path),
        config=cfg,
    )

    with Image.open(output_path) as im:
        assert im.size == (64, 48)
