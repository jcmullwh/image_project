"""Simple CLI to manually test image upscaling.

Usage:
    python scripts/manual_upscale.py path/to/image.jpg

The script writes the upscaled image next to the input with `_4k` appended
before the extension (e.g., `image_4k.jpg`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running as a loose script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upscaling import UpscaleConfig, upscale_image_to_4k
from run_config import parse_aspect_ratio, parse_bool
from utils import load_config


def build_output_path(input_path: Path) -> Path:
    return input_path.with_stem(input_path.stem + "_4k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually upscale an image to 4K-ish resolution.")
    parser.add_argument("image_path", type=Path, help="Path to the source image to upscale.")
    parser.add_argument(
        "--target-long-edge",
        type=int,
        default=None,
        help="Target long edge in pixels. If omitted, falls back to config or 3840.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="Explicit target width in pixels. Requires --target-height.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=None,
        help="Explicit target height in pixels. Requires --target-width.",
    )
    parser.add_argument(
        "--target-aspect-ratio",
        type=str,
        default=None,
        help='Target aspect ratio like "16:9". Ignored if explicit width/height are provided.',
    )
    parser.add_argument(
        "--realesrgan-binary",
        type=str,
        default=None,
        help="Optional path to realesrgan-ncnn-vulkan executable. If omitted, PATH/env lookup is used.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the Real-ESRGAN models directory (folder containing *.bin/param files).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name to pass to realesrgan-ncnn-vulkan. Falls back to config or realesrgan-x4plus.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size for realesrgan-ncnn-vulkan. 0 lets the binary choose automatically.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation flag for realesrgan-ncnn-vulkan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load config defaults if available.
    config = {}
    try:
        config, _cfg_meta = load_config()
    except Exception:
        config = {}

    upscale_cfg = config.get("upscale", {}) if isinstance(config, dict) else {}

    target_long_edge = (
        args.target_long_edge if args.target_long_edge is not None else int(upscale_cfg.get("target_long_edge_px", 3840))
    )
    if target_long_edge <= 0:
        print("target_long_edge must be greater than 0", file=sys.stderr)
        return 1

    target_width = args.target_width
    target_height = args.target_height
    if target_width is None and target_height is None:
        cfg_width = upscale_cfg.get("target_width_px")
        cfg_height = upscale_cfg.get("target_height_px")
        if cfg_width is not None and cfg_height is not None:
            try:
                target_width = int(cfg_width)
                target_height = int(cfg_height)
            except Exception:  # noqa: BLE001
                print("Invalid config values for target_width_px/target_height_px", file=sys.stderr)
                return 1

    if (target_width is None) != (target_height is None):
        print("Provide both --target-width and --target-height together.", file=sys.stderr)
        return 1
    if target_width is not None and target_width <= 0:
        print("target_width must be greater than 0", file=sys.stderr)
        return 1
    if target_height is not None and target_height <= 0:
        print("target_height must be greater than 0", file=sys.stderr)
        return 1

    raw_aspect_ratio = args.target_aspect_ratio
    if raw_aspect_ratio is None:
        raw_aspect_ratio = upscale_cfg.get("target_aspect_ratio")
    target_aspect_ratio: float | None = None
    if raw_aspect_ratio is not None:
        try:
            target_aspect_ratio = parse_aspect_ratio(raw_aspect_ratio, "upscale.target_aspect_ratio")
        except ValueError as exc:
            print(f"Invalid aspect ratio: {exc}", file=sys.stderr)
            return 1

    if target_aspect_ratio is not None and target_width is not None:
        print("Provide either explicit width/height or a target aspect ratio, not both.", file=sys.stderr)
        return 1

    realesrgan_binary = args.realesrgan_binary or upscale_cfg.get("realesrgan_binary")
    model_path = args.model_path or upscale_cfg.get("model_path")
    model_name = args.model_name or upscale_cfg.get("model_name", "realesrgan-x4plus")
    tile_size = args.tile_size if args.tile_size is not None else int(upscale_cfg.get("tile_size", 0))
    tta = args.tta or parse_bool(upscale_cfg.get("tta", False), "upscale.tta")
    # Explicitly disable any silent fallback paths.
    allow_fallback_resize = False

    input_path: Path = args.image_path
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}", file=sys.stderr)
        return 1

    output_path = build_output_path(input_path)

    model_path_str: str | None = None
    if model_path:
        model_path_path = Path(model_path)
        if not model_path_path.exists():
            print(f"Model path does not exist: {model_path_path}", file=sys.stderr)
            return 1
        model_path_str = str(model_path_path)

    cfg = UpscaleConfig(
        target_long_edge_px=target_long_edge,
        target_width_px=target_width,
        target_height_px=target_height,
        target_aspect_ratio=target_aspect_ratio,
        realesrgan_binary=realesrgan_binary,
        model_path=model_path_str,
        model_name=model_name,
        tile_size=tile_size,
        tta=tta,
        allow_fallback_resize=allow_fallback_resize,
    )

    try:
        upscale_image_to_4k(
            input_path=str(input_path),
            output_path=str(output_path),
            config=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Upscaling failed: {exc}", file=sys.stderr)
        return 1

    print(f"Upscaled image written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
