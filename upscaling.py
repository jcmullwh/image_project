"""Image upscaling utilities.

Goal: add an optional "upscale to 4K" step without pulling in heavyweight
ML dependencies (PyTorch, CUDA, etc.).

Primary backend: Real-ESRGAN (NCNN Vulkan portable executable) invoked via CLI.

Design principles:
  - Fail loudly by default if upscaling is enabled but the tool is missing.
  - Deterministic output sizing: define 4K as a configurable long-edge target.
  - Keep the integration surface small (one public function).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class UpscaleError(RuntimeError):
    """Raised when the upscaling step fails."""


@dataclass(frozen=True)
class UpscaleConfig:
    """Configuration for 4K upscaling.

    target_long_edge_px:
        "4K" target expressed as the maximum(width, height). 3840 matches UHD.

    engine:
        Currently only "realesrgan-ncnn-vulkan" is supported.

    realesrgan_binary:
        Optional path to the realesrgan-ncnn-vulkan executable.
        If None, we try environment variable and PATH lookup.

    model_name:
        E.g. "realesrgan-x4plus" or "realesrgan-x4plus-anime".

    model_path:
        Optional path to the Real-ESRGAN models directory. If not set, the
        binary's default lookup is used (local "models" directory next to exe).

    tile_size:
        0 lets Real-ESRGAN auto-select. Smaller values reduce VRAM usage.

    tta:
        Test-time augmentation. Slower; may improve quality.

    allow_fallback_resize:
        If True and the Real-ESRGAN binary is unavailable, fall back to a
        high-quality Lanczos resize (non-AI). Default False.
    """

    target_long_edge_px: int = 3840
    engine: str = "realesrgan-ncnn-vulkan"
    realesrgan_binary: str | None = None
    model_name: str = "realesrgan-x4plus"
    model_path: str | None = None
    tile_size: int = 0
    tta: bool = False
    allow_fallback_resize: bool = False


def _compute_target_size(width: int, height: int, target_long_edge_px: int) -> tuple[int, int]:
    """Compute (new_w, new_h) preserving aspect ratio with a fixed long edge."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    if target_long_edge_px <= 0:
        raise ValueError("target_long_edge_px must be > 0")

    long_edge = max(width, height)
    if long_edge == target_long_edge_px:
        return width, height

    scale = target_long_edge_px / float(long_edge)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))

    # Guard against rounding to 0 in degenerate cases.
    return max(1, new_w), max(1, new_h)


def _find_realesrgan_binary(explicit_path: str | None) -> str | None:
    """Locate realesrgan-ncnn-vulkan executable.

    Search order:
      1) explicit_path
      2) env var REALESRGAN_NCNN_VULKAN_PATH
      3) PATH lookup for common names
    """
    candidates: list[str] = []

    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("REALESRGAN_NCNN_VULKAN_PATH")
    if env_path:
        candidates.append(env_path)

    for name in [
        "realesrgan-ncnn-vulkan",
        "realesrgan-ncnn-vulkan.exe",
        "realesrgan-ncnn-vulkan-cli",
        "realesrgan-ncnn-vulkan-cli.exe",
    ]:
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for candidate in candidates:
        p = Path(candidate)
        if p.exists() and p.is_file():
            return str(p)
    return None


def _run_realesrgan_ncnn_vulkan(
    *,
    binary: str,
    input_path: str,
    output_path: str,
    model_name: str,
    scale: int = 4,
    tile_size: int = 0,
    tta: bool = False,
    model_path: str | None = None,
    gpu_id: str | None = None,
    output_format: str | None = None,
    timeout_s: int = 10 * 60,
) -> None:
    """Invoke realesrgan-ncnn-vulkan CLI."""

    if scale not in {2, 3, 4}:
        raise ValueError("realesrgan-ncnn-vulkan only supports scale 2, 3, or 4")
    if tile_size < 0:
        raise ValueError("tile_size must be >= 0")

    cmd: list[str] = [binary, "-i", input_path, "-o", output_path, "-n", model_name, "-s", str(scale)]
    cmd.extend(["-t", str(tile_size)])

    if model_path:
        cmd.extend(["-m", model_path])
    if gpu_id:
        cmd.extend(["-g", gpu_id])
    if output_format:
        cmd.extend(["-f", output_format])
    if tta:
        cmd.append("-x")

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise UpscaleError(
            "Real-ESRGAN (ncnn-vulkan) failed. "
            f"returncode={proc.returncode}. "
            f"stdout={stdout[-2000:]!r} stderr={stderr[-2000:]!r}"
        )


def upscale_image_to_4k(
    *,
    input_path: str,
    output_path: str,
    config: UpscaleConfig | None = None,
) -> str:
    """Upscale an image to a 4K-ish target long edge.

    Returns the output_path.

    Behavior:
      - If the input already meets/exceeds the target long edge, the image is
        copied to output_path.
      - Otherwise we run an x4 Real-ESRGAN model once, then resize (Lanczos) to
        hit the exact target long-edge.
      - If the input is extremely small (< target/4), x4 may still be below the
        target; in that case we do an additional Lanczos resize.
    """

    cfg = config or UpscaleConfig()

    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as im:
        width, height = im.size

    if max(width, height) >= cfg.target_long_edge_px:
        shutil.copyfile(in_path, out_path)
        return str(out_path)

    # Determine desired final size up-front.
    target_w, target_h = _compute_target_size(width, height, cfg.target_long_edge_px)

    if cfg.engine != "realesrgan-ncnn-vulkan":
        raise ValueError(f"Unsupported upscaling engine: {cfg.engine}")

    binary = _find_realesrgan_binary(cfg.realesrgan_binary)
    if not binary:
        if cfg.allow_fallback_resize:
            _lanczos_resize(in_path, out_path, target_w, target_h)
            return str(out_path)
        raise FileNotFoundError(
            "Upscaling is enabled but realesrgan-ncnn-vulkan was not found. "
            "Install the Real-ESRGAN NCNN Vulkan portable executable and either "
            "(a) put it on PATH, (b) set REALESRGAN_NCNN_VULKAN_PATH, or "
            "(c) set config.upscale.realesrgan_binary."
        )
    # Resolve model directory (explicit or default next to binary) and verify expected files exist.
    model_dir = Path(cfg.model_path) if cfg.model_path else Path(binary).parent / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Real-ESRGAN model directory not found: {model_dir}")

    param_file = model_dir / f"{cfg.model_name}.param"
    bin_file = model_dir / f"{cfg.model_name}.bin"
    missing_files = [p for p in (param_file, bin_file) if not p.exists()]
    if missing_files:
        missing_list = ", ".join(str(p) for p in missing_files)
        raise FileNotFoundError(f"Real-ESRGAN model files missing: {missing_list}")

    model_path_str = str(model_dir)

    # Run x4 upscaling once, output to a temporary file, then resize to the
    # exact target dimensions (if needed) and save as the desired format.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / (out_path.stem + "_x4.png")

        _run_realesrgan_ncnn_vulkan(
            binary=binary,
            input_path=str(in_path),
            output_path=str(tmp_out),
            model_name=cfg.model_name,
            model_path=model_path_str,
            scale=4,
            tile_size=cfg.tile_size,
            tta=cfg.tta,
            output_format="png",
        )

        # Now convert/resize to the final target.
        _lanczos_resize(tmp_out, out_path, target_w, target_h)

    return str(out_path)


def _lanczos_resize(src: Path, dst: Path, width: int, height: int) -> None:
    """Resize using Pillow's Lanczos filter and save as JPEG/PNG based on dst."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resize target: {width}x{height}")

    with Image.open(src) as im:
        im = im.convert("RGB")
        resized = im.resize((width, height), resample=Image.Resampling.LANCZOS)

        ext = dst.suffix.lower().lstrip(".")
        if ext in {"jpg", "jpeg"}:
            resized.save(dst, format="JPEG", quality=95, optimize=True)
        elif ext == "png":
            resized.save(dst, format="PNG", optimize=True)
        else:
            # Default to JPEG to keep output size reasonable.
            resized.save(dst, format="JPEG", quality=95, optimize=True)
