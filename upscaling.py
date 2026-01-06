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

from PIL import Image, ImageOps

from utils import _overlay_caption


class UpscaleError(RuntimeError):
    """Raised when the upscaling step fails."""


@dataclass(frozen=True)
class UpscaleConfig:
    """Configuration for 4K upscaling.

    target_long_edge_px:
        "4K" target expressed as the maximum(width, height). 3840 matches UHD.
        Ignored when an explicit target width/height is provided.

    target_width_px / target_height_px:
        Optional explicit output dimensions. When both are set, they override
        target_long_edge_px and target_aspect_ratio.

    target_aspect_ratio:
        Optional desired aspect ratio (width / height), e.g., 16:9. When set,
        the final output is resized/cropped to that ratio using the configured
        target_long_edge_px to determine the long edge.

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
    target_width_px: int | None = None
    target_height_px: int | None = None
    target_aspect_ratio: float | None = None
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


def _normalize_aspect_ratio(value: float | int | str | tuple[int, int] | None) -> float | None:
    """Parse and validate an aspect ratio, returning width/height as a float."""
    if value is None:
        return None

    ratio: float
    if isinstance(value, (int, float)):
        ratio = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if ":" in raw:
            parts = raw.split(":")
        elif "/" in raw:
            parts = raw.split("/")
        else:
            parts = [raw]

        if len(parts) == 1:
            try:
                ratio = float(parts[0])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid aspect ratio: {value!r}") from exc
        elif len(parts) == 2:
            try:
                num = float(parts[0])
                denom = float(parts[1])
                ratio = num / denom
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid aspect ratio: {value!r}") from exc
        else:
            raise ValueError(f"Invalid aspect ratio: {value!r}")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            ratio = float(value[0]) / float(value[1])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid aspect ratio: {value!r}") from exc
    else:
        raise ValueError(f"Invalid aspect ratio: {value!r}")

    if ratio <= 0:
        raise ValueError(f"Invalid aspect ratio (must be > 0): {value!r}")
    return ratio


def _resolve_target_dimensions(
    *,
    source_width: int,
    source_height: int,
    cfg: UpscaleConfig,
) -> tuple[int, int, float]:
    """Compute the desired output dimensions and aspect ratio."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid image size: {source_width}x{source_height}")
    if cfg.target_long_edge_px <= 0:
        raise ValueError("target_long_edge_px must be > 0")

    if (cfg.target_width_px is None) != (cfg.target_height_px is None):
        raise ValueError("Both target_width_px and target_height_px must be set together.")

    if cfg.target_width_px is not None and cfg.target_height_px is not None:
        if cfg.target_width_px <= 0 or cfg.target_height_px <= 0:
            raise ValueError("target_width_px and target_height_px must be > 0")
        ratio = cfg.target_width_px / float(cfg.target_height_px)
        return cfg.target_width_px, cfg.target_height_px, ratio

    aspect_ratio = _normalize_aspect_ratio(cfg.target_aspect_ratio)
    if aspect_ratio is not None:
        if aspect_ratio >= 1:
            target_w = cfg.target_long_edge_px
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = cfg.target_long_edge_px
            target_w = int(round(target_h * aspect_ratio))
        target_w = max(1, target_w)
        target_h = max(1, target_h)
        ratio = target_w / float(target_h)
        return target_w, target_h, ratio

    target_w, target_h = _compute_target_size(source_width, source_height, cfg.target_long_edge_px)
    ratio = target_w / float(target_h)
    return target_w, target_h, ratio


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
    caption_text: str | None = None,
    caption_font_path: str | None = None,
) -> str:
    """Upscale an image to a 4K-ish target long edge.

    Returns the output_path.

    Behavior:
      - If the input already matches the target size, the image is copied to
        output_path.
      - If the input meets/exceeds the target size, we resize/crop directly to
        the requested dimensions (no upscaling).
      - Otherwise we run an x4 Real-ESRGAN model once, then resize/crop
        (Lanczos) to the exact target dimensions.
      - If the Real-ESRGAN binary is unavailable and allow_fallback_resize is
        True, we fall back to a Lanczos resize/crop to the target dimensions.
      - If caption_text is provided, the caption overlay is applied after the
        final resize/crop so it remains visible in the output aspect ratio.
    """

    cfg = config or UpscaleConfig()

    in_path = Path(input_path)
    out_path = Path(output_path)

    def _finalize_output() -> str:
        if caption_text:
            with Image.open(out_path) as im:
                overlaid = _overlay_caption(im, caption_text, font_path=caption_font_path)
                ext = out_path.suffix.lower()
                fmt = "PNG" if ext == ".png" else "JPEG"
                save_kwargs: dict[str, object] = {"format": fmt}
                if fmt == "JPEG":
                    save_kwargs.update({"quality": 95, "optimize": True})
                overlaid.save(out_path, **save_kwargs)
        return str(out_path)

    if (
        cfg.target_width_px is not None
        and cfg.target_height_px is not None
        and cfg.target_aspect_ratio is not None
    ):
        raise ValueError(
            "Provide either explicit target_width_px/target_height_px or target_aspect_ratio, not both."
        )
    if not in_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as im:
        width, height = im.size

    has_explicit_size = cfg.target_width_px is not None and cfg.target_height_px is not None
    aspect_ratio = _normalize_aspect_ratio(cfg.target_aspect_ratio)

    target_w, target_h, _ = _resolve_target_dimensions(
        source_width=width,
        source_height=height,
        cfg=cfg,
    )

    if width == target_w and height == target_h:
        shutil.copyfile(in_path, out_path)
        return _finalize_output()

    if not has_explicit_size and aspect_ratio is None and max(width, height) >= cfg.target_long_edge_px:
        shutil.copyfile(in_path, out_path)
        return _finalize_output()

    needs_upscale = width < target_w or height < target_h

    if cfg.engine != "realesrgan-ncnn-vulkan":
        raise ValueError(f"Unsupported upscaling engine: {cfg.engine}")

    # If we already meet or exceed the target size, avoid the Real-ESRGAN hop and
    # just resize/crop to the requested dimensions.
    if not needs_upscale:
        _lanczos_resize(in_path, out_path, target_w, target_h)
        return _finalize_output()

    binary = _find_realesrgan_binary(cfg.realesrgan_binary)
    if not binary:
        if cfg.allow_fallback_resize:
            _lanczos_resize(in_path, out_path, target_w, target_h)
            return _finalize_output()
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

    return _finalize_output()


def _lanczos_resize(src: Path, dst: Path, width: int, height: int) -> None:
    """Resize using Pillow's Lanczos filter and save as JPEG/PNG based on dst."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resize target: {width}x{height}")

    with Image.open(src) as im:
        im = im.convert("RGB")
        # ImageOps.fit preserves aspect ratio and crops (center) when ratios differ.
        resized = ImageOps.fit(
            im,
            (int(width), int(height)),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

        ext = dst.suffix.lower().lstrip(".")
        if ext in {"jpg", "jpeg"}:
            resized.save(dst, format="JPEG", quality=95, optimize=True)
        elif ext == "png":
            resized.save(dst, format="PNG", optimize=True)
        else:
            # Default to JPEG to keep output size reasonable.
            resized.save(dst, format="JPEG", quality=95, optimize=True)
