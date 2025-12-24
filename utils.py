import base64
import csv
import io
import os
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import requests
import yaml
from PIL import Image, ImageDraw, ImageFont


def _load_caption_font(font_size_px: int, font_path: str | None = None) -> ImageFont.ImageFont:
    if font_path is not None:
        try:
            return ImageFont.truetype(font_path, font_size_px)
        except Exception as exc:
            raise ValueError(f"Failed to load caption font from {font_path!r}: {exc}") from exc

    # Try common bundled/system fonts, then fall back to Pillow's default bitmap font.
    candidates: list[str] = []
    try:
        import PIL  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        pil_dir = Path(PIL.__file__).resolve().parent
        candidates.extend(
            [
                str(pil_dir / "fonts" / "DejaVuSans.ttf"),
                str(pil_dir / "fonts" / "DejaVuSans-Bold.ttf"),
                str(pil_dir / "DejaVuSans.ttf"),
                str(pil_dir / "DejaVuSans-Bold.ttf"),
            ]
        )
    except Exception:
        pass

    candidates.extend(
        [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    )

    for path in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size_px)
        except Exception:
            continue

    return ImageFont.load_default()


def _overlay_caption(image: Image.Image, caption_text: str, font_path: str | None = None) -> Image.Image:
    if not caption_text:
        return image

    base = image.convert("RGBA")
    width, height = base.size

    # Subtle "gallery caption" strip: ~4% image height.
    strip_height = max(int(height * 0.04), 18)
    padding_x = max(int(width * 0.02), 10)
    padding_y = max(int(strip_height * 0.20), 4)

    font_size = max(int(height * 0.028), 14)
    font = _load_caption_font(font_size, font_path=font_path)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Fit text (defensive): reduce font size if needed.
    def text_bbox(fnt: ImageFont.ImageFont) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), caption_text, font=fnt)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    text_w, text_h = text_bbox(font)
    while text_w > (width - 2 * padding_x) and font_size > 10:
        font_size = max(font_size - 2, 10)
        font = _load_caption_font(font_size, font_path=font_path)
        text_w, text_h = text_bbox(font)

    strip_height = max(strip_height, text_h + 2 * padding_y)

    strip_top = height - strip_height
    draw.rectangle([(0, strip_top), (width, height)], fill=(0, 0, 0, 110))

    x = (width - text_w) // 2
    y = strip_top + (strip_height - text_h) // 2

    # Subtle shadow for readability.
    draw.text((x + 1, y + 1), caption_text, font=font, fill=(0, 0, 0, 160))
    draw.text((x, y), caption_text, font=font, fill=(255, 255, 255, 230))

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return composed

def save_to_csv(data, csv_file, include_headers=True):
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write headers if the file is being created and headers are needed
        if not file_exists and include_headers:
            writer.writerow(['ID', 'Description Prompt', 'Generation Prompt', 'Image URL'])

        writer.writerow(data)


def generate_unique_id():
    # Generate a random UUID
    unique_id = uuid.uuid4()

    # Optionally, add a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a filename with UUID (and timestamp)
    filename = f"{timestamp}_{unique_id}"
    return filename


def load_config(**kwargs):
    """
    Load configuration settings from a YAML file.

    This function reads configuration settings from a specified YAML file.
    It's used to setup parameters such as data directory, file patterns, and required fields for data ingestion.

    :param config_name: The name of the configuration file (without the file extension). Defaults to 'config'.
    :type config_name: str
    :return: The configuration settings.
    :rtype: dict
    :raises Exception: If there is an issue in reading or parsing the YAML file.
    """

    config_path_override = kwargs.get("config_path")
    env_var = kwargs.get("env_var", "IMAGE_PROJECT_CONFIG")
    config_file = kwargs.get("config_name", "config")
    config_filetype = kwargs.get("config_type", ".yaml")
    config_relative_path = kwargs.get("config_rel_path", "config")

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_directory = os.path.join(project_root, config_relative_path)
    base_config_path = os.path.join(config_directory, config_file + config_filetype)
    local_overlay_path = os.path.join(config_directory, "config.local.yaml")

    def _load_yaml_mapping(path: str) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        except FileNotFoundError:
            raise
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        except Exception:
            raise

        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        return dict(payload)

    def _deep_merge(base: Any, overlay: Any, *, path: str) -> Any:
        if overlay is None:
            return None

        if base is None:
            return overlay

        if isinstance(base, Mapping):
            if not isinstance(overlay, Mapping):
                raise ValueError(
                    f"Invalid config overlay merge at {path}: base is mapping but overlay is {type(overlay).__name__}"
                )
            merged: dict[str, Any] = dict(base)
            for key, overlay_value in overlay.items():
                next_path = f"{path}.{key}" if path else str(key)
                if key in base:
                    merged[key] = _deep_merge(base[key], overlay_value, path=next_path)
                else:
                    merged[key] = overlay_value
            return merged

        if isinstance(base, (list, tuple)):
            if not isinstance(overlay, (list, tuple)):
                raise ValueError(
                    f"Invalid config overlay merge at {path}: base is list but overlay is {type(overlay).__name__}"
                )
            return list(overlay)

        if isinstance(overlay, (Mapping, list, tuple)):
            raise ValueError(
                f"Invalid config overlay merge at {path}: base is {type(base).__name__} but overlay is {type(overlay).__name__}"
            )

        return overlay

    # Env override (or explicit config_path) loads a single file (no local overlay).
    explicit_path = None
    if config_path_override is not None:
        explicit_path = str(config_path_override).strip() or None
    elif env_var:
        raw_env = os.environ.get(str(env_var), "")
        explicit_path = raw_env.strip() or None

    if explicit_path:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(explicit_path)))
        cfg = _load_yaml_mapping(expanded)
        meta = {"mode": "env" if config_path_override is None else "explicit", "paths": [expanded], "env_var": env_var}
        return cfg, meta

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Missing base config file: {base_config_path}")

    cfg = _load_yaml_mapping(base_config_path)
    loaded_paths = [os.path.abspath(base_config_path)]
    mode = "base"

    if os.path.exists(local_overlay_path):
        overlay = _load_yaml_mapping(local_overlay_path)
        cfg = _deep_merge(cfg, overlay, path="")
        loaded_paths.append(os.path.abspath(local_overlay_path))
        mode = "base+local"

    meta = {"mode": mode, "paths": loaded_paths, "env_var": env_var}
    return cfg, meta
    
def download_and_convert_image(image_url, image_full_path_and_name):
    if isinstance(image_url, str) and image_url.startswith("http"):
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = response.content
    else:
        image_bytes = base64.b64decode(image_url)

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")  # Convert to RGB in case the PNG is in RGBA format
        rgb_image.save(image_full_path_and_name, format="JPEG")

def save_image(
    image_bytes,
    image_full_path_and_name,
    caption_text: str | None = None,
    caption_font_path: str | None = None,
):
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")  # Convert to RGB in case the PNG is in RGBA format

        if caption_text:
            rgb_image = _overlay_caption(rgb_image, caption_text, font_path=caption_font_path)

        # Save the image as JPG
        rgb_image.save(image_full_path_and_name, format="JPEG")
        
def generate_file_location(file_path, id,file_type):
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")
    if not id or not isinstance(id, str):
        raise ValueError("id must be a non-empty string")
    if not file_type or not isinstance(file_type, str):
        raise ValueError("file_type must be a non-empty string")

    return os.path.join(file_path, id + file_type)
