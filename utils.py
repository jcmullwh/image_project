import base64
import csv
import io
import os
import uuid
from datetime import datetime

import requests
import yaml
from PIL import Image, ImageDraw, ImageFont


def _load_caption_font(font_size_px: int, font_path: str | None = None) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size_px)
        except Exception:
            pass

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

    print(f"Data saved to CSV for ID: {data[0]}")


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
    
    # get parameters from kwargs
    config_file = kwargs.get('config_name','config')    
    config_filetype = kwargs.get('config_type','.yaml')
    config_relative_path = kwargs.get('config_rel_path','config')
    
    # make the path to the config file
    config_filename = config_file + config_filetype
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_directory = os.path.join(project_root, config_relative_path)
    config_path = os.path.join(config_directory,config_filename)
    
    #load config file
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        raise
    
def download_and_convert_image(image_url, image_full_path_and_name):
    try:
        if isinstance(image_url, str) and image_url.startswith("http"):
            response = requests.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
        else:
            image_bytes = base64.b64decode(image_url)

        image = Image.open(io.BytesIO(image_bytes))
        rgb_image = image.convert('RGB')  # Convert to RGB in case the PNG is in RGBA format

        # Save the image as JPG
        rgb_image.save(image_full_path_and_name, format='JPEG')
        print(f"Image saved to {image_full_path_and_name}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")

def save_image(
    image_bytes,
    image_full_path_and_name,
    caption_text: str | None = None,
    caption_font_path: str | None = None,
):
    try:

        image = Image.open(io.BytesIO(image_bytes))
        rgb_image = image.convert('RGB')  # Convert to RGB in case the PNG is in RGBA format

        if caption_text:
            rgb_image = _overlay_caption(rgb_image, caption_text, font_path=caption_font_path)

        # Save the image as JPG
        rgb_image.save(image_full_path_and_name, format='JPEG')
        print(f"Image saved to {image_full_path_and_name}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
        raise
        
def generate_file_location(file_path, id,file_type):
    try:
        # Combine save_path and id to create the complete file path
        file_path = os.path.join(file_path, id + file_type)
        return file_path
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
