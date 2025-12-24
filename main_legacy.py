from __future__ import annotations

import base64
import os
import random

import pandas as pd

from ai_backend import ImageAI, TextAI
from logging_utils import write_messages_log
from message_handling import MessageHandler
from titles import (
    append_manifest_row,
    generate_title,
    get_next_seq,
    manifest_lock,
    read_manifest,
    utc_now_iso8601,
)
from upscaling import UpscaleConfig, upscale_image_to_4k
from utils import (
    generate_file_location,
    generate_unique_id,
    load_config,
    save_image,
    save_to_csv,
)

from legacy_main import message_with_log

from main import (
    configure_stdio_utf8,
    enclave_opinion,
    generate_dalle_prompt,
    generate_fifth_prompt,
    generate_first_prompt,
    generate_fourth_prompt,
    generate_secondB_prompt,
    generate_second_prompt,
    generate_thirdB_prompt,
    generate_third_prompt,
    load_prompt_data,
    setup_operational_logger,
    upload_to_photos_via_rclone,
)


def main_legacy() -> None:
    """
    Legacy orchestration preserved for reference.

    Prefer `main.run_generation()` for the current step-driven pipeline.
    """

    configure_stdio_utf8()
    generation_id = generate_unique_id()
    try:
        config, _cfg_meta = load_config()
    except Exception:
        fallback_logger, fallback_log_path = setup_operational_logger(os.getcwd(), generation_id)
        fallback_logger.exception(
            "Failed to load configuration. Logging to fallback file at %s", fallback_log_path
        )
        raise

    image_cfg = config.get("image", {}) or {}
    if not image_cfg:
        raise ValueError("Missing required 'image' configuration section")

    def require_path(key: str) -> str:
        value = image_cfg.get(key)
        if not value:
            raise ValueError(f"Missing required config value image.{key}")
        return value

    generation_dir = require_path("generation_path")
    upscale_dir = require_path("upscale_path")
    log_dir = require_path("log_path")

    prompt_cfg = config.get("prompt", {}) or {}
    manifest_path = prompt_cfg.get("titles_manifest_path") or os.path.join(
        generation_dir, "titles_manifest.csv"
    )
    caption_font_path = image_cfg.get("caption_font_path")

    log_full_path_and_name = generate_file_location(log_dir, generation_id + "_log", ".txt")
    messages_text = ""
    logger, operational_log_path = setup_operational_logger(log_dir, generation_id)
    logger.info("Run started for generation %s", generation_id)

    try:
        categories_path = config["prompt"]["categories_path"]
        categories_names = config["prompt"]["categories_names"]
        _ = categories_names

        logger.info("Loading prompt data from %s", categories_path)
        prompt_data = load_prompt_data(categories_path)
        logger.info(
            "Loaded %d category rows with columns: %s",
            len(prompt_data),
            prompt_data.columns.tolist(),
        )

        profile_path = config["prompt"]["profile_path"]
        logger.info("Loading user profile from %s", profile_path)
        user_profile = pd.read_csv(profile_path)
        logger.info("Loaded %d user profile rows", len(user_profile))

        rng = random.Random()
        prompt_1, gen_keywords = generate_first_prompt(prompt_data, user_profile, rng)
        print("First Prompt:\n", prompt_1)
        logger.info("Generated first prompt with %d keyword selections", len(gen_keywords))

        ai_text = TextAI(model="gpt-5.2", reasoning={"effort": "medium"})
        logger.info("Initialized TextAI with model gpt-5.2")

        user_role = "user"
        agent_role = "assistant"
        system_prompt = (
            "You are a highly skilled enclave of Artists trained to generate meaningful, edgy, artistic images on "
            "par with the greatest artists of any time, anywhere, past or future, Earth or any other planet. The "
            "enclave invents unique images that weave together seemingly disparate elements into cohesive wholes "
            "that push boundaries and elicit deep emotions in a human viewer."
        )

        messages_main = MessageHandler(system_prompt)
        messages_main.continue_messages(user_role, prompt_1)
        logger.info("Sending initial prompt to AI")
        try:
            response_1 = ai_text.text_chat(messages_main.messages, temperature=0.8)
        except Exception:
            logger.exception("Initial prompt failed")
            raise
        print("First Response:\n", response_1)
        logger.info("Received initial response (chars=%d)", len(str(response_1)))
        messages_main.continue_messages(agent_role, response_1)
        # initialize messages_log
        messages_main, messages_log, _ = message_with_log(
            ai_text,
            messages_main,
            messages_main.copy(),
            enclave_opinion(),
            agent_role,
            user_role,
            logger=logger,
            step_name="enclave_opinion_initial",
            temperature=0.8,
        )

        print("SECTION 2-----------------------------------------")
        second_prompt = generate_second_prompt()
        messages_main, messages_log, _ = message_with_log(
            ai_text,
            messages_main,
            messages_log,
            second_prompt,
            agent_role,
            user_role,
            logger=logger,
            step_name="section_2_choice",
            temperature=0.8,
        )

        print("SECTION 2B-----------------------------------------")
        secondB_prompt = generate_secondB_prompt()
        messages_main, messages_log, _ = message_with_log(
            ai_text,
            messages_main,
            messages_log,
            secondB_prompt,
            agent_role,
            user_role,
            logger=logger,
            step_name="section_2b_title_and_story",
            temperature=0.8,
        )

        print("SECTION 3-----------------------------------------")
        third_prompt = generate_third_prompt()
        messages_main, messages_log, _ = message_with_log(
            ai_text,
            messages_main,
            messages_log,
            third_prompt,
            agent_role,
            user_role,
            logger=logger,
            step_name="section_3_message_focus",
            temperature=0.8,
        )

        print("SECTION 3B-----------------------------------------")
        thirdB_prompt = generate_thirdB_prompt()
        messages_main, messages_log, _ = message_with_log(
            ai_text,
            messages_main,
            messages_log,
            thirdB_prompt,
            agent_role,
            user_role,
            logger=logger,
            step_name="section_3b_message_clarity",
            temperature=0.8,
        )

        print("SECTION 4-----------------------------------------")
        fourth_prompt = generate_fourth_prompt()
        messages_main, messages_log, _ = message_with_log(
            ai_text,
            messages_main,
            messages_log,
            fourth_prompt,
            agent_role,
            user_role,
            logger=logger,
            step_name="section_4_concise_description",
            temperature=0.8,
        )

        print("DALLE-----------------------------------------")
        dalle_context = generate_dalle_prompt()
        messages, messages_log, dalle_prompt = message_with_log(
            ai_text,
            messages_main,
            messages_log,
            dalle_context,
            agent_role,
            user_role,
            logger=logger,
            step_name="dalle_prompt_creation",
            temperature=0.8,
        )
        _ = messages

        print("SECTION 5-----------------------------------------")
        fifth_prompt = generate_fifth_prompt()
        messages_main, messages_log, _ = message_with_log(
            ai_text,
            messages_main,
            messages_log,
            fifth_prompt,
            agent_role,
            user_role,
            logger=logger,
            step_name="section_5_midjourney_refine",
            temperature=0.8,
        )

        ai_image = ImageAI()
        logger.info("Initialized ImageAI")

        image_model = "gpt-image-1.5"
        image_size = "1536x1024"
        image_quality = "high"

        upload_target_path = ""
        seq = -1
        title_result = None

        try:
            with manifest_lock(manifest_path):
                manifest_rows = read_manifest(manifest_path)
                existing_titles = [row.get("title", "") for row in manifest_rows if row.get("title")]
                existing_titles = list(reversed(existing_titles))

                seq = get_next_seq(manifest_path)
                title_result = generate_title(
                    ai_text=ai_text,
                    image_prompt=dalle_prompt,
                    avoid_titles=existing_titles,
                )

                caption_text = f"#{seq:03d} - {title_result.title}"
                logger.info("Assigned image identifier %s", caption_text)

                image_result = ai_image.generate_image(
                    dalle_prompt,
                    model=image_model,
                    size=image_size,
                    quality=image_quality,
                    moderation="low",
                )
                logger.info(
                    "Image generation request sent (model=%s, size=%s, quality=%s)",
                    image_model,
                    image_size,
                    image_quality,
                )

                image_data = image_result["image"]
                logger.debug("Received image payload length: %d", len(image_data))

                image_bytes = base64.b64decode(image_data)

                data = [generation_id, gen_keywords, dalle_prompt]
                csv_file = config["prompt"]["generations_path"]
                save_to_csv(data, csv_file)
                logger.info("Saved generation metadata to %s", csv_file)

                os.makedirs(generation_dir, exist_ok=True)
                os.makedirs(log_dir, exist_ok=True)
                os.makedirs(upscale_dir, exist_ok=True)

                image_full_path_and_name = generate_file_location(
                    generation_dir, generation_id + "_image", ".jpg"
                )
                save_image(
                    image_bytes,
                    image_full_path_and_name,
                    caption_text=caption_text,
                    caption_font_path=caption_font_path,
                )
                logger.info("Saved image to %s", image_full_path_and_name)

                upload_target_path = image_full_path_and_name

                upscale_section = config.get("upscale", {}) or {}
                if upscale_section.get("enabled", False):
                    upscale_out_path = generate_file_location(
                        upscale_dir,
                        generation_id + "_image_4k",
                        ".jpg",
                    )

                    upscale_cfg = UpscaleConfig(
                        target_long_edge_px=int(upscale_section.get("target_long_edge_px", 3840)),
                        target_width_px=(
                            int(upscale_section["target_width_px"])
                            if "target_width_px" in upscale_section
                            and upscale_section.get("target_width_px") not in (None, "")
                            else None
                        ),
                        target_height_px=(
                            int(upscale_section["target_height_px"])
                            if "target_height_px" in upscale_section
                            and upscale_section.get("target_height_px") not in (None, "")
                            else None
                        ),
                        target_aspect_ratio=upscale_section.get("target_aspect_ratio"),
                        engine=str(upscale_section.get("engine", "realesrgan-ncnn-vulkan")),
                        realesrgan_binary=upscale_section.get("realesrgan_binary"),
                        model_name=str(upscale_section.get("model_name", "realesrgan-x4plus")),
                        model_path=upscale_section.get("model_path"),
                        tile_size=int(upscale_section.get("tile_size", 0)),
                        tta=bool(upscale_section.get("tta", False)),
                        allow_fallback_resize=bool(upscale_section.get("allow_fallback_resize", False)),
                    )

                    if upscale_cfg.target_width_px and upscale_cfg.target_height_px:
                        target_desc = f"{upscale_cfg.target_width_px}x{upscale_cfg.target_height_px}"
                    elif upscale_cfg.target_aspect_ratio:
                        target_desc = (
                            f"long_edge={upscale_cfg.target_long_edge_px} "
                            f"aspect={upscale_cfg.target_aspect_ratio}"
                        )
                    else:
                        target_desc = f"long_edge={upscale_cfg.target_long_edge_px} (preserve aspect)"

                    logger.info(
                        "Upscaling enabled: engine=%s model=%s target=%s",
                        upscale_cfg.engine,
                        upscale_cfg.model_name,
                        target_desc,
                    )
                    upscale_image_to_4k(
                        input_path=image_full_path_and_name,
                        output_path=upscale_out_path,
                        config=upscale_cfg,
                    )
                    logger.info("Saved 4K upscaled image to %s", upscale_out_path)
                    upload_target_path = upscale_out_path

                append_manifest_row(
                    manifest_path,
                    {
                        "seq": int(seq),
                        "title": title_result.title,
                        "generation_id": generation_id,
                        "image_prompt": dalle_prompt,
                        "image_path": upload_target_path,
                        "created_at": utc_now_iso8601(),
                        "model": image_model,
                        "size": image_size,
                        "quality": image_quality,
                        "seed": image_result.get("seed", ""),
                        "title_source": getattr(title_result, "title_source", "llm"),
                        "title_raw": getattr(title_result, "title_raw", ""),
                    },
                )
                logger.info("Appended manifest row to %s (seq=%d)", manifest_path, seq)
        except Exception:
            logger.exception("Generation failed during title/manifest/image pipeline")
            raise

        rclone_config = config.get("rclone", {}) or {}
        if rclone_config.get("enabled", False):
            remote = rclone_config.get("remote")
            album = rclone_config.get("album")
            if not remote or not album:
                raise ValueError("Rclone upload enabled but 'remote' or 'album' is missing in config.rclone")
            if not upload_target_path:
                logger.error("Rclone upload enabled but upload target path is empty; skipping upload.")
            else:
                uploaded = upload_to_photos_via_rclone(
                    upload_target_path,
                    remote=remote,
                    album=album,
                    logger=logger,
                )
                if uploaded:
                    logger.info("Uploaded image via rclone to %s", f"{remote}:album/{album}")
                else:
                    logger.error("Rclone upload failed; image remains at %s", upload_target_path)

        messages_text = str(messages_log.messages)
        write_messages_log(log_full_path_and_name, messages_text)
        logger.info("Wrote message transcript to %s", log_full_path_and_name)
        logger.info("Operational log stored at %s", operational_log_path)
        logger.info("Run completed successfully for generation %s", generation_id)
    except Exception:
        logger.exception("Run failed for generation %s", generation_id)
        try:
            if "messages_log" in locals():
                messages_text = str(messages_log.messages)
            elif "messages_main" in locals():
                messages_text = str(messages_main.messages)
            else:
                messages_text = ""
            write_messages_log(log_full_path_and_name, messages_text)
        except Exception:
            logger.exception("Failed to write message transcript during error handling")
        raise


if __name__ == "__main__":
    main_legacy()
