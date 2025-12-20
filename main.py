import base64
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

try:
    from ai_backend import ImageAI, TextAI
except ModuleNotFoundError:  # pragma: no cover
    ImageAI = None  # type: ignore[assignment]
    TextAI = None  # type: ignore[assignment]
from message_handling import MessageHandler
from pipeline import Block, ChatRunner, ChatStep, RunContext
from records import append_generation_row
from run_config import RunConfig
from transcript import write_transcript
from upscaling import UpscaleConfig, upscale_image_to_4k
from context_injectors import ContextManager
from titles import (
    append_manifest_row,
    generate_title,
    get_next_seq,
    manifest_lock,
    read_manifest,
    utc_now_iso8601,
)
from utils import (
    generate_file_location,
    generate_unique_id,
    load_config,
    save_image,
)

from prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_preferences_guidance,
    generate_image_prompt,
    generate_first_prompt,
    generate_fourth_prompt,
    generate_secondB_prompt,
    generate_second_prompt,
    generate_third_prompt,
    load_prompt_data,
    make_tot_enclave_block,
)

def configure_stdio_utf8():
    """Force stdout/stderr to UTF-8 so Unicode responses never crash on Windows consoles."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # If reconfigure is unavailable, continue with defaults.
        pass

def setup_operational_logger(log_dir: str, generation_id: str):
    """
    Configure a logger that writes an operational log for traceability.
    Logs go to both stdout and a UTF-8 file under the provided directory.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = generate_file_location(log_dir, f"{generation_id}_oplog", ".log")

    logger_name = f"image_project.{generation_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info("Operational logging initialized for generation %s", generation_id)
    logger.debug("Operational log file: %s", log_file)

    return logger, log_file


def upload_to_photos_via_rclone(
    file_path: str,
    remote: str = "gphotos",
    album: str = "Generated-Art",
    logger: logging.Logger | None = None,
) -> bool:
    dest = f"{remote}:album/{album}"
    if logger:
        logger.info("Uploading %s to %s via rclone", file_path, dest)
    try:
        subprocess.run(["rclone", "copy", file_path, dest], check=True)
    except FileNotFoundError as exc:
        if logger:
            logger.error("Upload skipped: rclone not found (%s).", exc)
        return False
    except subprocess.CalledProcessError as exc:
        if logger:
            logger.error("Upload failed (rclone exit=%s). Continuing.", exc.returncode)
        return False
    except Exception:
        if logger:
            logger.exception("Upload failed due to unexpected error. Continuing.")
        return False

    return True



GENERATION_CSV_FIELDNAMES: list[str] = [
    "generation_id",
    "selected_concepts",
    "final_image_prompt",
    "image_path",
    "created_at",
    "seed",
]


def run_generation(cfg_dict, *, generation_id: str | None = None) -> RunContext:
    cfg, cfg_warnings = RunConfig.from_dict(cfg_dict)

    generation_id = generation_id or generate_unique_id()
    logger, operational_log_path = setup_operational_logger(cfg.log_dir, generation_id)
    logger.info("Run started for generation %s", generation_id)

    for warning in cfg_warnings:
        logger.warning("%s", warning)

    ctx: RunContext | None = None
    transcript_path: str | None = None
    phase = "init"

    try:
        phase = "seed"
        seed = cfg.random_seed
        if seed is None:
            seed = int(time.time())
            logger.info("No prompt.random_seed configured; generated seed=%d", seed)
        else:
            logger.info("Using configured prompt.random_seed=%d", seed)

        rng = random.Random(seed)

        phase = "data_load"
        logger.info("Loading prompt data from %s", cfg.categories_path)
        prompt_data = load_prompt_data(cfg.categories_path)
        logger.info("Loaded %d category rows", len(prompt_data))

        logger.info("Loading user profile from %s", cfg.profile_path)
        user_profile = pd.read_csv(cfg.profile_path)
        logger.info("Loaded %d user profile rows", len(user_profile))
        preferences_guidance = build_preferences_guidance(user_profile)

        phase = "context_injection"
        context_guidance_text, context_metadata = ContextManager.build(
            enabled=cfg.context_enabled,
            injectors=cfg.context_injectors,
            context_cfg=cfg.context_cfg,
            seed=seed,
            today=datetime.now().date(),
            preferences_guidance=preferences_guidance,
            logger=logger,
        )

        phase = "prompt_generation"
        prompt_1, selected_concepts = generate_first_prompt(
            prompt_data,
            user_profile,
            rng,
            context_guidance=(context_guidance_text or None),
        )
        logger.info("Generated first prompt (selected_concepts=%d)", len(selected_concepts))

        phase = "init_text_ai"
        text_ai_cls = TextAI
        if text_ai_cls is None:  # pragma: no cover
            from ai_backend import TextAI as text_ai_cls  # type: ignore[assignment]

        ai_text = text_ai_cls(model="gpt-5.2", reasoning={"effort": "medium"})
        logger.info("Initialized TextAI with model gpt-5.2")

        system_prompt = DEFAULT_SYSTEM_PROMPT
        if context_guidance_text:
            system_prompt = DEFAULT_SYSTEM_PROMPT + "\n\n" + context_guidance_text

        phase = "context"
        ctx = RunContext(
            generation_id=generation_id,
            cfg=cfg,
            logger=logger,
            rng=rng,
            seed=seed,
            created_at=utc_now_iso8601(),
            messages=MessageHandler(system_prompt),
        )
        ctx.selected_concepts = selected_concepts
        ctx.outputs["preferences_guidance"] = preferences_guidance
        if context_guidance_text:
            ctx.outputs["context_guidance"] = context_guidance_text
        if context_metadata:
            ctx.outputs["context"] = context_metadata

        transcript_path = generate_file_location(cfg.log_dir, generation_id + "_transcript", ".json")

        phase = "pipeline"
        def refined_stage(stage_step: ChatStep, *, capture_key: str | None = None) -> Block:
            stage_name = stage_step.name
            if stage_name is None:
                raise ValueError("Stage steps must have names for stable pipeline paths")
            if stage_step.capture_key:
                raise ValueError(
                    f"Stage step {stage_name} must not set capture_key; pass capture_key to refined_stage()"
                )
            draft_step = ChatStep(
                name="draft",
                prompt=stage_step.prompt,
                temperature=stage_step.temperature,
                allow_empty_prompt=stage_step.allow_empty_prompt,
                allow_empty_response=stage_step.allow_empty_response,
                params=dict(stage_step.params),
            )
            enclave = make_tot_enclave_block(stage_name)
            return Block(
                name=stage_name,
                merge="last_response",
                nodes=[draft_step, enclave],
                capture_key=capture_key,
            )

        pipeline_root = Block(
            name="pipeline",
            merge="all_messages",
            nodes=[
                refined_stage(ChatStep(name="initial_prompt", prompt=prompt_1, temperature=0.8)),
                refined_stage(
                    ChatStep(
                        name="section_2_choice",
                        prompt=lambda _ctx: generate_second_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="section_2b_title_and_story",
                        prompt=lambda _ctx: generate_secondB_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="section_3_message_focus",
                        prompt=lambda _ctx: generate_third_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="section_4_concise_description",
                        prompt=lambda _ctx: generate_fourth_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="image_prompt_creation",
                        prompt=lambda _ctx: generate_image_prompt(),
                        temperature=0.8,
                    ),
                    capture_key="image_prompt",
                ),
            ],
        )

        runner = ChatRunner(ai_text=ai_text)
        runner.run(ctx, pipeline_root)

        phase = "capture"
        image_prompt = ctx.outputs.get("image_prompt")
        if not image_prompt:
            raise ValueError("Pipeline did not produce required output: image_prompt")

        phase = "init_image_ai"
        image_ai_cls = ImageAI
        if image_ai_cls is None:  # pragma: no cover
            from ai_backend import ImageAI as image_ai_cls  # type: ignore[assignment]

        ai_image = image_ai_cls()
        logger.info("Initialized ImageAI")

        image_model = "gpt-image-1.5"
        image_size = "1536x1024"
        image_quality = "high"

        os.makedirs(cfg.generation_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.upscale_dir, exist_ok=True)

        phase = "image_pipeline"
        upload_target_path = ""
        seq = -1
        title_result = None

        with manifest_lock(cfg.titles_manifest_path):
            manifest_rows = read_manifest(cfg.titles_manifest_path)
            existing_titles = [row.get("title", "") for row in manifest_rows if row.get("title")]
            existing_titles = list(reversed(existing_titles))

            seq = get_next_seq(cfg.titles_manifest_path)
            title_result = generate_title(
                ai_text=ai_text,
                image_prompt=image_prompt,
                avoid_titles=existing_titles,
                logger=logger,
            )
            ctx.outputs["title_generation"] = {
                "title": title_result.title,
                "title_source": getattr(title_result, "title_source", None),
                "title_raw": getattr(title_result, "title_raw", None),
                "attempts": list(getattr(title_result, "attempts", ())),
            }

            caption_text = f"#{seq:03d} - {title_result.title}"
            logger.info("Assigned image identifier %s", caption_text)

            image_result = ai_image.generate_image(
                image_prompt,
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

            image_full_path_and_name = generate_file_location(
                cfg.generation_dir, generation_id + "_image", ".jpg"
            )
            save_image(
                image_bytes,
                image_full_path_and_name,
                caption_text=caption_text,
                caption_font_path=cfg.caption_font_path,
            )
            logger.info("Saved image to %s", image_full_path_and_name)
            upload_target_path = image_full_path_and_name
            ctx.image_path = upload_target_path

            if cfg.upscale_enabled:
                upscale_out_path = generate_file_location(
                    cfg.upscale_dir,
                    generation_id + "_image_4k",
                    ".jpg",
                )

                upscale_cfg = UpscaleConfig(
                    target_long_edge_px=cfg.upscale_target_long_edge_px,
                    engine=cfg.upscale_engine,
                    realesrgan_binary=cfg.upscale_realesrgan_binary,
                    model_name=cfg.upscale_model_name,
                    model_path=cfg.upscale_model_path,
                    tile_size=cfg.upscale_tile_size,
                    tta=cfg.upscale_tta,
                    allow_fallback_resize=cfg.upscale_allow_fallback_resize,
                )

                logger.info(
                    "Upscaling enabled: engine=%s model=%s target_long_edge_px=%d",
                    upscale_cfg.engine,
                    upscale_cfg.model_name,
                    upscale_cfg.target_long_edge_px,
                )
                upscale_image_to_4k(
                    input_path=image_full_path_and_name,
                    output_path=upscale_out_path,
                    config=upscale_cfg,
                )
                logger.info("Saved 4K upscaled image to %s", upscale_out_path)
                upload_target_path = upscale_out_path
                ctx.image_path = upload_target_path

            append_manifest_row(
                cfg.titles_manifest_path,
                {
                    "seq": int(seq),
                    "title": title_result.title,
                    "generation_id": generation_id,
                    "image_prompt": image_prompt,
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
            logger.info("Appended manifest row to %s (seq=%d)", cfg.titles_manifest_path, seq)

        phase = "records"
        append_generation_row(
            cfg.generations_csv_path,
            {
                "generation_id": generation_id,
                "selected_concepts": json.dumps(selected_concepts, ensure_ascii=False),
                "final_image_prompt": image_prompt,
                "image_path": upload_target_path,
                "created_at": ctx.created_at,
                "seed": seed,
            },
            GENERATION_CSV_FIELDNAMES,
        )
        logger.info("Appended generation row to %s", cfg.generations_csv_path)

        phase = "rclone"
        if cfg.rclone_enabled:
            remote = cfg.rclone_remote
            album = cfg.rclone_album
            if not remote or not album:
                raise ValueError("rclone.enabled=true but rclone.remote/album missing")

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

        phase = "transcript"
        write_transcript(transcript_path, ctx)
        logger.info("Wrote transcript JSON to %s", transcript_path)
        logger.info("Operational log stored at %s", operational_log_path)
        logger.info("Run completed successfully for generation %s", generation_id)

        return ctx
    except Exception as exc:
        logger.exception("Run failed during phase %s", phase)
        if ctx is not None:
            step_name = getattr(exc, "pipeline_step", None)
            pipeline_path = getattr(exc, "pipeline_path", None)
            ctx.error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "phase": phase,
            }
            if step_name:
                ctx.error["step"] = step_name
            if pipeline_path:
                ctx.error["path"] = pipeline_path

            try:
                if transcript_path is None:
                    transcript_path = generate_file_location(
                        cfg.log_dir, generation_id + "_transcript", ".json"
                    )
                write_transcript(transcript_path, ctx)
                logger.info("Wrote transcript JSON to %s", transcript_path)
            except Exception:
                logger.exception("Failed to write transcript during error handling")
        raise
    finally:
        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers.clear()


def main() -> None:
    configure_stdio_utf8()
    generation_id = generate_unique_id()

    try:
        cfg_dict = load_config()
    except Exception:
        # This is a rare exeception to the "no fallbacks" rule. Fallbacks are ONLY acceptable here to prevent loss of logging data
        fallback_logger, fallback_log_path = setup_operational_logger(os.getcwd(), generation_id)
        fallback_logger.exception(
            "Failed to load configuration. Logging to fallback file at %s", fallback_log_path
        )
        raise

    run_generation(cfg_dict, generation_id=generation_id)


if __name__ == "__main__":
    main()
