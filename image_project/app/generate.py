import base64
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

import pandas as pd

try:
    from ai_backend import ImageAI, TextAI
except ModuleNotFoundError:  # pragma: no cover
    ImageAI = None  # type: ignore[assignment]
    TextAI = None  # type: ignore[assignment]

from image_project.foundation.config_io import load_config
from image_project.foundation.messages import MessageHandler
from image_project.foundation.pipeline import ChatRunner
from image_project.framework.artifacts import (
    append_generation_row,
    append_manifest_row,
    append_run_index_entry,
    generate_file_location,
    generate_unique_id,
    generate_title,
    get_next_seq,
    manifest_lock,
    read_manifest,
    utc_now_iso8601,
)
from image_project.framework.config import RunConfig
from image_project.framework.context import ContextManager
from image_project.framework.inputs import extract_dislikes, resolve_prompt_inputs
from image_project.framework.media import UpscaleConfig, save_image, upscale_image_to_4k
from image_project.framework.prompting import PlanInputs, resolve_stage_specs
from image_project.framework.runtime import RunContext
from image_project.framework.transcript import write_transcript
from image_project.impl.current.prompting import (
    DEFAULT_SYSTEM_PROMPT,
    build_preferences_guidance,
    load_prompt_data,
)
from image_project.impl.current.plans import PromptPlanManager

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


def _maybe_write_run_review_report(
    cfg: RunConfig,
    *,
    generation_id: str,
    oplog_path: str,
    transcript_path: str,
    logger: logging.Logger,
) -> tuple[dict[str, str] | None, str | None]:
    review_cfg = getattr(cfg, "run_review", None)
    if review_cfg is None or not getattr(review_cfg, "enabled", False):
        return None, None

    output_dir = getattr(review_cfg, "review_path", None)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return None, "run-review.enabled=true requires run-review.review_path"

    try:
        from image_project.run_review.report_builder import build_report, report_to_dict
        from image_project.run_review.report_model import RunInputs
        from image_project.run_review.render_html import render_html

        os.makedirs(output_dir, exist_ok=True)
        inputs = RunInputs(generation_id, oplog_path=oplog_path, transcript_path=transcript_path)
        report = build_report(inputs, best_effort=False, enable_evolution=True)

        json_out = os.path.join(output_dir, f"{generation_id}_run_report.json")
        html_out = os.path.join(output_dir, f"{generation_id}_run_report.html")

        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(report_to_dict(report), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with open(html_out, "w", encoding="utf-8") as handle:
            handle.write(render_html(report))

        logger.info("Wrote run_review report to %s", html_out)
        logger.info("Wrote run_review report to %s", json_out)
        return {"run_report_html": html_out, "run_report_json": json_out}, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_review failed: %s", exc)
        return None, str(exc)



GENERATION_CSV_FIELDNAMES: list[str] = [
    "generation_id",
    "selected_concepts",
    "final_image_prompt",
    "image_path",
    "created_at",
    "seed",
]


def run_generation(cfg_dict, *, generation_id: str | None = None, config_meta: dict[str, Any] | None = None) -> RunContext:
    from image_project.impl.current import context_plugins as _context_plugins

    _context_plugins.discover()

    cfg, cfg_warnings = RunConfig.from_dict(cfg_dict)

    generation_id = generation_id or generate_unique_id()
    logger, operational_log_path = setup_operational_logger(cfg.log_dir, generation_id)
    run_index_path = os.path.join(cfg.log_dir, "runs_index.jsonl")

    if config_meta:
        mode = config_meta.get("mode")
        paths = config_meta.get("paths") or []
        env_var = config_meta.get("env_var") or "IMAGE_PROJECT_CONFIG"
        if mode in {"env", "explicit"} and paths:
            label = f"env {env_var}" if mode == "env" else "explicit path"
            logger.info("Loaded config from %s=%s", label, paths[0])
        elif paths:
            base = paths[0]
            local = paths[1] if len(paths) > 1 else None
            if local:
                logger.info("Loaded config base=%s local=%s", base, local)
            else:
                logger.info("Loaded config base=%s", base)

    if getattr(cfg, "experiment", None) is not None:
        exp = cfg.experiment
        if exp.id or exp.variant:
            logger.info("Experiment: id=%s variant=%s", exp.id, exp.variant)
    logger.info("Run started for generation %s", generation_id)

    for warning in cfg_warnings:
        logger.warning("%s", warning)

    ctx: RunContext | None = None
    transcript_path: str | None = None
    final_prompt_path: str | None = None
    image_path: str | None = None
    upscaled_image_path: str | None = None
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
        user_dislikes = extract_dislikes(user_profile)

        phase = "plan_resolution"
        resolved_plan = PromptPlanManager.resolve(cfg)
        requested_plan = resolved_plan.requested_plan
        plan = resolved_plan.plan
        context_injection_mode = resolved_plan.metadata.context_injection
        effective_context_enabled = resolved_plan.effective_context_enabled

        phase = "context_injection"
        context_guidance_text, context_metadata = ContextManager.build(
            enabled=effective_context_enabled,
            injectors=cfg.context_injectors,
            context_cfg=cfg.context_cfg,
            seed=seed,
            today=datetime.now().date(),
            preferences_guidance=preferences_guidance,
            logger=logger,
        )

        phase = "init_text_ai"
        text_ai_cls = TextAI
        if text_ai_cls is None:  # pragma: no cover
            from ai_backend import TextAI as text_ai_cls  # type: ignore[assignment]

        ai_text = text_ai_cls(model="gpt-5.2", reasoning={"effort": "medium"})
        logger.info("Initialized TextAI with model gpt-5.2")

        system_prompt = DEFAULT_SYSTEM_PROMPT
        if context_guidance_text and cfg.context_injection_location in ("system", "both"):
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
        ctx.outputs["preferences_guidance"] = preferences_guidance
        ctx.outputs["dislikes"] = user_dislikes
        if context_guidance_text:
            ctx.outputs["context_guidance"] = context_guidance_text
        if context_metadata:
            ctx.outputs["context"] = context_metadata

        transcript_path = generate_file_location(cfg.log_dir, generation_id + "_transcript", ".json")

        scoring_cfg = cfg.prompt_scoring
        ctx.blackbox_scoring = {
            "enabled": scoring_cfg.enabled,
            "config_snapshot": asdict(scoring_cfg),
        }

        runner = ChatRunner(ai_text=ai_text)

        phase = "pipeline"

        draft_prompt: str | None = None
        if resolved_plan.metadata.required_inputs:
            resolved_inputs = resolve_prompt_inputs(cfg, required=resolved_plan.metadata.required_inputs)
            draft_prompt = resolved_inputs.draft_prompt

        inputs = PlanInputs(
            cfg=cfg,
            ai_text=ai_text,
            prompt_data=prompt_data,
            user_profile=user_profile,
            preferences_guidance=preferences_guidance,
            context_guidance=(
                (context_guidance_text or None)
                if context_guidance_text and cfg.context_injection_location in ("prompt", "both")
                else None
            ),
            rng=rng,
            draft_prompt=draft_prompt,
        )

        stage_specs = plan.stage_specs(inputs)
        resolved_stages = resolve_stage_specs(
            stage_specs,
            plan_name=plan.name,
            include=cfg.prompt_stages_include,
            exclude=cfg.prompt_stages_exclude,
            overrides=cfg.prompt_stages_overrides,
            capture_stage=cfg.prompt_output_capture_stage,
        )

        prompt_pipeline = dict(resolved_stages.metadata)
        prompt_pipeline["requested_plan"] = requested_plan
        prompt_pipeline["refinement_policy"] = cfg.prompt_refinement_policy
        prompt_pipeline["context_injection"] = context_injection_mode
        prompt_pipeline["context_injection_location"] = cfg.context_injection_location
        prompt_pipeline["context_enabled"] = bool(effective_context_enabled)
        if {
            "blackbox.idea_cards_judge_score",
            "blackbox.image_prompt_creation",
        }.intersection(resolved_stages.stage_ids):
            prompt_pipeline["blackbox_profile_sources"] = {
                "judge_profile_source": cfg.prompt_scoring.judge_profile_source,
                "final_profile_source": cfg.prompt_scoring.final_profile_source,
            }
        ctx.outputs["prompt_pipeline"] = prompt_pipeline

        logger.info(
            "Prompt plan requested=%s resolved=%s stages=%s refinement=%s capture=%s context=%s",
            requested_plan,
            plan.name,
            list(resolved_stages.stage_ids),
            cfg.prompt_refinement_policy,
            resolved_stages.capture_stage,
            context_injection_mode,
        )

        plan.execute(ctx, runner, resolved_stages, inputs)

        phase = "capture"
        image_prompt = ctx.outputs.get("image_prompt")
        if not image_prompt:
            raise ValueError("Pipeline did not produce required output: image_prompt")

        if cfg.run_mode == "prompt_only":
            logger.info(
                "run.mode=prompt_only; skipping media pipeline (image/upscale/upload/csv)"
            )
            final_prompt_path = generate_file_location(
                cfg.log_dir, generation_id + "_final_prompt", ".txt"
            )
            with open(final_prompt_path, "w", encoding="utf-8") as handle:
                handle.write(str(image_prompt).rstrip() + "\n")
            logger.info("Wrote final prompt text artifact to %s", final_prompt_path)

            phase = "transcript"
            write_transcript(transcript_path, ctx)
            logger.info("Wrote transcript JSON to %s", transcript_path)

            run_review_artifacts, run_review_error = _maybe_write_run_review_report(
                cfg,
                generation_id=generation_id,
                oplog_path=operational_log_path,
                transcript_path=transcript_path,
                logger=logger,
            )

            entry = {
                "schema_version": 1,
                "generation_id": generation_id,
                "created_at": ctx.created_at,
                "seed": ctx.seed,
                "status": "success",
                "phase": "complete",
                "run_mode": cfg.run_mode,
                "experiment": {
                    "id": cfg.experiment.id,
                    "variant": cfg.experiment.variant,
                    "notes": cfg.experiment.notes,
                    "tags": list(cfg.experiment.tags),
                },
                "prompt_pipeline": ctx.outputs.get("prompt_pipeline"),
                "artifacts": {
                    "transcript": transcript_path,
                    "oplog": operational_log_path,
                    "final_prompt": final_prompt_path,
                    "image": None,
                    "upscaled_image": None,
                },
            }
            if run_review_artifacts:
                entry["artifacts"].update(run_review_artifacts)
            if run_review_error:
                entry["run_review_error"] = run_review_error
            if ctx.error is not None:
                entry["error"] = ctx.error
            try:
                append_run_index_entry(run_index_path, entry)
                logger.info(
                    "Appended run index entry to %s (status=%s)", run_index_path, entry["status"]
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Run index append failed: %s", exc)
                ctx.outputs["run_index_error"] = str(exc)

            logger.info("Operational log stored at %s", operational_log_path)
            logger.info("Run completed successfully for generation %s", generation_id)
            return ctx

        phase = "init_image_ai"
        image_ai_cls = ImageAI
        if image_ai_cls is None:  # pragma: no cover
            from ai_backend import ImageAI as image_ai_cls  # type: ignore[assignment]

        ai_image = image_ai_cls()
        logger.info("Initialized ImageAI")

        image_model = "gpt-image-1.5"
        image_size = "1536x1024"
        image_quality = "high"

        if not cfg.generation_dir:
            raise ValueError("Missing required config: image.generation_path")
        os.makedirs(cfg.generation_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        if cfg.upscale_enabled:
            if not cfg.upscale_dir:
                raise ValueError(
                    "Missing required config: image.upscale_path (required when upscale.enabled=true)"
                )
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
            image_path = image_full_path_and_name
            ctx.image_path = upload_target_path

            if cfg.upscale_enabled:
                assert cfg.upscale_dir is not None
                upscale_out_path = generate_file_location(
                    cfg.upscale_dir,
                    generation_id + "_image_4k",
                    ".jpg",
                )

                upscale_cfg = UpscaleConfig(
                    target_long_edge_px=cfg.upscale_target_long_edge_px,
                    target_width_px=cfg.upscale_target_width_px,
                    target_height_px=cfg.upscale_target_height_px,
                    target_aspect_ratio=cfg.upscale_target_aspect_ratio,
                    engine=cfg.upscale_engine,
                    realesrgan_binary=cfg.upscale_realesrgan_binary,
                    model_name=cfg.upscale_model_name,
                    model_path=cfg.upscale_model_path,
                    tile_size=cfg.upscale_tile_size,
                    tta=cfg.upscale_tta,
                    allow_fallback_resize=cfg.upscale_allow_fallback_resize,
                )

                if upscale_cfg.target_width_px and upscale_cfg.target_height_px:
                    target_desc = f"{upscale_cfg.target_width_px}x{upscale_cfg.target_height_px}"
                elif upscale_cfg.target_aspect_ratio:
                    target_desc = (
                        f"long_edge={upscale_cfg.target_long_edge_px} "
                        f"aspect={upscale_cfg.target_aspect_ratio:.4g}"
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
                    caption_text=caption_text,
                    caption_font_path=cfg.caption_font_path,
                )
                logger.info("Saved 4K upscaled image to %s", upscale_out_path)
                upload_target_path = upscale_out_path
                upscaled_image_path = upscale_out_path
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
                "selected_concepts": json.dumps(list(ctx.selected_concepts), ensure_ascii=False),
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

        run_review_artifacts, run_review_error = _maybe_write_run_review_report(
            cfg,
            generation_id=generation_id,
            oplog_path=operational_log_path,
            transcript_path=transcript_path,
            logger=logger,
        )

        entry = {
            "schema_version": 1,
            "generation_id": generation_id,
            "created_at": ctx.created_at,
            "seed": ctx.seed,
            "status": "success",
            "phase": "complete",
            "run_mode": cfg.run_mode,
            "experiment": {
                "id": cfg.experiment.id,
                "variant": cfg.experiment.variant,
                "notes": cfg.experiment.notes,
                "tags": list(cfg.experiment.tags),
            },
            "prompt_pipeline": ctx.outputs.get("prompt_pipeline"),
            "artifacts": {
                "transcript": transcript_path,
                "oplog": operational_log_path,
                "final_prompt": None,
                "image": image_path,
                "upscaled_image": upscaled_image_path,
            },
        }
        if run_review_artifacts:
            entry["artifacts"].update(run_review_artifacts)
        if run_review_error:
            entry["run_review_error"] = run_review_error
        if ctx.error is not None:
            entry["error"] = ctx.error
        try:
            append_run_index_entry(run_index_path, entry)
            logger.info(
                "Appended run index entry to %s (status=%s)", run_index_path, entry["status"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Run index append failed: %s", exc)
            ctx.outputs["run_index_error"] = str(exc)

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

                run_review_artifacts, run_review_error = _maybe_write_run_review_report(
                    cfg,
                    generation_id=generation_id,
                    oplog_path=operational_log_path,
                    transcript_path=transcript_path,
                    logger=logger,
                )

                entry = {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "created_at": ctx.created_at,
                    "seed": ctx.seed,
                    "status": "error",
                    "phase": ctx.error.get("phase") if isinstance(ctx.error, dict) else phase,
                    "run_mode": cfg.run_mode,
                    "experiment": {
                        "id": cfg.experiment.id,
                        "variant": cfg.experiment.variant,
                        "notes": cfg.experiment.notes,
                        "tags": list(cfg.experiment.tags),
                    },
                    "prompt_pipeline": ctx.outputs.get("prompt_pipeline"),
                    "artifacts": {
                        "transcript": transcript_path,
                        "oplog": operational_log_path,
                        "final_prompt": final_prompt_path,
                        "image": image_path,
                        "upscaled_image": upscaled_image_path,
                    },
                }
                if run_review_artifacts:
                    entry["artifacts"].update(run_review_artifacts)
                if run_review_error:
                    entry["run_review_error"] = run_review_error
                if ctx.error is not None:
                    entry["error"] = ctx.error
                try:
                    append_run_index_entry(run_index_path, entry)
                    logger.info(
                        "Appended run index entry to %s (status=%s)",
                        run_index_path,
                        entry["status"],
                    )
                except Exception as run_index_exc:  # noqa: BLE001
                    logger.exception("Run index append failed: %s", run_index_exc)
                    ctx.outputs["run_index_error"] = str(run_index_exc)
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
        cfg_dict, cfg_meta = load_config()
    except Exception:
        # This is a rare exeception to the "no fallbacks" rule. Fallbacks are ONLY acceptable here to prevent loss of logging data
        fallback_logger, fallback_log_path = setup_operational_logger(os.getcwd(), generation_id)
        fallback_logger.exception(
            "Failed to load configuration. Logging to fallback file at %s", fallback_log_path
        )
        raise

    run_generation(cfg_dict, generation_id=generation_id, config_meta=cfg_meta)


if __name__ == "__main__":
    main()
