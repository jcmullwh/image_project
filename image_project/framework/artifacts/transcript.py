from __future__ import annotations

import json
import os
from typing import Any

from image_project.framework.runtime import RunContext


def write_transcript(path: str, ctx: RunContext) -> None:
    prompt_inputs = ctx.outputs.get("prompt_inputs")
    categories_path: str | None = None
    profile_path: str | None = None
    if isinstance(prompt_inputs, dict):
        raw_categories = prompt_inputs.get("categories_path")
        raw_profile = prompt_inputs.get("profile_path")
        if isinstance(raw_categories, str) and raw_categories.strip():
            categories_path = raw_categories
        if isinstance(raw_profile, str) and raw_profile.strip():
            profile_path = raw_profile

    payload: dict[str, Any] = {
        "generation_id": ctx.generation_id,
        "seed": ctx.seed,
        "selected_concepts": list(ctx.selected_concepts),
        "steps": list(ctx.steps),
        "image_path": ctx.image_path,
        "created_at": ctx.created_at,
    }
    if categories_path:
        payload["categories_file"] = os.path.basename(categories_path)
    if profile_path:
        payload["profile_file"] = os.path.basename(profile_path)
    if isinstance(prompt_inputs, dict) and prompt_inputs:
        payload["prompt_inputs"] = dict(prompt_inputs)
    experiment = getattr(ctx.cfg, "experiment", None)
    if experiment is not None:
        exp_payload = {
            "id": getattr(experiment, "id", None),
            "variant": getattr(experiment, "variant", None),
            "notes": getattr(experiment, "notes", None),
            "tags": list(getattr(experiment, "tags", ()) or ()),
        }
        if exp_payload["id"] or exp_payload["variant"] or exp_payload["notes"] or exp_payload["tags"]:
            payload["experiment"] = exp_payload
    final_image_prompt = ctx.outputs.get("image_prompt")
    if isinstance(final_image_prompt, str) and final_image_prompt.strip():
        payload["final_image_prompt"] = final_image_prompt
    if ctx.blackbox_scoring is not None:
        payload["blackbox_scoring"] = ctx.blackbox_scoring
    context = ctx.outputs.get("context")
    if context is not None:
        payload["context"] = context
    concept_filter_log = ctx.outputs.get("concept_filter_log")
    if concept_filter_log is not None:
        payload["concept_filter_log"] = concept_filter_log
    title_generation = ctx.outputs.get("title_generation")
    if title_generation is not None:
        payload["title_generation"] = title_generation

    prompt_pipeline = ctx.outputs.get("prompt_pipeline")
    if prompt_pipeline is not None:
        payload["outputs"] = {"prompt_pipeline": prompt_pipeline}
    if ctx.error is not None:
        payload["error"] = ctx.error

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
