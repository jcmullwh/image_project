from __future__ import annotations

import json
import os
from typing import Any

from pipeline import RunContext


def write_transcript(path: str, ctx: RunContext) -> None:
    payload: dict[str, Any] = {
        "generation_id": ctx.generation_id,
        "seed": ctx.seed,
        "selected_concepts": list(ctx.selected_concepts),
        "steps": list(ctx.steps),
        "image_path": ctx.image_path,
        "created_at": ctx.created_at,
    }
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
    if ctx.error is not None:
        payload["error"] = ctx.error

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
