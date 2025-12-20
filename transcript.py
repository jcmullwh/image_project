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
    if ctx.error is not None:
        payload["error"] = ctx.error

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
