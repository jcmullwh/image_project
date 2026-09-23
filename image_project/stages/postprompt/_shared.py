from __future__ import annotations

from image_project.framework.runtime import RunContext


def resolve_latest_prompt_for_postprompt(ctx: RunContext, *, stage_id: str) -> str:
    draft = ctx.outputs.get("image_prompt")
    if isinstance(draft, str) and draft.strip():
        return draft.strip()

    beams = ctx.outputs.get("bbref.beams")
    if isinstance(beams, list) and beams:
        first = beams[0] if isinstance(beams[0], dict) else None
        prompt = (first or {}).get("prompt") if isinstance(first, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

    for record in reversed(ctx.steps):
        if not isinstance(record, dict) or record.get("type") != "chat":
            continue
        response = record.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()

    raise ValueError(f"{stage_id} could not resolve a draft image prompt")

