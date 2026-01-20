from __future__ import annotations

from image_project.framework.runtime import RunContext


def require_text_output(ctx: RunContext, key: str) -> str:
    value = ctx.outputs.get(key)
    if not isinstance(value, str):
        value = str(value or "")
    text = value.strip()
    if not text:
        raise ValueError(f"Missing required output: {key}")
    return text

