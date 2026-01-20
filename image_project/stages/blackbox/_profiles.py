from __future__ import annotations

from image_project.framework.runtime import RunContext


def resolve_profile_text(
    ctx: RunContext,
    *,
    source: str,
    stage_id: str,
    config_path: str,
) -> str:
    if source == "raw":
        raw_profile = str(ctx.outputs.get("preferences_guidance") or "").strip()
        return raw_profile
    if source == "generator_hints":
        hints = ctx.outputs.get("generator_profile_hints")
        if not isinstance(hints, str) or not hints.strip():
            raise ValueError(
                f"{stage_id} requires generator_profile_hints for {config_path}=generator_hints"
            )
        return hints
    if source == "generator_hints_plus_dislikes":
        hints = ctx.outputs.get("generator_profile_hints")
        if not isinstance(hints, str) or not hints.strip():
            raise ValueError(
                f"{stage_id} requires generator_profile_hints for {config_path}=generator_hints_plus_dislikes"
            )

        dislikes_raw = ctx.outputs.get("dislikes")
        if dislikes_raw is None:
            dislikes = []
        elif isinstance(dislikes_raw, list):
            dislikes = [str(v).strip() for v in dislikes_raw if str(v).strip()]
        else:
            raise ValueError(
                f"{stage_id} requires dislikes to be a list for {config_path}=generator_hints_plus_dislikes"
            )

        dislikes_block = "\n".join(f"- {item}" for item in dislikes) if dislikes else "- <none>"

        return (
            "Profile extraction (generator-safe hints):\n"
            + hints.strip()
            + "\n\nDislikes:\n"
            + dislikes_block
        ).strip()

    raise ValueError(f"Unknown profile source for {config_path}: {source}")

