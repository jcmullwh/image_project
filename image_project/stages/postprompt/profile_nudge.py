from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import postprompt as prompts
from image_project.stages.postprompt._shared import resolve_latest_prompt_for_postprompt
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "postprompt.profile_nudge"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    max_chars = cfg.get_optional_int("max_prompt_chars", default=None, min_value=1)

    def _prompt(ctx: RunContext) -> str:
        draft = resolve_latest_prompt_for_postprompt(ctx, stage_id=KIND_ID)
        preferences = str(ctx.outputs.get("preferences_guidance") or "").strip()
        context_guidance = str(ctx.outputs.get("context_guidance") or "").strip() or None
        return prompts.profile_nudge_image_prompt_prompt(
            draft_prompt=draft,
            preferences_guidance=preferences,
            context_guidance=context_guidance,
            max_chars=max_chars,
        )

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=0.0,
        merge="none",
        step_capture_key="postprompt.nudged_prompt",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Nudge the latest image prompt toward the user profile (small changes only).",
    source="prompts.postprompt.profile_nudge_image_prompt_prompt",
    tags=("postprompt",),
    kind="chat",
    io=StageIO(
        provides=("postprompt.nudged_prompt",),
        captures=("postprompt.nudged_prompt",),
    ),
)
