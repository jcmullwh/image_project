from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.profile_abstraction"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _prompt(ctx: RunContext) -> str:
        return prompts.profile_abstraction_prompt(
            preferences_guidance=str(ctx.outputs.get("preferences_guidance") or "")
        )

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=0.0,
        merge="none",
        step_capture_key="generator_profile_hints",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Create generator-safe profile hints.",
    source="prompts.blackbox.profile_abstraction_prompt",
    tags=("blackbox",),
    kind="chat",
    io=StageIO(
        provides=("generator_profile_hints",),
        captures=("generator_profile_hints",),
    ),
)
