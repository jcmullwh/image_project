from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.prompts import standard as prompts
from pipelinekit.stage_types import StageRef

KIND_ID = "standard.initial_prompt_freeform"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    prompt = prompts.standard_initial_prompt_freeform_prompt(
        preferences_guidance=inputs.preferences_guidance,
        context_guidance=inputs.context_guidance,
    )
    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=prompt, temperature=0.8)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Generate candidate themes/stories without concept selection (freeform).",
    source="prompts.standard.standard_initial_prompt_freeform_prompt",
    tags=("standard",),
    kind="chat",
)
