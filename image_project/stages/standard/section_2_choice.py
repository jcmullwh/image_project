from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.prompts import standard as prompts
from pipelinekit.stage_types import StageRef

KIND_ID = "standard.section_2_choice"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=lambda _ctx: prompts.generate_second_prompt(),
        temperature=0.8,
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Pick the most compelling choice.",
    source="prompts.standard.generate_second_prompt",
    tags=("standard",),
    kind="chat",
)
