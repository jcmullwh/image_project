from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "direct.image_prompt_creation"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _prompt(ctx: RunContext) -> str:
        if not ctx.selected_concepts:
            raise ValueError(
                f"{KIND_ID} requires selected concepts; "
                "run preprompt.select_concepts first (or include it in the plan)."
            )

        return prompts.final_prompt_from_concepts_and_profile_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=str(ctx.outputs.get("preferences_guidance") or ""),
        )

    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=_prompt, temperature=0.8)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Create final prompt directly from concepts + profile.",
    source="prompts.blackbox.final_prompt_from_concepts_and_profile_prompt",
    tags=("direct",),
    kind="chat",
    io=StageIO(requires=("selected_concepts",)),
)
