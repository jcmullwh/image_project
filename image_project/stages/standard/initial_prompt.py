from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import standard as prompts
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "standard.initial_prompt"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _prompt(ctx: RunContext) -> str:
        selected = list(ctx.selected_concepts)
        if not selected:
            raise ValueError(
                f"{KIND_ID} requires selected concepts; "
                "run preprompt.select_concepts first (or include it in the plan)."
            )

        prompt_1, _selected = prompts.generate_first_prompt(
            inputs.prompt_data,
            inputs.user_profile,
            inputs.rng,
            context_guidance=(inputs.context_guidance or None),
            selected_concepts=selected,
        )
        return prompt_1

    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=_prompt, temperature=0.8)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Generate candidate themes/stories.",
    source="prompts.standard.generate_first_prompt",
    tags=("standard",),
    kind="chat",
    io=StageIO(requires=("selected_concepts",)),
)
