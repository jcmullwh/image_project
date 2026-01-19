from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from image_project.stages.blackbox._profiles import resolve_profile_text
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.image_prompt_creation"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    scoring_cfg = inputs.cfg.prompt_scoring
    final_profile_source = scoring_cfg.final_profile_source
    context_guidance = inputs.context_guidance or None

    def _prompt(ctx: RunContext) -> str:
        selected_card = ctx.outputs.get("selected_idea_card")
        if not isinstance(selected_card, dict):
            raise ValueError("Missing required output: selected_idea_card")
        return prompts.final_prompt_from_selected_idea_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=resolve_profile_text(
                ctx,
                source=final_profile_source,
                stage_id=KIND_ID,
                config_path="prompt.scoring.final_profile_source",
            ),
            selected_idea_card=selected_card,
            context_guidance=context_guidance,
        )

    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=_prompt, temperature=0.8)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Create final prompt from selected idea card.",
    source="prompts.blackbox.final_prompt_from_selected_idea_prompt",
    tags=("blackbox",),
    kind="chat",
    io=StageIO(
        requires=("selected_idea_card",),
    ),
)
