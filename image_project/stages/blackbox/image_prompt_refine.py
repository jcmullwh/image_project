from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from image_project.stages.blackbox._profiles import resolve_profile_text
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.image_prompt_refine"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    scoring_cfg = inputs.cfg.prompt_scoring
    final_profile_source = scoring_cfg.final_profile_source

    def _prompt(ctx: RunContext) -> str:
        selected_card = ctx.outputs.get("selected_idea_card")
        if not isinstance(selected_card, dict):
            raise ValueError("Missing required output: selected_idea_card")

        draft = ctx.outputs.get("blackbox_draft_image_prompt")
        if not isinstance(draft, str) or not draft.strip():
            raise ValueError("Missing required output: blackbox_draft_image_prompt")

        return prompts.refine_draft_prompt_from_selected_idea_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=resolve_profile_text(
                ctx,
                source=final_profile_source,
                stage_id=KIND_ID,
                config_path="prompt.scoring.final_profile_source",
            ),
            selected_idea_card=selected_card,
            draft_prompt=draft,
        )

    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=_prompt, temperature=0.4)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Refine the draft prompt into a final prompt (no ToT).",
    source="prompts.blackbox.refine_draft_prompt_from_selected_idea_prompt",
    tags=("blackbox",),
    kind="chat",
    io=StageIO(
        requires=("selected_idea_card", "blackbox_draft_image_prompt"),
    ),
)
