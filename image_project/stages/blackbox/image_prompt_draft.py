from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from image_project.stages.blackbox._profiles import resolve_profile_text
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.image_prompt_draft"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    final_profile_source = cfg.get_str(
        "final_profile_source",
        default="raw",
        choices=("raw", "generator_hints", "generator_hints_plus_dislikes"),
    )
    if final_profile_source is None:
        raise ValueError("blackbox.image_prompt_draft.final_profile_source cannot be null")
    temperature = cfg.get_float("temperature", default=0.8, min_value=0.0, max_value=2.0)

    def _prompt(ctx: RunContext) -> str:
        selected_card = ctx.outputs.get("selected_idea_card")
        if not isinstance(selected_card, dict):
            raise ValueError("Missing required output: selected_idea_card")
        return prompts.draft_prompt_from_selected_idea_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=resolve_profile_text(
                ctx,
                source=final_profile_source,
                stage_id=KIND_ID,
                config_path=f"{cfg.path}.final_profile_source",
            ),
            selected_idea_card=selected_card,
        )

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=float(temperature),
        merge="none",
        step_capture_key="blackbox_draft_image_prompt",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Create a draft prompt from selected idea card (for downstream refinement).",
    source="prompts.blackbox.draft_prompt_from_selected_idea_prompt",
    tags=("blackbox",),
    kind="chat",
    io=StageIO(
        requires=("selected_idea_card",),
        provides=("blackbox_draft_image_prompt",),
        captures=("blackbox_draft_image_prompt",),
    ),
)
