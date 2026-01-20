from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox_refine.seed_from_draft"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    draft_text = (inputs.draft_prompt or "").strip()
    if not draft_text:
        raise ValueError(
            "prompt.plan=blackbox_refine_only requires prompt.refine_only.draft or draft_path"
        )

    def _action(ctx: RunContext, *, draft_text=draft_text) -> str:
        ctx.logger.info("Blackbox prompt refine seed: source=draft_prompt chars=%d", len(draft_text))
        return draft_text

    cfg.assert_consumed()
    return make_action_stage_block(
        instance_id,
        fn=_action,
        merge="none",
        step_capture_key="bbref.seed_prompt",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Seed the blackbox refinement loop from prompt.refine_only.draft.",
    source="stages.blackbox_refine.seed_from_draft._build",
    tags=("blackbox_refine",),
    kind="action",
    io=StageIO(
        provides=("bbref.seed_prompt",),
        captures=("bbref.seed_prompt",),
    ),
)
