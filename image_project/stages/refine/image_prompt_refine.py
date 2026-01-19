from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.prompts import postprompt as prompts
from pipelinekit.stage_types import StageRef

KIND_ID = "refine.image_prompt_refine"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    draft_text = (inputs.draft_prompt or "").strip()
    if not draft_text:
        raise ValueError(
            "stage refine.image_prompt_refine requires inputs.draft_prompt (prompt.plan=refine_only)"
        )

    prompt = prompts.refine_image_prompt_prompt(draft_text)
    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=prompt, temperature=0.8)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Refine a provided draft into the final image prompt.",
    source="prompts.postprompt.refine_image_prompt_prompt",
    tags=("refine",),
    kind="chat",
)
