from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import postprompt as prompts
from image_project.stages.postprompt._shared import resolve_latest_prompt_for_postprompt
from pipelinekit.stage_types import StageRef

KIND_ID = "postprompt.openai_format"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    max_chars: int | None = None
    if inputs.cfg.prompt_blackbox_refine is not None:
        max_chars = inputs.cfg.prompt_blackbox_refine.max_prompt_chars

    def _prompt(ctx: RunContext) -> str:
        draft = resolve_latest_prompt_for_postprompt(ctx, stage_id=KIND_ID)

        nudged = ctx.outputs.get("postprompt.nudged_prompt")
        if isinstance(nudged, str) and nudged.strip():
            nudged_text = nudged.strip()
            draft_lower = draft.lower()
            nudged_lower = nudged_text.lower()

            forbidden_phrases = (
                "no people",
                "no person",
                "no humans",
                "no human",
                "no figures",
                "no figure",
                "no faces",
                "no face",
                "no bodies",
                "no body",
            )
            forbids_people = any(phrase in draft_lower for phrase in forbidden_phrases)

            if forbids_people and not any(phrase in nudged_lower for phrase in forbidden_phrases):
                nudged_text = ""

            if nudged_text:
                draft = nudged_text
        return prompts.refine_image_prompt_prompt(draft, max_chars=max_chars)

    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=_prompt, temperature=0.3)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Format the (nudged) prompt into OpenAI GPT Image 1.5 prompt text.",
    source="prompts.postprompt.refine_image_prompt_prompt",
    tags=("postprompt",),
    kind="chat",
)
