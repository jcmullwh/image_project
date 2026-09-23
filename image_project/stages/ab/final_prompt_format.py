from __future__ import annotations

import textwrap

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.stages.ab._shared import require_text_output
from pipelinekit.stage_types import StageRef

KIND_ID = "ab.final_prompt_format"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _prompt(ctx: RunContext) -> str:
        token = require_text_output(ctx, "ab_random_token")
        refined = require_text_output(ctx, "ab_scene_refined")

        return textwrap.dedent(
            f"""\
            Convert the refined scene description into a final image generation prompt using the exact one-line format below.

            Refined scene:
            {refined}

            Output format (exactly one line; keep labels and separators):
            SUBJECT=<...> | SETTING=<...> | ACTION=<...> | COMPOSITION=<...> | CAMERA=<...> | LIGHTING=<...> | COLOR=<...> | STYLE=<...> | TEXT_IN_SCENE="{token}" | AR=16:9

            Rules:
            - Ensure TEXT_IN_SCENE uses the token exactly as provided.
            - Do not add extra lines before/after.
            """
        ).strip()

    cfg.assert_consumed()
    return make_chat_stage_block(instance_id, prompt=_prompt, temperature=0.6)


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Format the refined scene into a strict single-line prompt template.",
    source="prompts.ab_final_prompt_format",
    tags=("ab",),
    kind="chat",
)
