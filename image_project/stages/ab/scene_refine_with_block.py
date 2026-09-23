from __future__ import annotations

import textwrap

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.stages.ab._shared import require_text_output
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "ab.scene_refine_with_block"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _prompt(ctx: RunContext) -> str:
        token = require_text_output(ctx, "ab_random_token")
        draft = require_text_output(ctx, "ab_scene_draft")
        return textwrap.dedent(
            f"""\
            Refine the following scene draft for an image prompt.

            Required token (must remain verbatim as visible text in the scene): {token}

            Refinement block (apply silently; do not output this block):
            - Subject: make the main subject unmistakable and unique (no generic "a person").
            - Setting: make time/place concrete (materials, era, weather, geography, props).
            - Composition: specify framing, lens feel, depth of field, foreground/background.
            - Lighting: specify the dominant light source(s) and the mood they create.
            - Color: specify a palette and a couple accent colors.
            - Specificity: add 3+ grounded details (textures, signage, wear, reflections, particles).
            - Token: integrate the token diegetically (a label, screen, tag), not as metadata.
            - Cliche filter: remove stock phrases and overused tropes.

            Draft:
            {draft}

            Output:
            - Return ONLY the revised scene description (4-6 sentences). No headings, no bullets.
            """
        ).strip()

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=0.75,
        step_capture_key="ab_scene_refined",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Refine the draft scene with an explicit refinement block.",
    source="prompts.ab_scene_refine_with_block",
    tags=("ab",),
    kind="chat",
    io=StageIO(
        requires=("ab_random_token", "ab_scene_draft"),
        provides=("ab_scene_refined",),
        captures=("ab_scene_refined",),
    ),
)
