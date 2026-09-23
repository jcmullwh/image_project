from __future__ import annotations

import textwrap

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "ab.scene_draft"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _prompt(ctx: RunContext) -> str:
        token = ctx.outputs.get("ab_random_token")
        token_text = str(token).strip() if token is not None else ""
        if not token_text:
            raise ValueError("Missing required output: ab_random_token")

        return textwrap.dedent(
            f"""\
            You are drafting a cinematic scene description that will later be converted into an image generation prompt.

            Required token: {token_text}

            Requirements:
            - Include the token verbatim as visible text in the scene (e.g., on a sign, label, screen, tattoo, receipt).
            - Describe a single coherent moment (no montages).
            - Be concrete and visual: subject, setting, action, lighting, mood, camera/framing.
            - Avoid cliches and generic phrasing.

            Output:
            - Return ONLY the scene description as 4-6 sentences. No headings, no bullets.
            """
        ).strip()

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=0.85,
        step_capture_key="ab_scene_draft",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Create a scene draft from a random token.",
    source="prompts.ab_scene_draft",
    tags=("ab",),
    kind="chat",
    io=StageIO(
        requires=("ab_random_token",),
        provides=("ab_scene_draft",),
        captures=("ab_scene_draft",),
    ),
)
