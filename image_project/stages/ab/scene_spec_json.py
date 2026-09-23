from __future__ import annotations

import textwrap

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.stages.ab._shared import require_text_output
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "ab.scene_spec_json"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _prompt(ctx: RunContext) -> str:
        token = require_text_output(ctx, "ab_random_token")
        draft = require_text_output(ctx, "ab_scene_draft")

        return textwrap.dedent(
            f"""\
            Convert the scene draft into a strict SceneSpec JSON object.

            Required token (must appear as visible text in the scene): {token}

            Scene draft:
            {draft}

            SceneSpec schema (required keys):
            {{
              "subject": "...",
              "setting": "...",
              "action": "...",
              "composition": "...",
              "camera": "...",
              "lighting": "...",
              "color": "...",
              "style": "...",
              "text_in_scene": "{token}",
              "must_keep": ["...", "..."],
              "avoid": ["...", "..."]
            }}

            Hard requirements:
            - Output ONLY valid JSON (no markdown, no code fences, no comments).
            - No empty strings. No empty arrays.
            - "subject" must be specific and unique (avoid generic subjects like "a person", "someone", "a figure").
            - "text_in_scene" must exactly equal the required token.
            - "must_keep" and "avoid" must each contain at least 3 concrete, visual items.
            - Keep it a single coherent moment (no montages, no scene cuts).

            Self-check before output (do not include this check in the output):
            - Every required key exists and is non-empty.
            - text_in_scene matches the token exactly.
            """
        ).strip()

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=0.55,
        step_capture_key="ab_scene_spec_json",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Convert the scene draft into a strict SceneSpec JSON intermediary.",
    source="inline",
    tags=("ab",),
    kind="chat",
    io=StageIO(
        requires=("ab_random_token", "ab_scene_draft"),
        provides=("ab_scene_spec_json",),
        captures=("ab_scene_spec_json",),
    ),
)
