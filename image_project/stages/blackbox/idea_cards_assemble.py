from __future__ import annotations

import json
from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.idea_cards_assemble"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    from image_project.framework.scoring import expected_idea_ids

    num_ideas = cfg.get_int("num_ideas", default=6, min_value=1)
    idea_ids = expected_idea_ids(int(num_ideas))

    def _action(ctx: RunContext) -> str:
        from image_project.framework import scoring as blackbox_scoring

        ideas: list[dict[str, Any]] = []
        for idea_id in idea_ids:
            key = f"blackbox.idea_card.{idea_id}.json"
            raw = ctx.outputs.get(key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Missing required output: {key}")
            try:
                ideas.append(blackbox_scoring.parse_idea_card_json(raw, expected_id=idea_id))
            except Exception as exc:
                setattr(exc, "pipeline_step", f"blackbox.idea_card_generate.{idea_id}")
                setattr(exc, "pipeline_path", f"pipeline/blackbox.idea_card_generate.{idea_id}/draft")
                raise

        return json.dumps({"ideas": ideas}, ensure_ascii=False, indent=2)

    cfg.assert_consumed()
    return make_action_stage_block(
        instance_id,
        fn=_action,
        merge="none",
        step_capture_key="idea_cards_json",
        tags=("blackbox",),
        doc="Assemble per-idea JSON outputs into a combined idea_cards_json payload.",
        source="framework.scoring.parse_idea_card_json",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Assemble isolated per-idea JSON artifacts into idea_cards_json.",
    source="framework.scoring.parse_idea_card_json",
    tags=("blackbox",),
    kind="action",
    io=StageIO(
        provides=("idea_cards_json",),
        captures=("idea_cards_json",),
    ),
)
