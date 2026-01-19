from __future__ import annotations

from image_project.framework.prompt_pipeline import PlanInputs
from image_project.stages.blackbox.idea_card_generate import STAGE as BLACKBOX_IDEA_CARD_GENERATE
from image_project.stages.blackbox.idea_cards_assemble import (
    STAGE as BLACKBOX_IDEA_CARDS_ASSEMBLE,
)
from pipelinekit.stage_types import StageInstance


def build_blackbox_isolated_idea_card_instances(inputs: PlanInputs) -> list[StageInstance]:
    from image_project.framework.scoring import expected_idea_ids

    scoring_cfg = inputs.cfg.prompt_scoring
    idea_ids = expected_idea_ids(scoring_cfg.num_ideas)

    instances: list[StageInstance] = [
        BLACKBOX_IDEA_CARD_GENERATE.instance(f"{BLACKBOX_IDEA_CARD_GENERATE.id}.{idea_id}")
        for idea_id in idea_ids
    ]
    instances.append(BLACKBOX_IDEA_CARDS_ASSEMBLE.instance())
    return instances
