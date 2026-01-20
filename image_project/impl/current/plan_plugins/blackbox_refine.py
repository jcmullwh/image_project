from __future__ import annotations

from image_project.framework.prompt_pipeline import PlanInputs
from image_project.impl.current.plans import register_plan
from image_project.stages.blackbox.generate_idea_cards import STAGE as BLACKBOX_GENERATE_IDEA_CARDS
from image_project.stages.blackbox.generator_profile_hints import (
    STAGE as BLACKBOX_GENERATOR_PROFILE_HINTS,
)
from image_project.stages.blackbox.idea_cards_judge_score import (
    STAGE as BLACKBOX_IDEA_CARDS_JUDGE_SCORE,
)
from image_project.stages.blackbox.prepare import STAGE as BLACKBOX_PREPARE
from image_project.stages.blackbox.select_idea_card import STAGE as BLACKBOX_SELECT_IDEA_CARD
from image_project.stages.blackbox_refine.loop import STAGE as BLACKBOX_REFINE_LOOP
from image_project.stages.blackbox_refine.seed_from_draft import (
    STAGE as BLACKBOX_REFINE_SEED_FROM_DRAFT,
)
from image_project.stages.blackbox_refine.seed_prompt import (
    STAGE as BLACKBOX_REFINE_SEED_PROMPT,
)
from image_project.stages.postprompt.openai_format import STAGE as POSTPROMPT_OPENAI_FORMAT
from image_project.stages.postprompt.profile_nudge import STAGE as POSTPROMPT_PROFILE_NUDGE
from image_project.stages.preprompt.filter_concepts import STAGE as PREPROMPT_FILTER_CONCEPTS
from image_project.stages.preprompt.select_concepts import STAGE as PREPROMPT_SELECT_CONCEPTS
from pipelinekit.stage_types import StageInstance


@register_plan
class BlackboxRefinePromptPlan:
    name = "blackbox_refine"
    requires_scoring = True

    def stage_nodes(self, inputs: PlanInputs) -> list[StageInstance]:
        base: list[StageInstance] = [
            PREPROMPT_SELECT_CONCEPTS.instance(),
            PREPROMPT_FILTER_CONCEPTS.instance(),
            BLACKBOX_PREPARE.instance(),
            BLACKBOX_GENERATOR_PROFILE_HINTS.instance(),
        ]

        base.extend(
            [
                BLACKBOX_GENERATE_IDEA_CARDS.instance(),
                BLACKBOX_IDEA_CARDS_JUDGE_SCORE.instance(),
                BLACKBOX_SELECT_IDEA_CARD.instance(),
            ]
        )

        base.append(BLACKBOX_REFINE_SEED_PROMPT.instance())

        return [
            *base,
            BLACKBOX_REFINE_LOOP.instance(),
            POSTPROMPT_PROFILE_NUDGE.instance(),
            POSTPROMPT_OPENAI_FORMAT.instance(),
        ]


@register_plan
class BlackboxRefineOnlyPromptPlan:
    name = "blackbox_refine_only"
    requires_scoring = True
    required_inputs = ("draft_prompt",)

    def stage_nodes(self, inputs: PlanInputs) -> list[StageInstance]:
        draft_text = (inputs.draft_prompt or "").strip()
        if not draft_text:
            raise ValueError(
                "prompt.plan=blackbox_refine_only requires prompt.refine_only.draft or draft_path"
            )

        specs: list[StageInstance] = [
            PREPROMPT_SELECT_CONCEPTS.instance(),
            PREPROMPT_FILTER_CONCEPTS.instance(),
            BLACKBOX_PREPARE.instance(),
            BLACKBOX_GENERATOR_PROFILE_HINTS.instance(),
        ]

        specs.append(BLACKBOX_REFINE_SEED_FROM_DRAFT.instance())

        specs.append(BLACKBOX_REFINE_LOOP.instance())

        specs.extend(
            [
                POSTPROMPT_PROFILE_NUDGE.instance(),
                POSTPROMPT_OPENAI_FORMAT.instance(),
            ]
        )

        return specs
