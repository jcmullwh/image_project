from __future__ import annotations

from image_project.framework.prompt_pipeline import PlanInputs
from image_project.impl.current.blackbox_idea_cards import build_blackbox_isolated_idea_card_instances
from image_project.stages.blackbox_refine.loop import build_blackbox_refine_loop_instances
from image_project.impl.current.plans import register_plan
from image_project.stages.blackbox.idea_cards_judge_score import (
    STAGE as BLACKBOX_IDEA_CARDS_JUDGE_SCORE,
)
from image_project.stages.blackbox.prepare import STAGE as BLACKBOX_PREPARE
from image_project.stages.blackbox.profile_abstraction import (
    STAGE as BLACKBOX_PROFILE_ABSTRACTION,
)
from image_project.stages.blackbox.profile_hints_load import (
    STAGE as BLACKBOX_PROFILE_HINTS_LOAD,
)
from image_project.stages.blackbox.select_idea_card import STAGE as BLACKBOX_SELECT_IDEA_CARD
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
        scoring_cfg = inputs.cfg.prompt_scoring
        if not scoring_cfg.enabled:
            raise ValueError("prompt.plan=blackbox_refine requires prompt.scoring.enabled=true")

        base: list[StageInstance] = [
            PREPROMPT_SELECT_CONCEPTS.instance(),
            PREPROMPT_FILTER_CONCEPTS.instance(),
            BLACKBOX_PREPARE.instance(),
        ]
        if scoring_cfg.generator_profile_hints_path:
            base.append(BLACKBOX_PROFILE_HINTS_LOAD.instance())
        elif scoring_cfg.generator_profile_abstraction:
            base.append(BLACKBOX_PROFILE_ABSTRACTION.instance())

        base.extend(
            [
                *build_blackbox_isolated_idea_card_instances(inputs),
                BLACKBOX_IDEA_CARDS_JUDGE_SCORE.instance(),
                BLACKBOX_SELECT_IDEA_CARD.instance(),
            ]
        )

        base.append(BLACKBOX_REFINE_SEED_PROMPT.instance())

        loop_instances = build_blackbox_refine_loop_instances(inputs)

        return [
            *base,
            *loop_instances,
            POSTPROMPT_PROFILE_NUDGE.instance(),
            POSTPROMPT_OPENAI_FORMAT.instance(),
        ]


@register_plan
class BlackboxRefineOnlyPromptPlan:
    name = "blackbox_refine_only"
    requires_scoring = True
    required_inputs = ("draft_prompt",)

    def stage_nodes(self, inputs: PlanInputs) -> list[StageInstance]:
        scoring_cfg = inputs.cfg.prompt_scoring
        if not scoring_cfg.enabled:
            raise ValueError("prompt.plan=blackbox_refine_only requires prompt.scoring.enabled=true")

        draft_text = (inputs.draft_prompt or "").strip()
        if not draft_text:
            raise ValueError(
                "prompt.plan=blackbox_refine_only requires prompt.refine_only.draft or draft_path"
            )

        specs: list[StageInstance] = [
            PREPROMPT_SELECT_CONCEPTS.instance(),
            PREPROMPT_FILTER_CONCEPTS.instance(),
            BLACKBOX_PREPARE.instance(),
        ]
        if scoring_cfg.generator_profile_hints_path:
            specs.append(BLACKBOX_PROFILE_HINTS_LOAD.instance())
        elif scoring_cfg.generator_profile_abstraction:
            specs.append(BLACKBOX_PROFILE_ABSTRACTION.instance())

        specs.append(BLACKBOX_REFINE_SEED_FROM_DRAFT.instance())

        specs.extend(build_blackbox_refine_loop_instances(inputs))

        specs.extend(
            [
                POSTPROMPT_PROFILE_NUDGE.instance(),
                POSTPROMPT_OPENAI_FORMAT.instance(),
            ]
        )

        return specs
