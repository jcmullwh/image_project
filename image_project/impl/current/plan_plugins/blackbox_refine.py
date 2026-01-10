from __future__ import annotations

from image_project.framework.blackbox_refine_loop import build_blackbox_refine_loop_specs
from image_project.framework.prompting import (
    ActionStageSpec,
    LinearStagePlan,
    PlanInputs,
    StageNodeSpec,
    StageSpec,
)
from image_project.framework.runtime import RunContext
from image_project.impl.current.plans import register_plan
from image_project.impl.current.prompting import StageCatalog, build_blackbox_isolated_idea_card_specs


@register_plan
class BlackboxRefinePromptPlan(LinearStagePlan):
    name = "blackbox_refine"
    requires_scoring = True

    def stage_specs(self, inputs: PlanInputs) -> list[StageNodeSpec]:
        scoring_cfg = inputs.cfg.prompt_scoring
        if not scoring_cfg.enabled:
            raise ValueError("prompt.plan=blackbox_refine requires prompt.scoring.enabled=true")

        base: list[StageNodeSpec] = [
            StageCatalog.build("preprompt.select_concepts", inputs),
            StageCatalog.build("preprompt.filter_concepts", inputs),
            StageCatalog.build("blackbox.prepare", inputs),
        ]
        if scoring_cfg.generator_profile_hints_path:
            base.append(StageCatalog.build("blackbox.profile_hints_load", inputs))
        elif scoring_cfg.generator_profile_abstraction:
            base.append(StageCatalog.build("blackbox.profile_abstraction", inputs))

        base.extend(
            [
                *build_blackbox_isolated_idea_card_specs(inputs),
                StageCatalog.build("blackbox.idea_cards_judge_score", inputs),
                StageCatalog.build("blackbox.select_idea_card", inputs),
            ]
        )

        seed_stage = StageCatalog.build("blackbox.image_prompt_creation", inputs)
        if not isinstance(seed_stage, StageSpec):
            raise TypeError("blackbox.image_prompt_creation must be a chat stage")

        base.append(
            StageSpec(
                stage_id=seed_stage.stage_id,
                prompt=seed_stage.prompt,
                temperature=seed_stage.temperature,
                params=dict(seed_stage.params),
                allow_empty_prompt=seed_stage.allow_empty_prompt,
                allow_empty_response=seed_stage.allow_empty_response,
                tags=tuple(seed_stage.tags),
                refinement_policy="none",
                is_default_capture=False,
                merge="none",
                output_key="bbref.seed_prompt",
                doc=seed_stage.doc,
                source=seed_stage.source,
            )
        )

        loop_specs = build_blackbox_refine_loop_specs(
            inputs,
            seed_output_key="bbref.seed_prompt",
            seed_source="blackbox",
        )

        return [
            *base,
            *loop_specs,
            StageCatalog.build("postprompt.profile_nudge", inputs),
            StageCatalog.build("postprompt.openai_format", inputs),
        ]


@register_plan
class BlackboxRefineOnlyPromptPlan(LinearStagePlan):
    name = "blackbox_refine_only"
    requires_scoring = True
    required_inputs = ("draft_prompt",)

    def stage_specs(self, inputs: PlanInputs) -> list[StageNodeSpec]:
        scoring_cfg = inputs.cfg.prompt_scoring
        if not scoring_cfg.enabled:
            raise ValueError("prompt.plan=blackbox_refine_only requires prompt.scoring.enabled=true")

        draft_text = (inputs.draft_prompt or "").strip()
        if not draft_text:
            raise ValueError(
                "prompt.plan=blackbox_refine_only requires prompt.refine_only.draft or draft_path"
            )

        specs: list[StageNodeSpec] = [
            StageCatalog.build("preprompt.select_concepts", inputs),
            StageCatalog.build("preprompt.filter_concepts", inputs),
            StageCatalog.build("blackbox.prepare", inputs),
        ]
        if scoring_cfg.generator_profile_hints_path:
            specs.append(StageCatalog.build("blackbox.profile_hints_load", inputs))
        elif scoring_cfg.generator_profile_abstraction:
            specs.append(StageCatalog.build("blackbox.profile_abstraction", inputs))

        def _seed_action(ctx: RunContext, *, draft_text=draft_text) -> str:
            ctx.logger.info("Blackbox prompt refine seed: source=draft_prompt chars=%d", len(draft_text))
            return draft_text

        specs.append(
            ActionStageSpec(
                stage_id="blackbox_refine.seed_from_draft",
                fn=_seed_action,
                merge="none",
                output_key="bbref.seed_prompt",
                tags=("blackbox_refine",),
            )
        )

        specs.extend(
            build_blackbox_refine_loop_specs(
                inputs,
                seed_output_key="bbref.seed_prompt",
                seed_source="draft",
            )
        )

        specs.extend(
            [
                StageCatalog.build("postprompt.profile_nudge", inputs),
                StageCatalog.build("postprompt.openai_format", inputs),
            ]
        )

        return specs
