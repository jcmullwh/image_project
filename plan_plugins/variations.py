from __future__ import annotations

from pipeline import RunContext
from prompt_plans import PlanInputs, SequencePromptPlan, StageNodeSpec, StageSpec, StandardPromptPlan, register_plan


def _freeform_first_prompt(*, preferences_guidance: str, context_guidance: str | None) -> str:
    preferences = (preferences_guidance or "").strip()
    preferences_block = ""
    if preferences:
        preferences_block = f"\n\nPreferences guidance (authoritative):\n{preferences}\n"

    context_block = ""
    rendered_context = (context_guidance or "").strip()
    if rendered_context:
        if rendered_context.lower().startswith("context guidance"):
            context_block = f"\n\n{rendered_context}\n"
        else:
            context_block = f"\n\nContext guidance (optional):\n{rendered_context}\n"

    preamble = (
        "The enclave's job is to describe an art piece (some form of image, painting, photography, still-frame, etc. displayed in 1792x1024 resolution) "
        "for a specific human, 'Lana'."
        + preferences_block
        + context_block
        + "\nCreate an art piece for Lana."
    )

    return (
        preamble
        + "\n\nWhat are four possible central themes or stories of the art piece and what important messages are each trying to tell the viewer? "
        "Ensure that your choices are highly sophisticated and nuanced, well integrated with the viewer's preferences and deeply meaningful to the viewer. "
        "Ensure that it is what an AI Artist would find meaningful and important to convey to a human viewer. "
        "Ensure that the themes and stories are not similar to each other. Ensure that they are not too abstract or conceptual. "
        "Finally, ensure that they are not boring, cliche, trite, overdone, obvious, or most importantly: milquetoast. Say something and say it with conviction."
    )


@register_plan
class BaselinePromptPlan(SequencePromptPlan):
    """One-stage baseline: capture the first stage output."""

    name = "baseline"
    sequence = (
        "preprompt.select_concepts",
        "preprompt.filter_concepts",
        "standard.initial_prompt",
    )


@register_plan
class SimplePromptPlan(SequencePromptPlan):
    """Two-stage pipeline: initial prompt + final image prompt creation."""

    name = "simple"
    sequence = (
        "preprompt.select_concepts",
        "preprompt.filter_concepts",
        "standard.initial_prompt",
        "standard.image_prompt_creation",
    )


@register_plan
class SimpleNoConceptsPromptPlan(SequencePromptPlan):
    """Two-stage pipeline, but skips concept selection and filtering."""

    name = "simple_no_concepts"

    def stage_specs(self, inputs: PlanInputs) -> list[StageNodeSpec]:
        from stage_catalog import StageCatalog

        def _prompt(_ctx: RunContext) -> str:
            return _freeform_first_prompt(
                preferences_guidance=inputs.preferences_guidance,
                context_guidance=inputs.context_guidance,
            )

        return [
            StageSpec(
                stage_id="standard.initial_prompt",
                prompt=_prompt,
                temperature=0.8,
                tags=("standard",),
                doc="Generate candidate themes/stories (no concept preprompt).",
                source="plan_plugins.variations._freeform_first_prompt",
            ),
            StageCatalog.build("standard.image_prompt_creation", inputs),
        ]


@register_plan
class ProfileOnlyPromptPlan(StandardPromptPlan):
    """
    Standard pipeline, but disables context injection regardless of `context.enabled`.
    """

    name = "profile_only"
    context_injection = "disabled"


@register_plan
class ProfileOnlySimplePromptPlan(SimplePromptPlan):
    """Simple pipeline, but disables context injection regardless of `context.enabled`."""

    name = "profile_only_simple"
    context_injection = "disabled"
