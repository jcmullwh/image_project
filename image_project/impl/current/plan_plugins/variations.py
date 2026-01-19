from __future__ import annotations

from image_project.impl.current.plans import (
    SequencePromptPlan,
    StandardPromptPlan,
    register_plan,
)
from image_project.stages.direct.image_prompt_creation import STAGE as DIRECT_IMAGE_PROMPT_CREATION
from image_project.stages.postprompt.openai_format import STAGE as POSTPROMPT_OPENAI_FORMAT
from image_project.stages.postprompt.profile_nudge import STAGE as POSTPROMPT_PROFILE_NUDGE
from image_project.stages.preprompt.filter_concepts import STAGE as PREPROMPT_FILTER_CONCEPTS
from image_project.stages.preprompt.select_concepts import STAGE as PREPROMPT_SELECT_CONCEPTS
from image_project.stages.standard.image_prompt_creation import (
    STAGE as STANDARD_IMAGE_PROMPT_CREATION,
)
from image_project.stages.standard.initial_prompt import STAGE as STANDARD_INITIAL_PROMPT
from image_project.stages.standard.initial_prompt_freeform import (
    STAGE as STANDARD_INITIAL_PROMPT_FREEFORM,
)


@register_plan
class BaselinePromptPlan(SequencePromptPlan):
    """One-stage baseline: capture the first stage output."""

    name = "baseline"
    sequence = (
        PREPROMPT_SELECT_CONCEPTS,
        PREPROMPT_FILTER_CONCEPTS,
        STANDARD_INITIAL_PROMPT,
    )


@register_plan
class SimplePromptPlan(SequencePromptPlan):
    """Two-stage pipeline: initial prompt + final image prompt creation."""

    name = "simple"
    sequence = (
        PREPROMPT_SELECT_CONCEPTS,
        PREPROMPT_FILTER_CONCEPTS,
        STANDARD_INITIAL_PROMPT,
        STANDARD_IMAGE_PROMPT_CREATION,
    )


@register_plan
class SimpleNoConceptsPromptPlan(SequencePromptPlan):
    """Two-stage pipeline, but skips concept selection and filtering."""

    name = "simple_no_concepts"
    sequence = (
        STANDARD_INITIAL_PROMPT_FREEFORM,
        STANDARD_IMAGE_PROMPT_CREATION,
    )


@register_plan
class DirectPromptPlan(SequencePromptPlan):
    """One-stage final prompt creation from concepts + profile."""

    name = "direct"
    sequence = (
        PREPROMPT_SELECT_CONCEPTS,
        PREPROMPT_FILTER_CONCEPTS,
        DIRECT_IMAGE_PROMPT_CREATION,
        POSTPROMPT_PROFILE_NUDGE,
        POSTPROMPT_OPENAI_FORMAT,
    )


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
