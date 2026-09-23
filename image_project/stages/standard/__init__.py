from __future__ import annotations

from image_project.stages.standard.image_prompt_creation import STAGE as IMAGE_PROMPT_CREATION
from image_project.stages.standard.initial_prompt import STAGE as INITIAL_PROMPT
from image_project.stages.standard.initial_prompt_freeform import STAGE as INITIAL_PROMPT_FREEFORM
from image_project.stages.standard.section_2_choice import STAGE as SECTION_2_CHOICE
from image_project.stages.standard.section_2b_title_and_story import STAGE as SECTION_2B_TITLE_AND_STORY
from image_project.stages.standard.section_3_message_focus import STAGE as SECTION_3_MESSAGE_FOCUS
from image_project.stages.standard.section_4_concise_description import STAGE as SECTION_4_CONCISE_DESCRIPTION

__all_stages__ = [
    INITIAL_PROMPT,
    INITIAL_PROMPT_FREEFORM,
    SECTION_2_CHOICE,
    SECTION_2B_TITLE_AND_STORY,
    SECTION_3_MESSAGE_FOCUS,
    SECTION_4_CONCISE_DESCRIPTION,
    IMAGE_PROMPT_CREATION,
]
