from __future__ import annotations

from image_project.stages.blackbox.idea_card_generate import STAGE as IDEA_CARD_GENERATE
from image_project.stages.blackbox.idea_cards_assemble import STAGE as IDEA_CARDS_ASSEMBLE
from image_project.stages.blackbox.idea_cards_generate import STAGE as IDEA_CARDS_GENERATE
from image_project.stages.blackbox.idea_cards_judge_score import STAGE as IDEA_CARDS_JUDGE_SCORE
from image_project.stages.blackbox.image_prompt_creation import STAGE as IMAGE_PROMPT_CREATION
from image_project.stages.blackbox.image_prompt_draft import STAGE as IMAGE_PROMPT_DRAFT
from image_project.stages.blackbox.image_prompt_openai import STAGE as IMAGE_PROMPT_OPENAI
from image_project.stages.blackbox.image_prompt_refine import STAGE as IMAGE_PROMPT_REFINE
from image_project.stages.blackbox.prepare import STAGE as PREPARE
from image_project.stages.blackbox.profile_abstraction import STAGE as PROFILE_ABSTRACTION
from image_project.stages.blackbox.profile_hints_load import STAGE as PROFILE_HINTS_LOAD
from image_project.stages.blackbox.select_idea_card import STAGE as SELECT_IDEA_CARD

__all_stages__ = [
    PREPARE,
    PROFILE_ABSTRACTION,
    PROFILE_HINTS_LOAD,
    IDEA_CARD_GENERATE,
    IDEA_CARDS_ASSEMBLE,
    IDEA_CARDS_GENERATE,
    IDEA_CARDS_JUDGE_SCORE,
    SELECT_IDEA_CARD,
    IMAGE_PROMPT_CREATION,
    IMAGE_PROMPT_OPENAI,
    IMAGE_PROMPT_DRAFT,
    IMAGE_PROMPT_REFINE,
]
