from __future__ import annotations

from image_project.stages.blackbox_refine.loop import (
    BLACKBOX_REFINE_FINALIZE_STAGE,
    BLACKBOX_REFINE_INIT_STATE_STAGE,
    BLACKBOX_REFINE_ITER_STAGE,
)
from image_project.stages.blackbox_refine.seed_from_draft import STAGE as SEED_FROM_DRAFT
from image_project.stages.blackbox_refine.seed_prompt import STAGE as SEED_PROMPT

__all_stages__ = [
    SEED_PROMPT,
    SEED_FROM_DRAFT,
    BLACKBOX_REFINE_INIT_STATE_STAGE,
    BLACKBOX_REFINE_ITER_STAGE,
    BLACKBOX_REFINE_FINALIZE_STAGE,
]
