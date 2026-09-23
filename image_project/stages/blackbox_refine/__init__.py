from __future__ import annotations

from image_project.stages.blackbox_refine.loop import STAGE as LOOP
from image_project.stages.blackbox_refine.seed_from_draft import STAGE as SEED_FROM_DRAFT
from image_project.stages.blackbox_refine.seed_prompt import STAGE as SEED_PROMPT

__all_stages__ = [
    SEED_PROMPT,
    SEED_FROM_DRAFT,
    LOOP,
]
