from __future__ import annotations

from image_project.stages.ab.final_prompt_format import STAGE as FINAL_PROMPT_FORMAT
from image_project.stages.ab.final_prompt_format_from_scenespec import (
    STAGE as FINAL_PROMPT_FORMAT_FROM_SCENESPEC,
)
from image_project.stages.ab.random_token import STAGE as RANDOM_TOKEN
from image_project.stages.ab.scene_draft import STAGE as SCENE_DRAFT
from image_project.stages.ab.scene_refine_no_block import STAGE as SCENE_REFINE_NO_BLOCK
from image_project.stages.ab.scene_refine_with_block import STAGE as SCENE_REFINE_WITH_BLOCK
from image_project.stages.ab.scene_spec_json import STAGE as SCENE_SPEC_JSON

__all_stages__ = [
    RANDOM_TOKEN,
    SCENE_DRAFT,
    SCENE_REFINE_NO_BLOCK,
    SCENE_REFINE_WITH_BLOCK,
    SCENE_SPEC_JSON,
    FINAL_PROMPT_FORMAT,
    FINAL_PROMPT_FORMAT_FROM_SCENESPEC,
]

