from __future__ import annotations

from image_project.stages.postprompt.openai_format import STAGE as OPENAI_FORMAT
from image_project.stages.postprompt.profile_nudge import STAGE as PROFILE_NUDGE

__all_stages__ = [
    PROFILE_NUDGE,
    OPENAI_FORMAT,
]

