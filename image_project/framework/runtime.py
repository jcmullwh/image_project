from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from image_project.foundation.messages import MessageHandler
from image_project.framework.config import RunConfig


@dataclass
class RunContext:
    generation_id: str
    cfg: RunConfig
    logger: logging.Logger
    rng: random.Random
    seed: int
    created_at: str

    messages: MessageHandler
    user_role: str = "user"
    agent_role: str = "assistant"

    selected_concepts: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    blackbox_scoring: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    image_path: str | None = None
    error: dict[str, Any] | None = None

