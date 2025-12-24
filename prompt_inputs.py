from __future__ import annotations

import os
from dataclasses import dataclass

from run_config import RunConfig


@dataclass(frozen=True)
class ResolvedPromptInputs:
    draft_prompt: str | None = None


def resolve_prompt_inputs(cfg: RunConfig, *, required: tuple[str, ...] = ()) -> ResolvedPromptInputs:
    """
    Resolve plan inputs that may involve I/O, separate from orchestration.

    Plans should consume these resolved inputs rather than reading files directly.
    """

    required_set = set(required)
    unknown_required = sorted(required_set - {"draft_prompt"})
    if unknown_required:
        raise ValueError(f"Unknown required inputs: {unknown_required}")

    draft_prompt: str | None = None
    if cfg.prompt_refine_only_draft:
        draft_prompt = cfg.prompt_refine_only_draft
    elif cfg.prompt_refine_only_draft_path:
        path = str(cfg.prompt_refine_only_draft_path)
        if not os.path.exists(path):
            raise ValueError(f"prompt.refine_only.draft_path not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            draft_prompt = handle.read()

    draft_prompt = (draft_prompt or "").strip() or None
    if "draft_prompt" in required_set and not draft_prompt:
        raise ValueError("prompt.plan=refine_only requires prompt.refine_only.draft or draft_path")

    return ResolvedPromptInputs(draft_prompt=draft_prompt)

