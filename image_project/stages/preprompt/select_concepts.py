from __future__ import annotations

import os
from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.foundation.config_io import find_repo_root
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import preprompt as prompts
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "preprompt.select_concepts"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    """Build the action stage that selects concepts via random/fixed/file strategy."""

    strategy = cfg.get_str(
        "strategy",
        default="random",
        choices=("random", "fixed", "file"),
    )
    if strategy is None:
        raise ValueError("preprompt.select_concepts.strategy cannot be null")

    fixed: list[str] | None = None
    selected_from_file: list[str] | None = None
    selected_file_path: str | None = None

    if strategy == "fixed":
        fixed = cfg.get_list_str("fixed")
    elif strategy == "file":
        file_path = cfg.get_str("file_path")
        if file_path is None:
            raise ValueError("preprompt.select_concepts.file_path cannot be null")

        def normalize_path(value: str) -> str:
            """Normalize a (possibly relative) path, anchored at the repo root."""

            text = str(value or "").strip()
            if not text:
                raise ValueError("preprompt.select_concepts.file_path must be a non-empty string")
            expanded = os.path.expandvars(os.path.expanduser(text))
            if not os.path.isabs(expanded):
                expanded = os.path.join(find_repo_root(), expanded)
            return os.path.abspath(expanded)

        selected_file_path = normalize_path(file_path)
        with open(selected_file_path, "r", encoding="utf-8") as handle:
            selected_from_file = [line.strip() for line in handle.read().splitlines() if line.strip()]
        if not selected_from_file:
            raise ValueError(f"Concept selection file produced no concepts: {selected_file_path}")

    def _action(ctx: RunContext) -> dict[str, Any]:
        if strategy == "random":
            selected = prompts.select_random_concepts(inputs.prompt_data, inputs.rng)
            selection_path: str | None = None
        elif strategy == "fixed":
            assert fixed is not None
            selected = list(fixed)
            selection_path = None
        else:
            assert selected_from_file is not None
            selected = list(selected_from_file)
            selection_path = selected_file_path

        if not selected:
            raise ValueError(f"Concept selection produced no concepts (strategy={strategy!r})")

        ctx.selected_concepts = list(selected)
        ctx.outputs["selected_concepts"] = list(ctx.selected_concepts)
        ctx.logger.info(
            "Selected concepts: strategy=%s count=%d",
            strategy,
            len(ctx.selected_concepts),
        )
        return {
            "strategy": strategy,
            "file_path": selection_path,
            "selected_concepts": list(ctx.selected_concepts),
        }

    cfg.assert_consumed()
    return make_action_stage_block(instance_id, fn=_action, merge="none")


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Select concepts (random/fixed/file) and store them on the run context.",
    source="prompts.preprompt.select_random_concepts",
    tags=("preprompt",),
    kind="action",
    io=StageIO(
        provides=("selected_concepts",),
    ),
)
