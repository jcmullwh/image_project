from __future__ import annotations

from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import preprompt as prompts
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "preprompt.select_concepts"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    selection_cfg = inputs.cfg.prompt_concepts.selection

    def _action(ctx: RunContext) -> dict[str, Any]:
        strategy = selection_cfg.strategy
        if strategy == "random":
            selected = prompts.select_random_concepts(inputs.prompt_data, inputs.rng)
            file_path: str | None = None
        elif strategy in ("fixed", "file"):
            selected = list(selection_cfg.fixed)
            file_path = selection_cfg.file_path if strategy == "file" else None
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown prompt.concepts.selection.strategy: {strategy!r}")

        if not selected:
            raise ValueError(f"Concept selection produced no concepts (strategy={strategy!r})")

        ctx.selected_concepts = list(selected)
        ctx.outputs["selected_concepts"] = list(ctx.selected_concepts)
        return {
            "strategy": strategy,
            "file_path": file_path,
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
