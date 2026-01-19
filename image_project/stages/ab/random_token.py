from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "ab.random_token"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    def _action(ctx: RunContext) -> str:
        roll = ctx.rng.randint(100000, 999999)
        return f"RV-{ctx.seed}-{roll}"

    cfg.assert_consumed()
    return make_action_stage_block(
        instance_id,
        fn=_action,
        merge="none",
        step_capture_key="ab_random_token",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Generate a deterministic per-run random token.",
    source="inline",
    tags=("ab",),
    kind="action",
    io=StageIO(
        provides=("ab_random_token",),
        captures=("ab_random_token",),
    ),
)
