from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.profile_hints_load"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    hints_path = cfg.get_str("hints_path", default=None)
    if not hints_path:
        raise ValueError("blackbox.profile_hints_load requires hints_path")

    def _action(ctx: RunContext) -> str:
        from image_project.framework.profile_io import load_generator_profile_hints

        hints = load_generator_profile_hints(hints_path)
        if not isinstance(hints, str) or not hints.strip():
            raise ValueError(f"Generator profile hints file was empty: {hints_path}")

        ctx.logger.info(
            "Loaded generator profile hints from %s (chars=%d)", hints_path, len(hints)
        )
        return hints

    cfg.assert_consumed()
    return make_action_stage_block(
        instance_id,
        fn=_action,
        merge="none",
        step_capture_key="generator_profile_hints",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Load generator-safe profile hints from a file.",
    source="framework.profile_io.load_generator_profile_hints",
    tags=("blackbox",),
    kind="action",
    io=StageIO(
        provides=("generator_profile_hints",),
        captures=("generator_profile_hints",),
    ),
)
