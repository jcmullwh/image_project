from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import (
    PlanInputs,
    make_action_stage_block,
    make_chat_stage_block,
)
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.generator_profile_hints"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    """Build the stage block that produces `generator_profile_hints`."""

    mode = cfg.get_str("mode", default="raw", choices=("raw", "file", "abstract"))
    if mode is None:
        raise ValueError("blackbox.generator_profile_hints.mode cannot be null")

    hints_path = cfg.get_str("hints_path", default=None)
    model = cfg.get_str("model", default=None)
    temperature = cfg.get_float("temperature", default=0.0, min_value=0.0, max_value=2.0)

    if mode == "raw":

        def _action(ctx: RunContext) -> str:
            """Use the raw preferences guidance as generator hints."""

            hints = str(ctx.outputs.get("preferences_guidance") or "").strip()
            if not hints:
                raise ValueError("blackbox.generator_profile_hints requires preferences_guidance (mode=raw)")
            ctx.logger.info("Generator profile hints: mode=raw chars=%d", len(hints))
            return hints

        cfg.assert_consumed()
        return make_action_stage_block(
            instance_id,
            fn=_action,
            merge="none",
            step_capture_key="generator_profile_hints",
            doc="Set generator_profile_hints from raw profile guidance.",
            source="RunContext.outputs.preferences_guidance",
            tags=("blackbox",),
        )

    if mode == "file":
        if not hints_path:
            raise ValueError("blackbox.generator_profile_hints.mode=file requires hints_path")

        def _action(ctx: RunContext) -> str:
            """Load generator hints from a configured file."""

            from image_project.framework.profile_io import load_generator_profile_hints

            hints = load_generator_profile_hints(hints_path)
            if not isinstance(hints, str) or not hints.strip():
                raise ValueError(f"Generator profile hints file was empty: {hints_path}")

            normalized = hints.strip()
            ctx.logger.info(
                "Generator profile hints: mode=file path=%s chars=%d", hints_path, len(normalized)
            )
            return normalized

        cfg.assert_consumed()
        return make_action_stage_block(
            instance_id,
            fn=_action,
            merge="none",
            step_capture_key="generator_profile_hints",
            doc="Load generator_profile_hints from a file.",
            source="framework.profile_io.load_generator_profile_hints",
            tags=("blackbox",),
        )

    if mode != "abstract":
        raise ValueError(f"Unknown blackbox.generator_profile_hints.mode: {mode!r}")

    def _prompt(ctx: RunContext) -> str:
        """Generate generator hints via a prompt-policy abstraction template."""

        return prompts.profile_abstraction_prompt(
            preferences_guidance=str(ctx.outputs.get("preferences_guidance") or "")
        )

    params = {"model": model} if model else None
    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=float(temperature),
        merge="none",
        params=params,
        step_capture_key="generator_profile_hints",
        doc="Create generator_profile_hints via a profile abstraction prompt.",
        source="prompts.blackbox.profile_abstraction_prompt",
        tags=("blackbox",),
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Produce generator_profile_hints from raw/profile file/abstraction.",
    source="stages.blackbox.generator_profile_hints._build",
    tags=("blackbox",),
    kind="composite",
    io=StageIO(
        provides=("generator_profile_hints",),
        captures=("generator_profile_hints",),
    ),
)

