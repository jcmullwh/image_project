from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from image_project.stages.blackbox._profiles import resolve_profile_text
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.idea_cards_generate"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    scoring_cfg = inputs.cfg.prompt_scoring

    def _prompt(ctx: RunContext) -> str:
        idea_profile_source = scoring_cfg.idea_profile_source
        if idea_profile_source == "none":
            hints = ""
        elif idea_profile_source in ("raw", "generator_hints", "generator_hints_plus_dislikes"):
            hints = resolve_profile_text(
                ctx,
                source=idea_profile_source,
                stage_id=KIND_ID,
                config_path="prompt.scoring.idea_profile_source",
            )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(
                "Unknown prompt.scoring.idea_profile_source: "
                f"{idea_profile_source!r} (expected: raw|generator_hints|generator_hints_plus_dislikes|none)"
            )
        return prompts.idea_cards_generate_prompt(
            concepts=list(ctx.selected_concepts),
            generator_profile_hints=str(hints or ""),
            num_ideas=scoring_cfg.num_ideas,
        )

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=0.8,
        merge="none",
        step_capture_key="idea_cards_json",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Generate idea cards (strict JSON).",
    source="prompts.blackbox.idea_cards_generate_prompt",
    tags=("blackbox",),
    kind="chat",
    io=StageIO(
        provides=("idea_cards_json",),
        captures=("idea_cards_json",),
    ),
)
