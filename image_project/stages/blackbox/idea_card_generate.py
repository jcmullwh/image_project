from __future__ import annotations

import re

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from image_project.stages.blackbox._profiles import resolve_profile_text
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.idea_card_generate"
_INSTANCE_RE = re.compile(r"^blackbox\.idea_card_generate\.(.+)$")

_DIVERSITY_DIRECTIVES: tuple[str, ...] = (
    "Develop your strongest interpretation of how these components fit together.",
    "Follow the most compelling idea that emerges from these components, even if it doesn't match a clear pattern.",
    "Synthesize the components into a single coherent idea, prioritizing internal logic over novelty.",
    "Reinterpret the role or meaning of one component while keeping all components recognizable.",
    "Let one component strongly shape how the others are understood or used.",
    "Treat all components as equally fundamental, without an obvious focal element.",
    "Explore an interpretation that feels unexpected but still defensible given the components.",
    "Look for a non-obvious relationship or alignment between the components.",
)


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    match = _INSTANCE_RE.match((instance_id or "").strip())
    if match is None:
        raise ValueError(
            "blackbox.idea_card_generate requires instance_id of the form: "
            "blackbox.idea_card_generate.<idea_id>"
        )
    idea_id = match.group(1).strip()
    if not idea_id:
        raise ValueError(
            "blackbox.idea_card_generate requires instance_id of the form: "
            "blackbox.idea_card_generate.<idea_id>"
        )

    from image_project.framework.scoring import expected_idea_ids

    scoring_cfg = inputs.cfg.prompt_scoring
    idea_ids = expected_idea_ids(scoring_cfg.num_ideas)
    try:
        idea_ordinal = idea_ids.index(idea_id) + 1
    except ValueError:
        idea_ordinal = 1

    cfg.set_effective("idea_id", idea_id)
    cfg.set_effective("idea_ordinal", int(idea_ordinal))

    output_key = f"blackbox.idea_card.{idea_id}.json"
    context_guidance = inputs.context_guidance or None
    directive = _DIVERSITY_DIRECTIVES[(idea_ordinal - 1) % len(_DIVERSITY_DIRECTIVES)]

    def _prompt(ctx: RunContext) -> str:
        idea_profile_source = scoring_cfg.idea_profile_source
        if idea_profile_source == "none":
            hints = ""
        elif idea_profile_source in ("raw", "generator_hints", "generator_hints_plus_dislikes"):
            hints = resolve_profile_text(
                ctx,
                source=idea_profile_source,
                stage_id=instance_id,
                config_path="prompt.scoring.idea_profile_source",
            )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(
                "Unknown prompt.scoring.idea_profile_source: "
                f"{idea_profile_source!r} (expected: raw|generator_hints|generator_hints_plus_dislikes|none)"
            )

        return prompts.idea_card_generate_prompt(
            concepts=list(ctx.selected_concepts),
            generator_profile_hints=str(hints or ""),
            idea_id=idea_id,
            idea_ordinal=int(idea_ordinal),
            num_ideas=scoring_cfg.num_ideas,
            context_guidance=context_guidance,
            diversity_directive=directive,
        )

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=0.8,
        merge="none",
        step_capture_key=output_key,
        tags=("blackbox",),
        doc="Generate one idea card (strict JSON).",
        source="prompts.blackbox.idea_card_generate_prompt",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Generate a single idea card (strict JSON) as an isolated stage.",
    source="prompts.blackbox.idea_card_generate_prompt",
    tags=("blackbox",),
    kind="chat",
    io=StageIO(
        requires=("selected_concepts",),
    ),
)
