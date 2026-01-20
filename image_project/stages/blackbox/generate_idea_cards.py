from __future__ import annotations

import json
from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.pipeline import ActionStep, Block, ChatStep
from image_project.framework.prompt_pipeline import PlanInputs
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from image_project.stages.blackbox._profiles import resolve_profile_text
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.generate_idea_cards"

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


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace) -> Block:
    """Build a composite stage that generates N isolated idea cards and assembles them."""

    from image_project.framework.scoring import expected_idea_ids

    num_ideas = cfg.get_int("num_ideas", default=6, min_value=1)
    idea_ids = expected_idea_ids(int(num_ideas))

    idea_profile_source = cfg.get_str(
        "idea_profile_source",
        default="none",
        choices=("raw", "generator_hints", "generator_hints_plus_dislikes", "none"),
    )
    if idea_profile_source is None:
        raise ValueError("blackbox.generate_idea_cards.idea_profile_source cannot be null")

    model = cfg.get_str("model", default=None)
    temperature = cfg.get_float("temperature", default=0.8, min_value=0.0, max_value=2.0)
    context_guidance = inputs.context_guidance or None

    generator_params = {"model": model} if model else {}
    profile_source_path = f"{cfg.path}.idea_profile_source"

    chat_steps: list[ChatStep] = []
    for idx, idea_id in enumerate(idea_ids, start=1):
        directive = _DIVERSITY_DIRECTIVES[(idx - 1) % len(_DIVERSITY_DIRECTIVES)]
        output_key = f"blackbox.idea_card.{idea_id}.json"
        stage_id = f"{instance_id}.idea_{idea_id}"

        def _prompt(
            ctx: RunContext,
            *,
            idea_id: str = idea_id,
            idea_ordinal: int = idx,
            directive: str = directive,
            stage_id: str = stage_id,
        ) -> str:
            """Prompt for generating a single strict-JSON idea card."""

            if idea_profile_source == "none":
                hints = ""
            elif idea_profile_source in ("raw", "generator_hints", "generator_hints_plus_dislikes"):
                hints = resolve_profile_text(
                    ctx,
                    source=idea_profile_source,
                    stage_id=stage_id,
                    config_path=profile_source_path,
                )
            else:  # pragma: no cover - guarded by config validation
                raise ValueError(
                    "Unknown blackbox.generate_idea_cards.idea_profile_source: "
                    f"{idea_profile_source!r} (expected: raw|generator_hints|generator_hints_plus_dislikes|none)"
                )

            return prompts.idea_card_generate_prompt(
                concepts=list(ctx.selected_concepts),
                generator_profile_hints=str(hints or ""),
                idea_id=idea_id,
                idea_ordinal=int(idea_ordinal),
                num_ideas=int(num_ideas),
                context_guidance=context_guidance,
                diversity_directive=directive,
            )

        chat_steps.append(
            ChatStep(
                name=f"idea_{idea_id}",
                prompt=_prompt,
                temperature=float(temperature),
                merge="none",
                allow_empty_prompt=False,
                allow_empty_response=False,
                capture_key=output_key,
                params=dict(generator_params),
                meta={
                    "doc": "Generate one idea card (strict JSON).",
                    "source": "prompts.blackbox.idea_card_generate_prompt",
                    "idea_id": idea_id,
                    "idea_ordinal": int(idx),
                    "num_ideas": int(num_ideas),
                },
            )
        )

    def _assemble_action(ctx: RunContext) -> str:
        """Parse per-idea JSON artifacts and return the combined idea_cards_json payload."""

        from image_project.framework import scoring as blackbox_scoring

        ideas: list[dict[str, Any]] = []
        for idea_id in idea_ids:
            key = f"blackbox.idea_card.{idea_id}.json"
            raw = ctx.outputs.get(key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Missing required output: {key}")
            try:
                ideas.append(blackbox_scoring.parse_idea_card_json(raw, expected_id=idea_id))
            except Exception as exc:
                setattr(exc, "pipeline_step", f"{instance_id}.idea_{idea_id}")
                setattr(exc, "pipeline_path", f"pipeline/{instance_id}/idea_{idea_id}")
                raise

        return json.dumps({"ideas": ideas}, ensure_ascii=False, indent=2)

    nodes: list[Any] = [*chat_steps]
    nodes.append(
        ActionStep(
            name="assemble",
            fn=_assemble_action,
            merge="none",
            capture_key="idea_cards_json",
            meta={
                "doc": "Assemble per-idea JSON artifacts into idea_cards_json.",
                "source": "framework.scoring.parse_idea_card_json",
                "num_ideas": int(num_ideas),
            },
        )
    )

    cfg.assert_consumed()
    return Block(
        name=instance_id,
        merge="none",
        nodes=nodes,
        meta={
            "doc": "Generate idea cards via isolated per-idea prompts, then assemble to idea_cards_json.",
            "tags": ["blackbox"],
        },
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Generate N isolated idea cards then assemble into idea_cards_json.",
    source="stages.blackbox.generate_idea_cards._build",
    tags=("blackbox",),
    kind="composite",
    io=StageIO(
        provides=("idea_cards_json",),
        captures=("idea_cards_json",),
    ),
)

