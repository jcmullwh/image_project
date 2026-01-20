from __future__ import annotations

from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "preprompt.filter_concepts"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    filters_enabled = cfg.get_bool("enabled", default=False)
    order = tuple(
        cfg.get_list_str(
            "order",
            default=("dislike_rewrite",),
            allow_empty=not filters_enabled,
        )
    )

    dislike_cfg = cfg.namespace("dislike_rewrite", default={})
    dislike_enabled = dislike_cfg.get_bool("enabled", default=True)
    dislike_temperature = dislike_cfg.get_float(
        "temperature", default=0.25, min_value=0.0, max_value=2.0
    )

    def _action(ctx: RunContext) -> dict[str, Any]:
        from image_project.framework.inputs import apply_concept_filters, make_dislike_rewrite_filter

        concepts_in = list(ctx.selected_concepts)
        filters = []
        applied: list[str] = []
        skipped: list[dict[str, str]] = []

        if not filters_enabled:
            skipped.append({"name": "<all>", "reason": "disabled"})
        else:
            for name in order:
                if name == "dislike_rewrite":
                    if not dislike_enabled:
                        skipped.append({"name": name, "reason": "disabled"})
                        continue

                    dislikes_raw = ctx.outputs.get("dislikes")
                    dislikes = (
                        [str(value).strip() for value in dislikes_raw if str(value).strip()]
                        if isinstance(dislikes_raw, list)
                        else []
                    )
                    filters.append(
                        make_dislike_rewrite_filter(
                            dislikes=dislikes,
                            ai_text=inputs.ai_text,
                            temperature=float(dislike_temperature),
                        )
                    )
                    applied.append(name)
                else:
                    raise ValueError(f"Unknown concept filter: {name!r}")

        filtered, outcomes = apply_concept_filters(concepts_in, filters, logger=ctx.logger)

        ctx.outputs["concept_filter_log"] = {
            "enabled": bool(filters_enabled),
            "order": list(order),
            "applied": applied,
            "skipped": skipped,
            "input": concepts_in,
            "output": list(filtered),
            "filters": outcomes,
        }
        ctx.selected_concepts = list(filtered)
        ctx.outputs["selected_concepts"] = list(ctx.selected_concepts)

        return {
            "input": concepts_in,
            "output": list(filtered),
            "applied": applied,
            "skipped": skipped,
        }

    cfg.assert_consumed()
    return make_action_stage_block(instance_id, fn=_action, merge="none")


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Apply configured concept filters (in order) and record outcomes.",
    source="concept_filters.apply_concept_filters",
    tags=("preprompt",),
    kind="action",
    io=StageIO(
        requires=("selected_concepts",),
        provides=("selected_concepts", "concept_filter_log"),
    ),
)
