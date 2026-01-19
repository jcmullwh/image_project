from __future__ import annotations

from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "preprompt.filter_concepts"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    filters_cfg = inputs.cfg.prompt_concepts.filters

    def _action(ctx: RunContext) -> dict[str, Any]:
        from image_project.framework.inputs import apply_concept_filters, make_dislike_rewrite_filter

        concepts_in = list(ctx.selected_concepts)
        filters = []
        applied: list[str] = []
        skipped: list[dict[str, str]] = []

        if not filters_cfg.enabled:
            skipped.append({"name": "<all>", "reason": "disabled"})
        else:
            for name in filters_cfg.order:
                if name == "dislike_rewrite":
                    if not filters_cfg.dislike_rewrite.enabled:
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
                            temperature=filters_cfg.dislike_rewrite.temperature,
                        )
                    )
                    applied.append(name)
                else:  # pragma: no cover - guarded by config validation
                    skipped.append({"name": name, "reason": "unknown"})

        filtered, outcomes = apply_concept_filters(concepts_in, filters, logger=ctx.logger)

        ctx.outputs["concept_filter_log"] = {
            "enabled": bool(filters_cfg.enabled),
            "order": list(filters_cfg.order),
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
