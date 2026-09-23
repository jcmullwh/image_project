from __future__ import annotations

from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_chat_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts import blackbox as prompts
from image_project.stages.blackbox._profiles import resolve_profile_text
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.idea_cards_judge_score"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    judge_temperature = cfg.get_float("judge_temperature", default=0.0, min_value=0.0, max_value=2.0)
    judge_model = cfg.get_str("judge_model", default=None)
    judge_profile_source = cfg.get_str(
        "judge_profile_source",
        default="raw",
        choices=("raw", "generator_hints", "generator_hints_plus_dislikes"),
    )
    if judge_profile_source is None:
        raise ValueError("blackbox.idea_cards_judge_score.judge_profile_source cannot be null")

    context_guidance = inputs.context_guidance or None

    def _prompt(ctx: RunContext) -> str:
        import json

        idea_cards_json = ctx.outputs.get("idea_cards_json")
        if not isinstance(idea_cards_json, str) or not idea_cards_json.strip():
            raise ValueError("Missing required output: idea_cards_json")

        novelty_summary = (ctx.blackbox_scoring or {}).get("novelty_summary")
        recent_motif_summary: str | None = None
        if isinstance(novelty_summary, dict) and novelty_summary.get("enabled"):
            recent_motif_summary = json.dumps(novelty_summary, ensure_ascii=False, indent=2)

        return prompts.idea_cards_judge_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=resolve_profile_text(
                ctx,
                source=judge_profile_source,
                stage_id=KIND_ID,
                config_path=f"{cfg.path}.judge_profile_source",
            ),
            idea_cards_json=idea_cards_json,
            recent_motif_summary=recent_motif_summary,
            context_guidance=context_guidance,
        )

    judge_params: dict[str, Any] = {}
    if judge_model:
        judge_params["model"] = judge_model

    cfg.assert_consumed()
    return make_chat_stage_block(
        instance_id,
        prompt=_prompt,
        temperature=float(judge_temperature),
        merge="none",
        params=judge_params or None,
        step_capture_key="idea_scores_json",
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Judge idea cards and emit scores (strict JSON).",
    source="prompts.blackbox.idea_cards_judge_prompt",
    tags=("blackbox",),
    kind="chat",
    io=StageIO(
        requires=("idea_cards_json",),
        provides=("idea_scores_json",),
        captures=("idea_scores_json",),
    ),
)
