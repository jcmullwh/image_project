from __future__ import annotations

from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.select_idea_card"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    scoring_cfg = inputs.cfg.prompt_scoring

    def _action(ctx: RunContext) -> dict[str, Any]:
        import random

        from image_project.framework import scoring as blackbox_scoring

        if not scoring_cfg.enabled:
            raise ValueError("blackbox.select_idea_card requires prompt.scoring.enabled=true")

        idea_cards_json = ctx.outputs.get("idea_cards_json")
        if not isinstance(idea_cards_json, str) or not idea_cards_json.strip():
            raise ValueError("Missing required output: idea_cards_json")
        idea_scores_json = ctx.outputs.get("idea_scores_json")
        if not isinstance(idea_scores_json, str) or not idea_scores_json.strip():
            raise ValueError("Missing required output: idea_scores_json")

        novelty_cfg = scoring_cfg.novelty
        novelty_enabled_cfg = bool(novelty_cfg.enabled and novelty_cfg.window > 0)
        novelty_summary: dict[str, Any] | None = None
        novelty_available = False
        if ctx.blackbox_scoring is not None:
            raw = ctx.blackbox_scoring.get("novelty_summary")
            if isinstance(raw, dict):
                novelty_summary = raw
                novelty_available = bool(raw.get("enabled"))

        novelty_missing = bool(novelty_enabled_cfg and not novelty_available)
        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        novelty_meta = ctx.blackbox_scoring.get("novelty")
        if not isinstance(novelty_meta, dict):
            novelty_meta = {}
            ctx.blackbox_scoring["novelty"] = novelty_meta
        novelty_meta.update(
            {
                "method": novelty_cfg.method,
                "window": int(novelty_cfg.window),
                "enabled_cfg": bool(novelty_enabled_cfg),
                "summary_available": bool(novelty_available),
                "missing_summary": bool(novelty_missing),
            }
        )

        if novelty_missing:
            warn = (
                "pipeline/blackbox.select_idea_card/action: prompt.scoring.novelty.enabled=true but novelty summary is "
                "unavailable; novelty penalties disabled for this run"
            )
            ctx.logger.warning(warn)
            warnings_log = ctx.blackbox_scoring.get("warnings")
            if not isinstance(warnings_log, list):
                warnings_log = []
                ctx.blackbox_scoring["warnings"] = warnings_log
            warnings_log.append(warn)

        try:
            idea_cards = blackbox_scoring.parse_idea_cards_json(
                idea_cards_json, expected_num_ideas=scoring_cfg.num_ideas
            )
        except Exception as exc:
            pipeline = ctx.outputs.get("prompt_pipeline")
            resolved = pipeline.get("resolved_stages") if isinstance(pipeline, dict) else None
            if isinstance(resolved, list) and "blackbox.idea_cards_assemble" in resolved:
                setattr(exc, "pipeline_step", "blackbox.idea_cards_assemble")
                setattr(exc, "pipeline_path", "pipeline/blackbox.idea_cards_assemble/action")
            else:
                setattr(exc, "pipeline_step", "blackbox.idea_cards_generate")
                setattr(exc, "pipeline_path", "pipeline/blackbox.idea_cards_generate/draft")
            raise

        expected_ids = [card.get("id") for card in idea_cards if isinstance(card, dict)]

        try:
            scores = blackbox_scoring.parse_judge_scores_json(
                idea_scores_json, expected_ids=expected_ids
            )
        except Exception as exc:
            setattr(exc, "pipeline_step", "blackbox.idea_cards_judge_score")
            setattr(exc, "pipeline_path", "pipeline/blackbox.idea_cards_judge_score/draft")
            raise

        scoring_seed = int(ctx.seed) ^ 0xB10C5C0F
        selection = blackbox_scoring.select_candidate(
            scores=scores,
            idea_cards=idea_cards,
            exploration_rate=scoring_cfg.exploration_rate,
            rng=random.Random(scoring_seed),
            novelty_cfg=novelty_cfg,
            novelty_summary=novelty_summary,
        )

        selected_card = next(
            (card for card in idea_cards if card.get("id") == selection.selected_id),
            None,
        )
        if not isinstance(selected_card, dict):
            raise ValueError(
                f"selection inconsistency: selected id not found: {selection.selected_id}"
            )

        ctx.outputs["selected_idea_card"] = selected_card
        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        ctx.blackbox_scoring.update(
            {
                "scoring_seed": scoring_seed,
                "exploration_rate": scoring_cfg.exploration_rate,
                "exploration_roll": selection.exploration_roll,
                "selection_mode": selection.selection_mode,
                "selected_id": selection.selected_id,
                "selected_score": selection.selected_score,
                "selected_effective_score": selection.selected_effective_score,
                "score_table": selection.score_table,
            }
        )

        ctx.logger.info(
            "Selected candidate: id=%s, score=%d, selection_mode=%s",
            selection.selected_id,
            selection.selected_score,
            selection.selection_mode,
        )

        return {
            "selected_id": selection.selected_id,
            "selection_mode": selection.selection_mode,
        }

    cfg.assert_consumed()
    return make_action_stage_block(instance_id, fn=_action, merge="none")


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Select an idea card using judge scores (and novelty penalties when enabled).",
    source="blackbox_scoring.select_candidate",
    tags=("blackbox",),
    kind="action",
    io=StageIO(
        requires=("idea_cards_json", "idea_scores_json"),
        provides=("selected_idea_card",),
    ),
)
