from __future__ import annotations

from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.prepare"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    scoring_cfg = inputs.cfg.prompt_scoring

    def _action(ctx: RunContext) -> dict[str, Any]:
        from image_project.framework import scoring as blackbox_scoring

        if not scoring_cfg.enabled:
            raise ValueError("blackbox.prepare requires prompt.scoring.enabled=true")

        novelty_enabled_cfg = bool(scoring_cfg.novelty.enabled and scoring_cfg.novelty.window > 0)
        novelty_enabled_effective = bool(novelty_enabled_cfg)
        ctx.logger.info(
            "Blackbox scoring enabled: num_ideas=%d, exploration_rate=%.3g, novelty=%s",
            scoring_cfg.num_ideas,
            scoring_cfg.exploration_rate,
            novelty_enabled_cfg,
        )

        novelty_summary: dict[str, Any]
        if novelty_enabled_cfg:
            try:
                novelty_summary = blackbox_scoring.extract_recent_motif_summary(
                    generations_csv_path=ctx.cfg.generations_csv_path,
                    novelty_cfg=scoring_cfg.novelty,
                )
            except Exception as exc:
                ctx.logger.warning(
                    "Novelty enabled but history unavailable; disabling novelty for this run: %s %s",
                    ctx.cfg.generations_csv_path,
                    exc,
                )
                novelty_enabled_effective = False
                novelty_summary = {
                    "enabled": False,
                    "window": scoring_cfg.novelty.window,
                    "rows_considered": 0,
                    "top_tokens": [],
                }
        else:
            novelty_summary = {
                "enabled": False,
                "window": scoring_cfg.novelty.window,
                "rows_considered": 0,
                "top_tokens": [],
            }

        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        ctx.blackbox_scoring["novelty_summary"] = novelty_summary
        ctx.blackbox_scoring["novelty"] = {
            "method": scoring_cfg.novelty.method,
            "window": int(scoring_cfg.novelty.window),
            "enabled_cfg": bool(novelty_enabled_cfg),
            "enabled_effective": bool(novelty_enabled_effective),
        }

        # Default generator hints (may be overwritten by profile_abstraction stage).
        ctx.outputs["generator_profile_hints"] = str(ctx.outputs.get("preferences_guidance") or "")

        return {
            "novelty_enabled": bool(novelty_enabled_effective),
            "novelty_window": int(scoring_cfg.novelty.window),
            "novelty_method": scoring_cfg.novelty.method,
        }

    cfg.assert_consumed()
    return make_action_stage_block(instance_id, fn=_action, merge="none")


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Prepare blackbox scoring (novelty summary + default generator hints).",
    source="blackbox_scoring.extract_recent_motif_summary",
    tags=("blackbox",),
    kind="action",
    io=StageIO(
        provides=("generator_profile_hints",),
    ),
)
