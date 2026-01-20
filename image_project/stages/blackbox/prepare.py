from __future__ import annotations

from typing import Any

from pipelinekit.config_namespace import ConfigNamespace
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "blackbox.prepare"


def _build(inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    novelty_ns = cfg.namespace("novelty", default={})
    novelty_enabled = novelty_ns.get_bool("enabled", default=False)
    novelty_window = novelty_ns.get_int("window", default=0, min_value=0)
    novelty_method = novelty_ns.get_str(
        "method",
        default="df_overlap_v1",
        choices=("df_overlap_v1",),
    )
    if novelty_method is None:
        raise ValueError("blackbox.prepare.novelty.method cannot be null")
    novelty_df_min = novelty_ns.get_int("df_min", default=3, min_value=1)
    novelty_max_motifs = novelty_ns.get_int("max_motifs", default=200, min_value=1)
    novelty_min_token_len = novelty_ns.get_int("min_token_len", default=3, min_value=1)
    novelty_stopwords_extra = tuple(
        novelty_ns.get_list_str("stopwords_extra", default=(), allow_empty=True)
    )
    novelty_max_penalty = novelty_ns.get_int("max_penalty", default=20, min_value=0)
    novelty_df_cap = novelty_ns.get_int("df_cap", default=10, min_value=1)
    novelty_alpha_only = novelty_ns.get_bool("alpha_only", default=True)
    novelty_scaling = novelty_ns.get_str(
        "scaling",
        default="linear",
        choices=("linear", "sqrt", "quadratic"),
    )
    if novelty_scaling is None:
        raise ValueError("blackbox.prepare.novelty.scaling cannot be null")

    def _action(ctx: RunContext) -> dict[str, Any]:
        from image_project.framework import scoring as blackbox_scoring
        from image_project.framework.scoring import NoveltyConfig

        novelty_cfg = NoveltyConfig(
            enabled=bool(novelty_enabled),
            window=int(novelty_window),
            method=str(novelty_method),
            df_min=int(novelty_df_min),
            max_motifs=int(novelty_max_motifs),
            min_token_len=int(novelty_min_token_len),
            stopwords_extra=tuple(novelty_stopwords_extra),
            max_penalty=int(novelty_max_penalty),
            df_cap=int(novelty_df_cap),
            alpha_only=bool(novelty_alpha_only),
            scaling=str(novelty_scaling),
        )

        novelty_enabled_cfg = bool(novelty_cfg.enabled and novelty_cfg.window > 0)
        novelty_enabled_effective = bool(novelty_enabled_cfg)
        ctx.logger.info(
            "Blackbox scoring prepare: novelty_enabled_cfg=%s window=%d method=%s",
            novelty_enabled_cfg,
            novelty_cfg.window,
            novelty_cfg.method,
        )

        novelty_summary: dict[str, Any]
        if novelty_enabled_cfg:
            prompt_inputs = ctx.outputs.get("prompt_inputs")
            generations_csv_path: str | None = None
            if isinstance(prompt_inputs, dict):
                raw_path = prompt_inputs.get("generations_csv_path")
                if isinstance(raw_path, str) and raw_path.strip():
                    generations_csv_path = raw_path

            try:
                if not generations_csv_path:
                    raise ValueError("generations_csv_path is missing")
                novelty_summary = blackbox_scoring.extract_recent_motif_summary(
                    generations_csv_path=generations_csv_path,
                    novelty_cfg=novelty_cfg,
                )
            except Exception as exc:
                ctx.logger.warning(
                    "Novelty enabled but history unavailable; disabling novelty for this run: %s %s",
                    generations_csv_path,
                    exc,
                )
                novelty_enabled_effective = False
                novelty_summary = {
                    "enabled": False,
                    "window": novelty_cfg.window,
                    "rows_considered": 0,
                    "top_tokens": [],
                }
        else:
            novelty_summary = {
                "enabled": False,
                "window": novelty_cfg.window,
                "rows_considered": 0,
                "top_tokens": [],
            }

        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        ctx.blackbox_scoring["novelty_summary"] = novelty_summary
        ctx.blackbox_scoring["novelty"] = {
            "method": novelty_cfg.method,
            "window": int(novelty_cfg.window),
            "enabled_cfg": bool(novelty_enabled_cfg),
            "enabled_effective": bool(novelty_enabled_effective),
        }

        # Default generator hints (may be overwritten by profile_abstraction stage).
        ctx.outputs["generator_profile_hints"] = str(ctx.outputs.get("preferences_guidance") or "")

        return {
            "novelty_enabled": bool(novelty_enabled_effective),
            "novelty_window": int(novelty_cfg.window),
            "novelty_method": novelty_cfg.method,
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
