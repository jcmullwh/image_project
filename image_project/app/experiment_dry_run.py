from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from image_project.app.generate import generation_defaults
from image_project.framework.config import RunConfig
from image_project.framework.inputs import resolve_prompt_inputs
from image_project.framework.prompting import (
    ActionStageSpec,
    PlanInputs,
    ResolvedStages,
    StageSpec,
    resolve_stage_specs,
)
from image_project.impl.current.plans import PromptPlanManager


def _utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _callable_ref(value: Any) -> str | None:
    if not callable(value):
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(module, str) and module and isinstance(qualname, str) and qualname:
        return f"{module}.{qualname}"
    if isinstance(qualname, str) and qualname:
        return qualname
    return None


def _stage_spec_to_dict(spec: StageSpec | ActionStageSpec) -> dict[str, Any]:
    if isinstance(spec, ActionStageSpec):
        return {
            "type": "action",
            "stage_id": spec.stage_id,
            "fn": _callable_ref(spec.fn),
            "merge": spec.merge,
            "tags": list(spec.tags),
            "output_key": spec.output_key,
            "doc": spec.doc,
            "source": spec.source,
            "is_default_capture": bool(spec.is_default_capture),
        }

    prompt_text: str | None = spec.prompt if isinstance(spec.prompt, str) else None
    prompt_fn: str | None = _callable_ref(spec.prompt) if prompt_text is None else None

    return {
        "type": "prompt",
        "stage_id": spec.stage_id,
        "prompt": prompt_text,
        "prompt_fn": prompt_fn,
        "temperature": spec.temperature,
        "params": dict(spec.params),
        "merge": spec.merge,
        "allow_empty_prompt": bool(spec.allow_empty_prompt),
        "allow_empty_response": bool(spec.allow_empty_response),
        "tags": list(spec.tags),
        "refinement_policy": spec.refinement_policy,
        "output_key": spec.output_key,
        "doc": spec.doc,
        "source": spec.source,
        "is_default_capture": bool(spec.is_default_capture),
    }


def _resolved_stages_to_dict(resolved: ResolvedStages) -> dict[str, Any]:
    return {
        "metadata": dict(resolved.metadata),
        "stages": [_stage_spec_to_dict(spec) for spec in resolved.stages],
    }


def _build_prompt_pipeline_details(cfg: RunConfig, *, seed: int) -> dict[str, Any]:
    resolved_plan = PromptPlanManager.resolve(cfg)

    required_inputs = tuple(resolved_plan.metadata.required_inputs)
    resolved_inputs_error: dict[str, Any] | None = None
    draft_prompt: str | None = None
    if required_inputs:
        try:
            resolved_inputs = resolve_prompt_inputs(cfg, required=required_inputs)
            draft_prompt = resolved_inputs.draft_prompt
        except Exception as exc:  # noqa: BLE001
            resolved_inputs_error = {"type": exc.__class__.__name__, "message": str(exc)}

    try:
        import pandas as pd  # type: ignore[import-not-found]

        empty_df = pd.DataFrame()
    except Exception:  # pragma: no cover - pandas is a core dependency, but keep dry-run resilient.
        empty_df = None

    inputs = PlanInputs(
        cfg=cfg,
        ai_text=None,
        prompt_data=empty_df,
        user_profile=empty_df,
        preferences_guidance="",
        context_guidance=None,
        rng=random.Random(seed),
        draft_prompt=draft_prompt,
    )

    stage_specs = resolved_plan.plan.stage_specs(inputs)
    resolved_stages = resolve_stage_specs(
        stage_specs,
        plan_name=resolved_plan.plan.name,
        include=cfg.prompt_stages_include,
        exclude=cfg.prompt_stages_exclude,
        overrides=cfg.prompt_stages_overrides,
        capture_stage=cfg.prompt_output_capture_stage,
    )

    return {
        "plan_resolution": {
            "requested_plan": resolved_plan.requested_plan,
            "resolved_plan": resolved_plan.plan.name,
            "metadata": asdict(resolved_plan.metadata),
            "effective_context_enabled": bool(resolved_plan.effective_context_enabled),
        },
        "inputs": {
            "required": list(required_inputs),
            "draft_prompt": draft_prompt,
            "resolution_error": resolved_inputs_error,
        },
        "resolved_stages": _resolved_stages_to_dict(resolved_stages),
    }


def _build_run_details(run: Mapping[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = dict(run)

    cfg_dict = run.get("cfg_dict")
    if not isinstance(cfg_dict, Mapping):
        details["config_error"] = {
            "type": "TypeError",
            "message": "Run entry missing cfg_dict mapping",
        }
        return details

    seed: int | None = run.get("seed") if isinstance(run.get("seed"), int) else None

    try:
        cfg, cfg_warnings = RunConfig.from_dict(cfg_dict)
        details["config_parsed"] = asdict(cfg)
        details["config_warnings"] = list(cfg_warnings)

        prompt_seed = seed if seed is not None else (cfg.random_seed if cfg.random_seed is not None else 0)
        details["prompt_pipeline"] = _build_prompt_pipeline_details(cfg, seed=int(prompt_seed))
    except Exception as exc:  # noqa: BLE001
        details["config_error"] = {"type": exc.__class__.__name__, "message": str(exc)}

    return details


def write_experiment_plan_full(
    output_path: str,
    *,
    summary_payload: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> None:
    payload = dict(summary_payload)
    payload["kind"] = "experiment_plan_full"
    payload["generated_at"] = _utc_now_iso8601()
    payload["generation_defaults"] = generation_defaults()
    payload["planned_runs_full"] = [_build_run_details(run) for run in runs]

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

