from __future__ import annotations

"""Experiment plan expansion for dry-runs.

The experiment runner writes two plan artifacts:

- `experiment_plan.json`: the concise summary + planned run entries.
- `experiment_plan_full.json`: an expanded artifact that parses each run's
  `cfg_dict` into strongly-typed config objects and compiles the prompt pipeline
  into a stage preview. This is the canonical place to surface configuration
  errors *before* executing any runs.
"""

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from image_project.app.generate import generation_defaults
from pipelinekit.engine.pipeline import ActionStep, Block, ChatStep
from image_project.framework.config import RunConfig
from image_project.framework.context import ContextManager
from image_project.framework.inputs import resolve_prompt_inputs
from image_project.framework.prompt_pipeline import (
    PlanInputs,
    ResolvedStages,
    compile_stage_nodes,
    resolve_stage_blocks,
)
from image_project.framework.prompt_pipeline.pipeline_overrides import PromptPipelineConfig
from image_project.impl.current.plans import PromptPlanManager
from image_project.stages.registry import get_stage_registry


def _utc_now_iso8601() -> str:
    """Return a canonical UTC timestamp string for artifact metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _callable_ref(value: Any) -> str | None:
    """Return a stable 'module.qualname' reference for callables, else None."""

    if not callable(value):
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(module, str) and module and isinstance(qualname, str) and qualname:
        return f"{module}.{qualname}"
    if isinstance(qualname, str) and qualname:
        return qualname
    return None


def _block_contains_chat_step(block: Block) -> bool:
    """Return True if a block contains any ChatStep (recursively)."""

    for node in block.nodes:
        if isinstance(node, ChatStep):
            return True
        if isinstance(node, Block) and _block_contains_chat_step(node):
            return True
    return False


def _block_contains_action_step(block: Block) -> bool:
    """Return True if a block contains any ActionStep (recursively)."""

    for node in block.nodes:
        if isinstance(node, ActionStep):
            return True
        if isinstance(node, Block) and _block_contains_action_step(node):
            return True
    return False


def _stage_block_to_dict(block: Block) -> dict[str, Any]:
    """Summarize a stage Block into a JSON-friendly dict for plan artifacts."""

    kind: str
    has_chat = _block_contains_chat_step(block)
    has_action = _block_contains_action_step(block)
    if has_chat and not has_action:
        kind = "chat"
    elif has_action and not has_chat:
        kind = "action"
    else:
        kind = "composite"

    summary: dict[str, Any] = {
        "stage_id": block.name,
        "kind": kind,
        "merge": block.merge,
        "capture_key": block.capture_key,
        "meta": dict(block.meta),
    }

    draft = next(
        (
            node
            for node in block.nodes
            if isinstance(node, ChatStep)
            and node.name == "draft"
            and isinstance(node.meta, dict)
            and node.meta.get("role") == "primary"
        ),
        None,
    )
    if isinstance(draft, ChatStep):
        prompt_text: str | None = draft.prompt if isinstance(draft.prompt, str) else None
        prompt_fn: str | None = _callable_ref(draft.prompt) if prompt_text is None else None
        summary["primary_draft"] = {
            "temperature": draft.temperature,
            "params": dict(draft.params),
            "allow_empty_prompt": bool(draft.allow_empty_prompt),
            "allow_empty_response": bool(draft.allow_empty_response),
            "capture_key": draft.capture_key,
            "prompt": prompt_text,
            "prompt_fn": prompt_fn,
        }

    action_step = next((node for node in block.nodes if isinstance(node, ActionStep)), None)
    if isinstance(action_step, ActionStep):
        summary["action"] = {
            "fn": _callable_ref(action_step.fn),
            "capture_key": action_step.capture_key,
        }

    summary["nodes_preview"] = [
        {"type": "block", "name": node.name}
        if isinstance(node, Block)
        else {"type": "chat", "name": node.name}
        if isinstance(node, ChatStep)
        else {"type": "action", "name": node.name}
        for node in block.nodes
    ]
    return summary


def _resolved_stages_to_dict(resolved: ResolvedStages) -> dict[str, Any]:
    """Convert resolved stage blocks to a JSON-friendly structure."""

    return {
        "metadata": dict(resolved.metadata),
        "stages": [_stage_block_to_dict(block) for block in resolved.stages],
    }


def _build_prompt_pipeline_details(
    cfg: RunConfig,
    *,
    prompt_cfg: PromptPipelineConfig,
    seed: int,
) -> dict[str, Any]:
    """Compile the prompt pipeline and return plan/pipeline details for inspection."""

    resolved_plan = PromptPlanManager.resolve(run_cfg=cfg, pipeline_cfg=prompt_cfg)

    required_inputs = tuple(resolved_plan.metadata.required_inputs)
    resolved_inputs_error: dict[str, Any] | None = None
    draft_prompt: str | None = None
    if required_inputs:
        try:
            resolved_inputs = resolve_prompt_inputs(prompt_cfg, required=required_inputs)
            draft_prompt = resolved_inputs.draft_prompt
        except Exception as exc:  # noqa: BLE001
            resolved_inputs_error = {"type": exc.__class__.__name__, "message": str(exc)}

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency: pandas (required for experiment plan expansion). Install via `pdm install`."
        ) from exc

    empty_df = pd.DataFrame()

    inputs = PlanInputs(
        cfg=cfg,
        pipeline=prompt_cfg.stages,
        ai_text=None,
        prompt_data=empty_df,
        user_profile=empty_df,
        preferences_guidance="",
        context_guidance=None,
        rng=random.Random(seed),
        draft_prompt=draft_prompt,
    )

    stage_nodes = resolved_plan.plan.stage_nodes(inputs)
    compiled = compile_stage_nodes(
        stage_nodes,
        plan_name=resolved_plan.plan.name,
        include=prompt_cfg.stages.include,
        exclude=prompt_cfg.stages.exclude,
        overrides=prompt_cfg.stages.overrides,
        stage_configs_defaults=prompt_cfg.stage_configs_defaults,
        stage_configs_instances=prompt_cfg.stage_configs_instances,
        stage_registry=get_stage_registry(),
        inputs=inputs,
    )
    resolved_stages = resolve_stage_blocks(
        list(compiled.blocks),
        plan_name=resolved_plan.plan.name,
        include=(),
        exclude=(),
        overrides=compiled.overrides,
        capture_stage=prompt_cfg.stages.capture_stage,
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
    """Expand a planned run entry into config/pipeline details (or config_error)."""

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
        prompt_cfg, prompt_warnings = PromptPipelineConfig.from_root_dict(
            cfg_dict,
            run_mode=cfg.run_mode,
            generation_dir=cfg.generation_dir,
        )
        details["config_parsed"] = asdict(cfg)
        details["config_warnings"] = [*cfg_warnings, *prompt_warnings]

        prompt_seed = seed if seed is not None else (prompt_cfg.random_seed if prompt_cfg.random_seed is not None else 0)
        details["prompt_pipeline"] = _build_prompt_pipeline_details(
            cfg,
            prompt_cfg=prompt_cfg,
            seed=int(prompt_seed),
        )
    except Exception as exc:  # noqa: BLE001
        details["config_error"] = {"type": exc.__class__.__name__, "message": str(exc)}

    return details


def _discover_context_injectors(*, logger: logging.Logger | None) -> None:
    """Discover context injector plugins before parsing config.

    `RunConfig.from_dict(...)` validates `context.injectors` against the injector
    registry in `image_project.framework.context`. That registry is populated by
    importing modules under `image_project.impl.current.context_plugins`, so any
    plan expansion must run discovery first.
    """

    from image_project.impl.current import context_plugins as _context_plugins  # noqa: PLC0415

    _context_plugins.discover()
    if logger is not None:
        available = ", ".join(ContextManager.available_injectors()) or "<none>"
        logger.debug("Context injectors discovered (dry-run): %s", available)


def write_experiment_plan_full(
    output_path: str,
    *,
    summary_payload: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    logger: logging.Logger | None = None,
) -> None:
    """Write `experiment_plan_full.json` with per-run config parse + pipeline preview.

    Any per-run parsing failure is recorded under `planned_runs_full[*].config_error`.
    This function does not execute model calls.
    """

    _discover_context_injectors(logger=logger)

    payload = dict(summary_payload)
    payload["kind"] = "experiment_plan_full"
    payload["generated_at"] = _utc_now_iso8601()
    payload["generation_defaults"] = generation_defaults()
    payload["planned_runs_full"] = [_build_run_details(run) for run in runs]

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
