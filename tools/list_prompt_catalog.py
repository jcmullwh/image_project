from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_project.foundation.config_io import load_config
from pipelinekit.engine.pipeline import ActionStep, Block, ChatStep
from image_project.framework.config import RunConfig
from image_project.framework.inputs import resolve_prompt_inputs
from image_project.framework.prompt_pipeline import (
    PlanInputs,
    compile_stage_nodes,
    resolve_stage_blocks,
)
from pipelinekit.config_namespace import ConfigNamespace
from image_project.impl.current.plans import PromptPlanManager, SequencePromptPlan
from image_project.stages.registry import get_stage_registry


def _print_json(value: Any) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


def _plan_summary(name: str) -> dict[str, Any]:
    plan = PromptPlanManager.get(name)
    summary: dict[str, Any] = {
        "name": name,
        "type": plan.__class__.__name__,
    }

    if isinstance(plan, SequencePromptPlan):
        # Best-effort: many plans are static `sequence=...`, but some are dynamic
        # (e.g., blackbox depends on config; custom depends on prompt.stages.sequence).
        sequence = getattr(plan, "sequence", None)
        if isinstance(sequence, tuple) and sequence:
            summary["sequence"] = list(sequence)
        else:
            summary["sequence"] = "<dynamic>"

    context_injection = getattr(plan, "context_injection", None)
    if isinstance(context_injection, str) and context_injection.strip():
        summary["context_injection"] = context_injection.strip().lower()

    requires_scoring = getattr(plan, "requires_scoring", None)
    if requires_scoring is not None:
        summary["requires_scoring"] = bool(requires_scoring)

    required_inputs = getattr(plan, "required_inputs", None)
    if isinstance(required_inputs, (list, tuple)) and required_inputs:
        summary["required_inputs"] = list(required_inputs)

    return summary


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


def _block_contains_chat_step(block: Block) -> bool:
    for node in block.nodes:
        if isinstance(node, ChatStep):
            return True
        if isinstance(node, Block) and _block_contains_chat_step(node):
            return True
    return False


def _block_contains_action_step(block: Block) -> bool:
    for node in block.nodes:
        if isinstance(node, ActionStep):
            return True
        if isinstance(node, Block) and _block_contains_action_step(node):
            return True
    return False


def _primary_draft_step(block: Block) -> ChatStep | None:
    for node in block.nodes:
        if (
            isinstance(node, ChatStep)
            and node.name == "draft"
            and isinstance(node.meta, dict)
            and node.meta.get("role") == "primary"
        ):
            return node
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="list_prompt_catalog",
        description="Print available prompt plans and catalog stages (offline; no AI calls).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plans", help="List available plans.")
    stages_parser = subparsers.add_parser("stages", help="List available catalog stages.")
    stages_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include stage type/merge/output_key/capture/refinement metadata.",
    )
    plan_parser = subparsers.add_parser("plan", help="Describe a single plan.")
    plan_parser.add_argument("name", help="Plan name (e.g., standard, blackbox, custom).")
    resolved_parser = subparsers.add_parser(
        "resolved",
        help="Resolve the requested plan + stage modifiers using the current config (no AI calls, no CSV reads).",
    )
    resolved_parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional explicit config path (otherwise uses config/config.yaml + config/config.local.yaml).",
    )

    args = parser.parse_args(argv)

    if args.command == "plans":
        plans = list(PromptPlanManager.available())
        if args.json:
            _print_json({"plans": [_plan_summary(name) for name in plans]})
        else:
            for name in plans:
                summary = _plan_summary(name)
                line = f"{summary['name']}"
                if summary.get("sequence") and summary["sequence"] != "<dynamic>":
                    line += f"  stages={summary['sequence']}"
                else:
                    line += "  stages=<dynamic>"
                if "context_injection" in summary:
                    line += f"  context={summary['context_injection']}"
                print(line)
        return 0

    if args.command == "stages":
        stages = list(get_stage_registry().describe())
        if getattr(args, "verbose", False):
            dummy_cfg_dict = {
                "run": {"mode": "prompt_only"},
                "prompt": {"categories_path": "categories.csv", "profile_path": "profile.csv"},
                "image": {"log_path": "."},
                "rclone": {"enabled": False},
                "upscale": {"enabled": False},
            }
            cfg, _warnings = RunConfig.from_dict(dummy_cfg_dict)
            inputs = PlanInputs(
                cfg=cfg,
                ai_text=None,
                prompt_data=pd.DataFrame(),
                user_profile=pd.DataFrame({"Likes": [], "Dislikes": []}),
                preferences_guidance="",
                context_guidance=None,
                rng=random.Random(0),
                draft_prompt="(draft)",
            )

            enriched: list[dict[str, Any]] = []
            for entry in stages:
                stage_id = entry.get("stage_id")
                if not isinstance(stage_id, str) or not stage_id.strip():
                    enriched.append(entry)
                    continue
                try:
                    ref = get_stage_registry().resolve(stage_id)
                    block = ref.build(
                        inputs,
                        instance_id=stage_id,
                        cfg=ConfigNamespace.empty(path=f"tools.list_prompt_catalog.stage[{stage_id}]"),
                    )
                except Exception as exc:  # noqa: BLE001
                    next_entry = dict(entry)
                    next_entry["verbose_error"] = f"{exc.__class__.__name__}: {exc}"
                    enriched.append(next_entry)
                    continue

                next_entry = dict(entry)
                has_chat = _block_contains_chat_step(block)
                has_action = _block_contains_action_step(block)
                if has_chat and not has_action:
                    next_entry["type"] = "chat"
                elif has_action and not has_chat:
                    next_entry["type"] = "action"
                else:
                    next_entry["type"] = "composite"

                next_entry["merge"] = block.merge
                next_entry["capture_key"] = block.capture_key

                draft = _primary_draft_step(block)
                if draft is not None:
                    prompt_text: str | None = draft.prompt if isinstance(draft.prompt, str) else None
                    next_entry["primary_temperature"] = draft.temperature
                    next_entry["primary_params"] = dict(draft.params)
                    next_entry["primary_step_capture_key"] = draft.capture_key
                    if prompt_text is not None:
                        next_entry["primary_prompt"] = prompt_text
                    else:
                        next_entry["primary_prompt_fn"] = _callable_ref(draft.prompt)
                enriched.append(next_entry)
            stages = enriched

        if args.json:
            _print_json({"stages": stages})
        else:
            for entry in stages:
                stage_id = entry.get("stage_id")
                doc = entry.get("doc") or ""
                source = entry.get("source") or ""
                tags = ",".join(entry.get("tags") or [])
                if getattr(args, "verbose", False):
                    stype = entry.get("type") or "?"
                    merge = entry.get("merge") or "?"
                    capture_key = entry.get("capture_key")
                    primary_capture = entry.get("primary_step_capture_key")
                    verbose_bits = [
                        f"type={stype}",
                        f"merge={merge}",
                        f"capture_key={capture_key}",
                        f"primary_capture={primary_capture}",
                    ]
                    if entry.get("verbose_error"):
                        verbose_bits.append(f"verbose_error={entry['verbose_error']}")
                    print(
                        f"{stage_id}  {'  '.join(verbose_bits)}  tags=[{tags}]  source={source}  doc={doc}"
                    )
                else:
                    print(f"{stage_id}  tags=[{tags}]  source={source}  doc={doc}")
        return 0

    if args.command == "plan":
        summary = _plan_summary(args.name)
        if args.json:
            _print_json(summary)
        else:
            print(summary["name"])
            print(f"  type={summary['type']}")
            if "sequence" in summary:
                print(f"  sequence={summary['sequence']}")
            if "required_inputs" in summary:
                print(f"  required_inputs={summary['required_inputs']}")
            if "requires_scoring" in summary:
                print(f"  requires_scoring={summary['requires_scoring']}")
            if "context_injection" in summary:
                print(f"  context_injection={summary['context_injection']}")
        return 0

    if args.command == "resolved":
        cfg_dict, _cfg_meta = load_config(config_path=args.config_path)
        cfg, _warnings = RunConfig.from_dict(cfg_dict)

        resolved_plan = PromptPlanManager.resolve(cfg)
        plan = resolved_plan.plan

        resolved_inputs = resolve_prompt_inputs(cfg, required=resolved_plan.metadata.required_inputs)
        inputs = PlanInputs(
            cfg=cfg,
            ai_text=None,
            prompt_data=pd.DataFrame(),
            user_profile=pd.DataFrame(),
            preferences_guidance="",
            context_guidance=None,
            rng=random.Random(0),
            draft_prompt=resolved_inputs.draft_prompt,
        )

        stage_nodes = plan.stage_nodes(inputs)
        compiled = compile_stage_nodes(
            stage_nodes,
            plan_name=plan.name,
            include=cfg.prompt_stages_include,
            exclude=cfg.prompt_stages_exclude,
            overrides=cfg.prompt_stages_overrides,
            stage_configs_defaults=cfg.prompt_stage_configs_defaults,
            stage_configs_instances=cfg.prompt_stage_configs_instances,
            stage_registry=get_stage_registry(),
            inputs=inputs,
        )
        resolved_stages = resolve_stage_blocks(
            list(compiled.blocks),
            plan_name=plan.name,
            include=(),
            exclude=(),
            overrides=compiled.overrides,
            capture_stage=cfg.prompt_output_capture_stage,
        )

        payload: dict[str, Any] = {
            "requested_plan": resolved_plan.requested_plan,
            "resolved_plan": plan.name,
            "run_mode": cfg.run_mode,
            "context_injection": resolved_plan.metadata.context_injection,
            "effective_context_enabled": bool(resolved_plan.effective_context_enabled),
            "context_injection_location": cfg.context_injection_location,
            "stage_ids": list(resolved_stages.stage_ids),
            "capture_stage": resolved_stages.capture_stage,
        }

        if args.json:
            _print_json(payload)
        else:
            print(f"requested_plan={payload['requested_plan']}")
            print(f"resolved_plan={payload['resolved_plan']}")
            print(f"run_mode={payload['run_mode']}")
            print(
                "context_injection="
                f"{payload['context_injection']} (effective_enabled={payload['effective_context_enabled']}, "
                f"location={payload['context_injection_location']})"
            )
            print(f"capture_stage={payload['capture_stage']}")
            for stage_id in payload["stage_ids"]:
                print(f"stage={stage_id}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
