from __future__ import annotations

"""Canonical experiment runner.

This module centralizes the infrastructure that was historically duplicated
across multiple `tools/run_experiment_*.py` scripts:

- loading base config + deep-merging overlays
- consistent artifact directory layout under `_artifacts/experiments/<exp_dir>/`
- writing `experiment_plan.json` and `experiment_plan_full.json`
- executing planned runs via `run_generation(...)`
- writing `experiment_results.json` and optional `pairs.json`
- updating the artifacts index once at the end

The *only* experiment-specific logic should live in experiment plugins under
`image_project.impl.current.experiment_plugins`.
"""

import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from image_project.app.experiment_dry_run import write_experiment_plan_full
from image_project.app.generate import configure_stdio_utf8, run_generation
from image_project.foundation.config_io import deep_merge, find_repo_root, load_config
from image_project.framework.artifacts import utc_now_iso8601, update_artifacts_index
from image_project.framework.experiment_manifest import record_pair_error, write_pairs_manifest
from image_project.impl.current.experiments import Experiment, ExperimentManager, RunSpec


def _utc_timestamp_compact() -> str:
    """Return a filesystem-friendly local timestamp like YYYYMMDD_HHMMSS."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_output_root(*, experiment_name: str) -> str:
    """Compute the default output directory for an experiment run."""

    name = (experiment_name or "").strip()
    if not name:
        raise ValueError("experiment_name must be a non-empty string")
    timestamp = _utc_timestamp_compact()
    return os.path.join("_artifacts", "experiments", f"{timestamp}_{name}")


def default_experiment_id(*, experiment_name: str) -> str:
    """Compute a default experiment id when the user does not provide one."""

    name = (experiment_name or "").strip()
    if not name:
        raise ValueError("experiment_name must be a non-empty string")
    timestamp = _utc_timestamp_compact()
    return f"{name}_{timestamp}"


def _setup_experiment_logger(*, output_root: str, experiment_id: str) -> logging.Logger:
    """Configure an experiment-level logger (stdout + file)."""

    os.makedirs(output_root, exist_ok=True)
    log_path = os.path.join(output_root, "experiment.log")

    logger_name = f"image_project.experiments.{experiment_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info("Experiment logging initialized (id=%s)", experiment_id)
    logger.debug("Experiment log file: %s", log_path)
    return logger


def _discover_context_injectors(*, logger: logging.Logger | None = None) -> None:
    """Discover context injector plugins so config parsing can validate injectors.

    `RunConfig.from_dict(...)` validates `context.injectors` against the injector
    registry in `image_project.framework.context`. That registry is populated by
    importing modules under `image_project.impl.current.context_plugins`, so any
    experiment flow that parses config must run discovery first (including plan
    building, dry-run expansion, and full execution).
    """

    from image_project.impl.current import context_plugins as _context_plugins  # noqa: PLC0415
    from image_project.framework.context import ContextManager  # noqa: PLC0415

    _context_plugins.discover()
    if logger is not None:
        available = ", ".join(ContextManager.available_injectors()) or "<none>"
        logger.debug("Context injectors discovered: %s", available)


def _write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _infer_repo_root(config_meta: Mapping[str, Any] | None) -> str | None:
    """Infer repo root for indexing and path normalization."""

    raw = (config_meta or {}).get("repo_root")
    if isinstance(raw, str) and raw.strip():
        return str(Path(raw).resolve())
    try:
        return find_repo_root()
    except Exception:
        return None


def _extract_runner_cfg(base_cfg: Mapping[str, Any], *, experiment_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split runner-only config (`experiment_runners`) from the base config."""

    base_clean = dict(base_cfg)
    runners_cfg_raw = base_clean.pop("experiment_runners", None)
    runners_cfg = runners_cfg_raw if isinstance(runners_cfg_raw, Mapping) else {}
    runner_raw = runners_cfg.get(experiment_name) if isinstance(runners_cfg, Mapping) else None
    runner_cfg = dict(runner_raw) if isinstance(runner_raw, Mapping) else {}
    return base_clean, runner_cfg


def _merge_cfg(base_cfg: Mapping[str, Any], *overlays: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge overlays onto a base mapping."""

    merged: Any = dict(base_cfg)
    for overlay in overlays:
        merged = deep_merge(merged, overlay, path="")
    if not isinstance(merged, dict):
        raise TypeError("Merged config must be a mapping")
    return merged


def _standard_overrides(
    *,
    output_root: str,
    experiment_id: str,
    run_mode: str,
    enable_upscale: bool,
    enable_upload: bool,
) -> dict[str, Any]:
    """Build the runner-owned standard config overlay for all experiments."""

    output_root_abs = os.path.abspath(output_root)
    log_dir = os.path.join(output_root_abs, "logs")
    generation_dir = os.path.join(output_root_abs, "generated")
    upscale_dir = os.path.join(output_root_abs, "upscaled")

    return {
        "run": {"mode": str(run_mode)},
        "image": {
            "log_path": log_dir,
            "generation_path": generation_dir,
            "upscale_path": upscale_dir,
        },
        "prompt": {
            "generations_path": os.path.join(log_dir, "generations_v2.csv"),
            "titles_manifest_path": os.path.join(generation_dir, "titles_manifest.csv"),
        },
        "experiment": {"id": str(experiment_id)},
        "upscale": {"enabled": bool(enable_upscale)},
        "rclone": {"enabled": bool(enable_upload)},
    }


def _validate_run_spec(run: RunSpec, *, idx: int) -> None:
    """Validate a single RunSpec for runner invariants."""

    if not isinstance(run.variant, str) or not run.variant.strip():
        raise ValueError(f"plan[{idx}].variant must be a non-empty string")
    if run.variant_name is not None and (not isinstance(run.variant_name, str) or not run.variant_name.strip()):
        raise ValueError(f"plan[{idx}].variant_name must be a non-empty string or null")
    if not isinstance(run.run, int) or run.run <= 0:
        raise ValueError(f"plan[{idx}].run must be an int >= 1")
    if not isinstance(run.generation_id, str) or not run.generation_id.strip():
        raise ValueError(f"plan[{idx}].generation_id must be a non-empty string")
    if run.seed is not None and (not isinstance(run.seed, int) or isinstance(run.seed, bool)):
        raise ValueError(f"plan[{idx}].seed must be an int or null")
    if not isinstance(run.cfg_dict, dict):
        raise TypeError(f"plan[{idx}].cfg_dict must be a dict")
    if not isinstance(run.meta, dict):
        raise TypeError(f"plan[{idx}].meta must be a dict")


def _planned_run_entry(run: RunSpec) -> dict[str, Any]:
    """Convert a RunSpec into a summary entry for `experiment_plan.json`."""

    prompt_cfg = run.cfg_dict.get("prompt") if isinstance(run.cfg_dict.get("prompt"), Mapping) else {}
    plan_name = prompt_cfg.get("plan") if isinstance(prompt_cfg, Mapping) else None
    if plan_name is not None and (not isinstance(plan_name, str) or not plan_name.strip()):
        raise ValueError(f"Invalid prompt.plan for generation_id={run.generation_id}: {plan_name!r}")

    entry: dict[str, Any] = {
        "variant": run.variant,
        "variant_name": run.variant_name,
        "plan": plan_name,
        "run": run.run,
        "generation_id": run.generation_id,
        "seed": run.seed,
    }

    for key, value in run.meta.items():
        if key in entry:
            raise ValueError(
                f"RunSpec.meta key {key!r} conflicts with a reserved planned_run field "
                f"(generation_id={run.generation_id})"
            )
        entry[key] = value

    return entry


def _runs_full_entries(plan: Sequence[RunSpec]) -> list[dict[str, Any]]:
    """Convert RunSpecs into the payload expected by `write_experiment_plan_full()`."""

    runs: list[dict[str, Any]] = []
    for run in plan:
        entry = _planned_run_entry(run)
        entry["cfg_dict"] = run.cfg_dict
        runs.append(entry)
    return runs


def _load_plan_full_errors(plan_full_path: str) -> list[dict[str, Any]]:
    """Extract config errors from `experiment_plan_full.json`."""

    payload = json.loads(Path(plan_full_path).read_text(encoding="utf-8"))
    planned = payload.get("planned_runs_full")
    if not isinstance(planned, list):
        raise ValueError("experiment_plan_full.json missing planned_runs_full list")
    errors: list[dict[str, Any]] = []
    for item in planned:
        if not isinstance(item, dict):
            continue
        err = item.get("config_error")
        if isinstance(err, dict):
            errors.append(
                {
                    "variant": item.get("variant"),
                    "run": item.get("run"),
                    "generation_id": item.get("generation_id"),
                    "error": dict(err),
                }
            )
    return errors


def _load_plan_from_plan_full(plan_full_path: str) -> tuple[list[RunSpec], dict[str, Any]]:
    """Reconstruct a RunSpec plan from an existing `experiment_plan_full.json`.

    This is used by `--resume` to ensure generation ids and configs match the
    original plan (avoiding drift when re-planning).
    """

    payload = json.loads(Path(plan_full_path).read_text(encoding="utf-8"))
    planned = payload.get("planned_runs_full")
    if not isinstance(planned, list):
        raise ValueError("experiment_plan_full.json missing planned_runs_full list")

    reserved = {
        "variant",
        "variant_name",
        "set",
        "set_name",
        "plan",
        "run",
        "generation_id",
        "seed",
        "cfg_dict",
        "config_parsed",
        "config_warnings",
        "config_error",
        "prompt_pipeline",
    }

    plan: list[RunSpec] = []
    for idx, item in enumerate(planned):
        if not isinstance(item, Mapping):
            continue
        variant = item.get("variant") if isinstance(item.get("variant"), str) else item.get("set")
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError(f"planned_runs_full[{idx}].variant missing/invalid")

        variant_name = item.get("variant_name")
        if variant_name is None:
            variant_name = item.get("set_name")
        if variant_name is not None and (not isinstance(variant_name, str) or not variant_name.strip()):
            raise ValueError(f"planned_runs_full[{idx}].variant_name invalid")

        run_number = item.get("run")
        if not isinstance(run_number, int) or run_number <= 0:
            raise ValueError(f"planned_runs_full[{idx}].run missing/invalid")

        generation_id = item.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise ValueError(f"planned_runs_full[{idx}].generation_id missing/invalid")

        seed = item.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"planned_runs_full[{idx}].seed invalid")

        cfg_dict = item.get("cfg_dict")
        if not isinstance(cfg_dict, Mapping):
            raise ValueError(f"planned_runs_full[{idx}].cfg_dict missing/invalid")

        meta: dict[str, Any] = {
            k: v for k, v in item.items() if isinstance(k, str) and k not in reserved
        }

        plan.append(
            RunSpec(
                variant=variant.strip(),
                variant_name=variant_name.strip() if isinstance(variant_name, str) else None,
                run=int(run_number),
                generation_id=generation_id.strip(),
                seed=int(seed) if isinstance(seed, int) else None,
                cfg_dict=dict(cfg_dict),
                meta=meta,
            )
        )

    if not plan:
        raise ValueError("experiment_plan_full.json contains no planned runs")
    return plan, dict(payload) if isinstance(payload, dict) else {}


def _resume_filter_plan(plan: Sequence[RunSpec], *, log_dir: str, logger: logging.Logger) -> list[RunSpec]:
    """Skip runs that already have a success entry in runs_index.jsonl."""

    run_index_path = Path(log_dir) / "runs_index.jsonl"
    if not run_index_path.exists():
        logger.warning("--resume requested but %s does not exist; nothing to skip", str(run_index_path))
        return list(plan)

    succeeded: set[str] = set()
    try:
        with run_index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("status") != "success":
                    continue
                generation_id = record.get("generation_id")
                if isinstance(generation_id, str) and generation_id.strip():
                    succeeded.add(generation_id.strip())
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read %s for resume: %s", str(run_index_path), exc)
        return list(plan)

    remaining = [run for run in plan if run.generation_id not in succeeded]
    skipped = len(plan) - len(remaining)
    if skipped:
        logger.info("Resume: skipping %d/%d runs already marked success", skipped, len(plan))
    else:
        logger.info("Resume: no successful runs detected; executing full plan (%d runs)", len(plan))
    return remaining


def run_experiment(
    *,
    experiment_name: str,
    config_path: str | None,
    output_root: str | None,
    experiment_id: str | None,
    run_mode: str,
    dry_run: bool,
    resume: bool,
    continue_on_error: bool,
    enable_upscale: bool,
    enable_upload: bool,
    fail_on_config_error: bool,
    plugin_args: Mapping[str, Any],
) -> int:
    """Run an experiment by name.

    Returns:
        Process-style exit code:
          - 0: success / dry-run validated
          - 1: one or more run failures (or analysis failures)
          - 2: configuration/planning validation errors
    """

    configure_stdio_utf8()

    exp_name = (experiment_name or "").strip()
    if not exp_name:
        raise ValueError("experiment_name must be a non-empty string")

    run_mode_norm = (run_mode or "").strip().lower()
    if run_mode_norm not in {"full", "prompt_only"}:
        raise ValueError(f"Unknown run mode: {run_mode!r} (expected: full|prompt_only)")

    if resume and not output_root:
        raise ValueError("--resume requires --output-root pointing at an existing experiment directory")

    output_root_final = os.path.abspath(output_root) if output_root else os.path.abspath(default_output_root(experiment_name=exp_name))

    summary_path = os.path.join(output_root_final, "experiment_plan.json")
    plan_full_path = os.path.join(output_root_final, "experiment_plan_full.json")

    cfg_meta: Mapping[str, Any] | None = None
    repo_root: str | None = None
    runner_cfg: Mapping[str, Any] = {}
    overrides: dict[str, Any] | None = None
    plan_meta: Mapping[str, Any] | None = None

    experiment: Experiment = ExperimentManager.get(exp_name)

    if resume:
        if config_path:
            # Resume uses the stored cfg_dict in the plan; config_path is ignored.
            pass

        if not os.path.exists(plan_full_path):
            raise FileNotFoundError(f"--resume requires {plan_full_path} to exist")

        plan, plan_full_payload = _load_plan_from_plan_full(plan_full_path)
        stored_name = plan_full_payload.get("experiment_name")
        if isinstance(stored_name, str) and stored_name.strip() and stored_name.strip() != exp_name:
            raise ValueError(
                f"--resume experiment name mismatch: CLI={exp_name!r} plan_full={stored_name!r}"
            )

        stored_run_mode = plan_full_payload.get("run_mode")
        if isinstance(stored_run_mode, str) and stored_run_mode.strip().lower() in {"full", "prompt_only"}:
            stored_norm = stored_run_mode.strip().lower()
            if stored_norm != run_mode_norm:
                # Resume uses the planned cfg_dict; prefer the stored run mode for metadata.
                run_mode_norm = stored_norm

        stored_id = plan_full_payload.get("experiment_id")
        experiment_id_final = (experiment_id or "").strip() or (stored_id.strip() if isinstance(stored_id, str) and stored_id.strip() else Path(output_root_final).name)
        cfg_meta_raw = plan_full_payload.get("config_meta")
        cfg_meta = dict(cfg_meta_raw) if isinstance(cfg_meta_raw, Mapping) else None
        plan_meta_raw = plan_full_payload.get("experiment_meta")
        plan_meta = dict(plan_meta_raw) if isinstance(plan_meta_raw, Mapping) else None
        repo_root = _infer_repo_root(cfg_meta or {})

        logger = _setup_experiment_logger(output_root=output_root_final, experiment_id=experiment_id_final)
    else:
        experiment_id_final = (experiment_id or "").strip() or default_experiment_id(experiment_name=exp_name)
        logger = _setup_experiment_logger(output_root=output_root_final, experiment_id=experiment_id_final)

        _discover_context_injectors(logger=logger)

        cfg_dict, cfg_meta_loaded = load_config(config_path=config_path) if config_path else load_config()
        if not isinstance(cfg_dict, Mapping):
            raise TypeError("load_config() returned a non-mapping config")
        cfg_meta = cfg_meta_loaded if isinstance(cfg_meta_loaded, Mapping) else None
        repo_root = _infer_repo_root(cfg_meta or {})

        base_cfg_clean, runner_cfg = _extract_runner_cfg(cfg_dict, experiment_name=exp_name)

        overrides = _standard_overrides(
            output_root=output_root_final,
            experiment_id=experiment_id_final,
            run_mode=run_mode_norm,
            enable_upscale=enable_upscale,
            enable_upload=enable_upload,
        )
        base_with_overrides = _merge_cfg(base_cfg_clean, overrides)

        try:
            plan, plan_meta_dict = experiment.build_plan(
                base_cfg=base_with_overrides,
                runner_cfg=runner_cfg,
                output_root=output_root_final,
                experiment_id=experiment_id_final,
                run_mode=run_mode_norm,
                cli_args=plugin_args,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Experiment plan build failed: %s", exc)
            raise
        plan_meta = plan_meta_dict

        if not plan:
            raise ValueError(f"Experiment {exp_name!r} produced an empty plan")

    logger.info("Experiment: name=%s id=%s", exp_name, experiment_id_final)
    logger.info("Output root: %s", output_root_final)
    if output_root is None and not resume:
        logger.warning("No --output-root provided; defaulted output_root=%s", output_root_final)
    if experiment_id is None and not resume:
        logger.warning("No --experiment-id provided; defaulted experiment_id=%s", experiment_id_final)

    if repo_root:
        logger.info("Repo root: %s", repo_root)
    else:
        logger.warning("Repo root unavailable; artifacts index paths may be absolute")

    if runner_cfg:
        logger.info("Loaded runner cfg: experiment_runners.%s (%d keys)", exp_name, len(runner_cfg))

    generation_ids: set[str] = set()
    for idx, run in enumerate(plan):
        _validate_run_spec(run, idx=idx)
        if run.generation_id in generation_ids:
            raise ValueError(f"Duplicate generation_id in plan: {run.generation_id}")
        generation_ids.add(run.generation_id)

    summary_payload: dict[str, Any] | None = None
    if not resume:
        if overrides is None:
            raise AssertionError("runner overrides missing for non-resume execution")

        summary_payload = {
            "schema_version": 1,
            "kind": "experiment_plan",
            "generated_at": utc_now_iso8601(),
            "experiment_name": exp_name,
            "experiment_id": experiment_id_final,
            "output_root": output_root_final,
            "run_mode": run_mode_norm,
            "config_meta": dict(cfg_meta) if isinstance(cfg_meta, Mapping) else None,
            "plugin": {
                "ref": f"{experiment.__class__.__module__}.{experiment.__class__.__name__}",
                "summary": getattr(experiment, "summary", ""),
            },
            "runner": {
                "standard_overrides": overrides,
                "flags": {
                    "dry_run": bool(dry_run),
                    "resume": bool(resume),
                    "continue_on_error": bool(continue_on_error),
                    "enable_upscale": bool(enable_upscale),
                    "enable_upload": bool(enable_upload),
                    "fail_on_config_error": bool(fail_on_config_error),
                },
            },
            "experiment_args": dict(plugin_args),
            "experiment_meta": dict(plan_meta or {}),
            "planned_runs": [_planned_run_entry(run) for run in plan],
        }

        _write_json(summary_path, summary_payload)
        logger.info("Wrote experiment plan: %s", summary_path)

        write_experiment_plan_full(
            plan_full_path,
            summary_payload=summary_payload,
            runs=_runs_full_entries(plan),
            logger=logger,
        )
        logger.info("Wrote expanded experiment plan: %s", plan_full_path)

    config_errors = _load_plan_full_errors(plan_full_path)
    if config_errors:
        logger.error("Config validation failed for %d planned run(s)", len(config_errors))
        logger.error("%s", json.dumps(config_errors, ensure_ascii=False, indent=2))
        if fail_on_config_error:
            logger.error("Aborting due to --fail-on-config-error")
            _update_artifacts_index_once(
                output_root=output_root_final,
                repo_root=repo_root,
                logger=logger,
            )
            return 2

    if dry_run:
        _update_artifacts_index_once(
            output_root=output_root_final,
            repo_root=repo_root,
            logger=logger,
        )
        return 0

    plan_to_run = _resume_filter_plan(plan, log_dir=os.path.join(output_root_final, "logs"), logger=logger) if resume else list(plan)

    pairs_payload = experiment.build_pairs_manifest(
        plan, experiment_id=experiment_id_final, run_mode=run_mode_norm
    ) if hasattr(experiment, "build_pairs_manifest") else None

    results: list[dict[str, Any]] = []
    failures = 0
    analysis_failures = 0

    attempted: set[tuple[int, str]] = set()
    for run in plan_to_run:
        attempted.add((run.run, run.variant))
        logger.info(
            "Running %s run=%d generation_id=%s seed=%s",
            run.variant,
            run.run,
            run.generation_id,
            str(run.seed) if run.seed is not None else "<none>",
        )
        try:
            ctx = run_generation(
                run.cfg_dict,
                generation_id=run.generation_id,
                config_meta=dict(cfg_meta) if isinstance(cfg_meta, Mapping) else None,
                update_artifacts_index=False,
            )
            outputs = ctx.outputs if isinstance(ctx.outputs, dict) else {}
            entry: dict[str, Any] = {
                "variant": run.variant,
                "variant_name": run.variant_name,
                "run": run.run,
                "generation_id": run.generation_id,
                "seed": run.seed,
                "status": "success",
                "image_path": getattr(ctx, "image_path", None),
                "final_prompt": outputs.get("image_prompt"),
                "outputs": {"prompt_pipeline": outputs.get("prompt_pipeline")},
                "error": ctx.error,
            }

            if hasattr(experiment, "analyze_run"):
                try:
                    analysis = experiment.analyze_run(run_spec=run, ctx=ctx)
                except Exception as analysis_exc:  # noqa: BLE001
                    analysis_failures += 1
                    entry["analysis_error"] = {
                        "type": analysis_exc.__class__.__name__,
                        "message": str(analysis_exc),
                    }
                    logger.exception("Analysis failed for generation_id=%s", run.generation_id)
                else:
                    if analysis is not None:
                        if not isinstance(analysis, Mapping):
                            raise TypeError(
                                f"{exp_name}.analyze_run returned non-mapping for generation_id={run.generation_id}"
                            )
                        entry["analysis"] = dict(analysis)

            results.append(entry)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if pairs_payload is not None:
                try:
                    record_pair_error(
                        pairs_payload,
                        run_index=run.run,
                        variant=run.variant,
                        error={"type": exc.__class__.__name__, "message": str(exc)},
                    )
                except Exception as pair_exc:  # noqa: BLE001
                    logger.error("Failed to record pair error: %s", pair_exc)

            results.append(
                {
                    "variant": run.variant,
                    "variant_name": run.variant_name,
                    "run": run.run,
                    "generation_id": run.generation_id,
                    "seed": run.seed,
                    "status": "error",
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                }
            )
            logger.exception("Run failed: %s", exc)
            if not continue_on_error:
                break

    if pairs_payload is not None and failures and not continue_on_error:
        for run in plan_to_run:
            key = (run.run, run.variant)
            if key in attempted:
                continue
            try:
                record_pair_error(
                    pairs_payload,
                    run_index=run.run,
                    variant=run.variant,
                    error={"type": "Skipped", "message": "Skipped due to an earlier failure"},
                )
            except Exception:
                pass

    results_path = os.path.join(output_root_final, "experiment_results.json")
    merged_results = results
    if resume and os.path.exists(results_path):
        try:
            existing = json.loads(Path(results_path).read_text(encoding="utf-8"))
            existing_list = existing.get("results") if isinstance(existing, dict) else None
            if isinstance(existing_list, list):
                by_id: dict[str, dict[str, Any]] = {}
                for item in existing_list:
                    if isinstance(item, dict):
                        gid = item.get("generation_id")
                        if isinstance(gid, str) and gid.strip():
                            by_id[gid.strip()] = item
                for item in results:
                    gid = item.get("generation_id")
                    if isinstance(gid, str) and gid.strip():
                        by_id[gid.strip()] = item
                merged_results = [by_id.get(run.generation_id) for run in plan if run.generation_id in by_id]
                merged_results = [item for item in merged_results if isinstance(item, dict)]
        except Exception:  # noqa: BLE001
            merged_results = results

    _write_json(results_path, {"schema_version": 1, "kind": "experiment_results", "results": merged_results})
    logger.info("Wrote experiment results: %s", results_path)

    if pairs_payload is not None:
        pairs_path = write_pairs_manifest(output_root_final, pairs_payload)
        logger.info("Wrote pairs manifest: %s", pairs_path)

    _update_artifacts_index_once(output_root=output_root_final, repo_root=repo_root, logger=logger)
    if failures or analysis_failures:
        return 1
    return 0


def _update_artifacts_index_once(*, output_root: str, repo_root: str | None, logger: logging.Logger) -> None:
    """Update the artifacts index once for the store containing this experiment."""

    output_path = Path(output_root).resolve()
    artifacts_root: Path | None = None
    for idx, part in enumerate(output_path.parts):
        if str(part).casefold() == "_artifacts":
            artifacts_root = Path(*output_path.parts[: idx + 1])
            break

    if artifacts_root is None:
        if repo_root:
            artifacts_root = Path(repo_root) / "_artifacts"
            logger.warning(
                "Output root is not under an _artifacts directory; defaulting artifacts_root=%s",
                str(artifacts_root),
            )
        else:
            logger.warning(
                "Output root is not under an _artifacts directory and repo_root is unknown; skipping index update"
            )
            return

    logger.info("Updating artifacts index under %s", str(artifacts_root))
    payload = update_artifacts_index(
        artifacts_root=str(artifacts_root),
        repo_root=str(repo_root) if repo_root else None,
    )
    counts = payload.get("counts") if isinstance(payload, dict) else None
    if isinstance(counts, dict):
        logger.info("Artifacts index updated: %s", json.dumps(counts, ensure_ascii=False))
