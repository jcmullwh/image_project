from __future__ import annotations

"""Experiment registry + plugin interfaces for the current implementation.

This module provides:

- A small, explicit interface for "first-class experiments" that can be invoked
  via a single canonical runner (see `image_project.app.experiment_runner`).
- A registry + plugin discovery mechanism, mirroring the prompt plan plugin
  approach (`image_project.impl.current.plans`).

Experiments are implementation glue: they compose config overlays, define
variant/run matrices, and (optionally) add post-run analysis. They must not be a
dumping ground for prompt templates; prompt text belongs in `image_project.prompts`
and stage wiring belongs in `image_project.stages`.
"""

import importlib
import pkgutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from image_project.framework.runtime import RunContext


@dataclass(frozen=True)
class RunSpec:
    """One planned run within an experiment.

    Attributes:
        variant: Short variant key used for grouping (e.g. "A", "B", "C").
        variant_name: Optional human label for the variant (e.g. "baseline").
        run: 1-based run index within the variant group.
        generation_id: Generation id used for artifact filenames and indexing.
        seed: Optional prompt seed. When provided, experiments should set
            `prompt.random_seed` to this value so runs are reproducible.
        cfg_dict: Fully merged config mapping to pass to `run_generation(...)`.
            This must not include runner-only keys like `experiment_runners`.
        meta: Optional experiment-specific metadata recorded in plan/results.
    """

    variant: str
    variant_name: str | None
    run: int
    generation_id: str
    seed: int | None
    cfg_dict: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Experiment(Protocol):
    """Interface for an experiment plugin."""

    name: str
    summary: str
    default_run_mode: str

    def add_cli_args(self, parser: Any) -> None:
        """Register experiment-specific CLI arguments on the provided parser."""

    def build_plan(
        self,
        *,
        base_cfg: Mapping[str, Any],
        runner_cfg: Mapping[str, Any],
        output_root: str,
        experiment_id: str,
        run_mode: str,
        cli_args: Mapping[str, Any],
    ) -> tuple[list[RunSpec], dict[str, Any]]:
        """Build a run plan.

        Args:
            base_cfg: Base config mapping with standard runner overrides already
                applied (output dirs, run mode, experiment.id, etc).
            runner_cfg: Runner-specific config block from `experiment_runners`.
            output_root: Absolute experiment directory path.
            experiment_id: Experiment id to record in artifacts/transcripts.
            run_mode: "full" or "prompt_only".
            cli_args: Parsed experiment-specific CLI args (plugin-owned).

        Returns:
            (runs, summary_metadata)

            - runs: List of `RunSpec` entries, one per planned run.
            - summary_metadata: Extra JSON-serializable fields to merge into the
              top-level `experiment_plan.json` payload (e.g. base_seed).
        """

    def analyze_run(self, *, run_spec: RunSpec, ctx: RunContext) -> dict[str, Any] | None:
        """Optional post-run analysis hook.

        Implementations should return a JSON-serializable mapping, or None to
        record no analysis for this run.
        """

        return None

    def build_pairs_manifest(
        self,
        plan: Sequence[RunSpec],
        *,
        experiment_id: str,
        run_mode: str,
    ) -> dict[str, Any] | None:
        """Optional pairing manifest generator (e.g. A/B experiments).

        If implemented, the runner will write the returned payload to
        `<output_root>/pairs.json`.
        """

        return None


class ExperimentBase:
    """Convenience base class for experiments.

    Experiment plugins should generally inherit from this class so optional hooks
    (`analyze_run`, `build_pairs_manifest`) have predictable defaults.
    """

    name: str = ""
    summary: str = ""
    default_run_mode: str = ""

    def add_cli_args(self, parser: Any) -> None:
        """Register experiment-specific CLI args (default: none)."""

    def build_plan(
        self,
        *,
        base_cfg: Mapping[str, Any],
        runner_cfg: Mapping[str, Any],
        output_root: str,
        experiment_id: str,
        run_mode: str,
        cli_args: Mapping[str, Any],
    ) -> tuple[list[RunSpec], dict[str, Any]]:
        """Build a run plan (required)."""

        raise NotImplementedError

    def analyze_run(self, *, run_spec: RunSpec, ctx: RunContext) -> dict[str, Any] | None:
        """Optional post-run analysis hook (default: none)."""

        return None

    def build_pairs_manifest(
        self,
        plan: Sequence[RunSpec],
        *,
        experiment_id: str,
        run_mode: str,
    ) -> dict[str, Any] | None:
        """Optional pairing manifest generator (default: none)."""

        return None


_EXPERIMENT_REGISTRY: dict[str, type[Experiment]] = {}
_PLUGINS_DISCOVERED = False
_PLUGINS_IMPORT_ERROR: Exception | None = None


def register_experiment(cls: type[Experiment]) -> type[Experiment]:
    """Class decorator to register an experiment plugin."""

    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise TypeError("Experiment must define a non-empty 'name' attribute")

    key = name.strip().lower()
    if key in _EXPERIMENT_REGISTRY:
        raise ValueError(f"Duplicate experiment name: {key}")

    summary = getattr(cls, "summary", None)
    if not isinstance(summary, str) or not summary.strip():
        raise TypeError(f"Experiment {name!r} must define a non-empty 'summary' attribute")

    default_run_mode = getattr(cls, "default_run_mode", None)
    if default_run_mode not in {"full", "prompt_only"}:
        raise TypeError(
            f"Experiment {name!r} must define default_run_mode as 'full' or 'prompt_only' "
            f"(got {default_run_mode!r})"
        )

    build_plan = getattr(cls, "build_plan", None)
    if not callable(build_plan):
        raise TypeError(f"Experiment {name!r} must define a callable build_plan(...) method")

    add_cli_args = getattr(cls, "add_cli_args", None)
    if not callable(add_cli_args):
        raise TypeError(f"Experiment {name!r} must define a callable add_cli_args(parser) method")

    _EXPERIMENT_REGISTRY[key] = cls
    return cls


def _discover_plugins() -> None:
    """Import all experiment plugin modules exactly once."""

    global _PLUGINS_DISCOVERED, _PLUGINS_IMPORT_ERROR
    if _PLUGINS_DISCOVERED:
        return

    try:
        from image_project.impl.current import experiment_plugins as _experiment_plugins
    except Exception as exc:
        _PLUGINS_IMPORT_ERROR = exc
        _PLUGINS_DISCOVERED = True
        raise

    try:
        for module in pkgutil.iter_modules(
            _experiment_plugins.__path__,
            prefix=_experiment_plugins.__name__ + ".",
        ):
            importlib.import_module(module.name)
    except Exception as exc:
        _PLUGINS_IMPORT_ERROR = exc
        _PLUGINS_DISCOVERED = True
        raise

    _PLUGINS_DISCOVERED = True


class ExperimentManager:
    """Lookup helpers for registered experiment plugins."""

    @classmethod
    def available(cls) -> tuple[str, ...]:
        """List available experiment names."""

        _discover_plugins()
        return tuple(sorted(_EXPERIMENT_REGISTRY.keys()))

    @classmethod
    def get(cls, name: str) -> Experiment:
        """Get an experiment instance by name."""

        _discover_plugins()
        key = (name or "").strip().lower()
        if not key:
            raise ValueError("Unknown experiment: <empty>")
        exp_cls = _EXPERIMENT_REGISTRY.get(key)
        if exp_cls is None:
            available = ", ".join(sorted(_EXPERIMENT_REGISTRY.keys())) or "<none>"
            hint = ""
            if _PLUGINS_IMPORT_ERROR is not None:
                hint = f" (plugin import error: {_PLUGINS_IMPORT_ERROR.__class__.__name__}: {_PLUGINS_IMPORT_ERROR})"
            raise ValueError(f"Unknown experiment: {name!r}. Available: {available}{hint}")
        return exp_cls()
