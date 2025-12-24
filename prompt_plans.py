from __future__ import annotations

import importlib
import pkgutil
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, TypeAlias

import pandas as pd

from pipeline import ALLOWED_MERGE_MODES, ActionStep, Block, ChatRunner, MergeMode, RunContext
from refinement import NoRefinement, RefinementPolicy, TotEnclaveRefinement
from run_config import RunConfig


class RefinementPolicyRegistry:
    _REGISTRY: dict[str, type[RefinementPolicy]] = {
        "tot": TotEnclaveRefinement,
        "none": NoRefinement,
    }

    @classmethod
    def normalize(cls, policy: str, *, path: str) -> str:
        if not isinstance(policy, str) or not policy.strip():
            raise ValueError(f"{path} must be a non-empty string")

        key = policy.strip().lower()
        if key not in cls._REGISTRY:
            raise ValueError(f"Unknown refinement policy for {path}: {policy}")
        return key

    @classmethod
    def get(cls, policy: str, *, path: str) -> RefinementPolicy:
        key = cls.normalize(policy, path=path)
        return cls._REGISTRY[key]()


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    prompt: str | Callable[[RunContext], str]
    temperature: float
    params: dict[str, Any] = field(default_factory=dict)
    allow_empty_prompt: bool = False
    allow_empty_response: bool = False
    tags: tuple[str, ...] = ()
    refinement_policy: str | None = None
    is_default_capture: bool = False
    merge: MergeMode = "last_response"
    output_key: str | None = None
    doc: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise TypeError("StageSpec.stage_id must be a non-empty string")
        object.__setattr__(self, "stage_id", self.stage_id.strip())

        if not (isinstance(self.prompt, str) or callable(self.prompt)):
            raise TypeError("StageSpec.prompt must be a string or callable")

        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("StageSpec.temperature must be a float")

        if not isinstance(self.params, dict):
            raise TypeError("StageSpec.params must be a dict")
        if "temperature" in self.params:
            raise ValueError(
                f"Stage {self.stage_id} sets params['temperature']; use the StageSpec.temperature field instead"
            )

        if self.merge not in ALLOWED_MERGE_MODES:
            raise ValueError(f"Invalid stage merge mode: {self.merge} (stage={self.stage_id})")

        if self.refinement_policy is not None:
            if not isinstance(self.refinement_policy, str) or not self.refinement_policy.strip():
                raise TypeError("StageSpec.refinement_policy must be a non-empty string if set")
            object.__setattr__(self, "refinement_policy", self.refinement_policy.strip())

        if self.output_key is not None:
            if not isinstance(self.output_key, str):
                raise TypeError(
                    f"StageSpec.output_key must be a string or None (type={type(self.output_key).__name__})"
                )
            output_key = self.output_key.strip()
            if not output_key:
                raise ValueError("StageSpec.output_key cannot be empty")
            object.__setattr__(self, "output_key", output_key)

        if self.doc is not None:
            if not isinstance(self.doc, str):
                raise TypeError(
                    f"StageSpec.doc must be a string or None (type={type(self.doc).__name__})"
                )
            doc = self.doc.strip()
            if not doc:
                raise ValueError("StageSpec.doc cannot be empty")
            object.__setattr__(self, "doc", doc)

        if self.source is not None:
            if not isinstance(self.source, str):
                raise TypeError(
                    f"StageSpec.source must be a string or None (type={type(self.source).__name__})"
                )
            source = self.source.strip()
            if not source:
                raise ValueError("StageSpec.source cannot be empty")
            object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class ActionStageSpec:
    stage_id: str
    fn: Callable[[RunContext], Any]
    merge: MergeMode = "none"
    tags: tuple[str, ...] = ()
    output_key: str | None = None
    doc: str | None = None
    source: str | None = None
    is_default_capture: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise TypeError("ActionStageSpec.stage_id must be a non-empty string")
        object.__setattr__(self, "stage_id", self.stage_id.strip())

        if not callable(self.fn):
            raise TypeError("ActionStageSpec.fn must be callable")

        if self.merge not in ALLOWED_MERGE_MODES:
            raise ValueError(
                f"Invalid stage merge mode: {self.merge} (stage={self.stage_id})"
            )
        if self.merge == "last_response":
            raise ValueError(
                f"Action stage cannot use merge='last_response' (stage={self.stage_id})"
            )

        if self.output_key is not None:
            if not isinstance(self.output_key, str):
                raise TypeError(
                    "ActionStageSpec.output_key must be a string or None "
                    f"(type={type(self.output_key).__name__})"
                )
            output_key = self.output_key.strip()
            if not output_key:
                raise ValueError("ActionStageSpec.output_key cannot be empty")
            object.__setattr__(self, "output_key", output_key)

        if self.doc is not None:
            if not isinstance(self.doc, str):
                raise TypeError(
                    "ActionStageSpec.doc must be a string or None "
                    f"(type={type(self.doc).__name__})"
                )
            doc = self.doc.strip()
            if not doc:
                raise ValueError("ActionStageSpec.doc cannot be empty")
            object.__setattr__(self, "doc", doc)

        if self.source is not None:
            if not isinstance(self.source, str):
                raise TypeError(
                    "ActionStageSpec.source must be a string or None "
                    f"(type={type(self.source).__name__})"
                )
            source = self.source.strip()
            if not source:
                raise ValueError("ActionStageSpec.source cannot be empty")
            object.__setattr__(self, "source", source)


StageNodeSpec: TypeAlias = StageSpec | ActionStageSpec


class StageOverrideLike(Protocol):
    temperature: float | None
    params: dict[str, Any] | None
    refinement_policy: str | None


@dataclass(frozen=True)
class ResolvedStages:
    stages: tuple[StageNodeSpec, ...]
    capture_stage: str
    metadata: dict[str, Any]

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)


def resolve_stage_specs(
    stage_specs: list[StageNodeSpec],
    *,
    plan_name: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    overrides: Mapping[str, StageOverrideLike],
    capture_stage: str | None,
) -> ResolvedStages:
    stage_ids = [spec.stage_id for spec in stage_specs]
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError(f"Plan {plan_name} defines duplicate stage ids: {stage_ids}")

    available = set(stage_ids)
    available_list = ", ".join(stage_ids) or "<none>"

    def normalize_stage_id(raw: str, *, path: str) -> str:
        if not isinstance(raw, str):
            raise TypeError(f"{path} must contain only strings")
        stage_id = raw.strip()
        if not stage_id:
            raise ValueError(f"{path} must not contain empty strings")
        if stage_id in available:
            return stage_id
        if "." in stage_id:
            raise ValueError(
                f"Unknown stage id in {path}: {stage_id} (plan={plan_name}, available={available_list})"
            )

        matches = sorted(s for s in available if s.endswith("." + stage_id))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous stage id in {path}: {stage_id} (matches: {', '.join(matches)})"
            )
        raise ValueError(
            f"Unknown stage id in {path}: {stage_id} (plan={plan_name}, available={available_list})"
        )

    include_list: list[str] = [normalize_stage_id(raw, path="prompt.stages.include") for raw in include]

    exclude_list: list[str] = []
    exclude_set: set[str] = set()
    for raw in exclude:
        normalized = normalize_stage_id(raw, path="prompt.stages.exclude")
        exclude_list.append(normalized)
        exclude_set.add(normalized)

    normalized_overrides: dict[str, StageOverrideLike] = {}
    override_sources: dict[str, str] = {}
    for raw_stage_id, override in overrides.items():
        stage_id = normalize_stage_id(raw_stage_id, path="prompt.stages.overrides")
        if stage_id in normalized_overrides:
            raise ValueError(
                "Duplicate stage override after stage id normalization: "
                f"{override_sources.get(stage_id, stage_id)} and {raw_stage_id} both map to {stage_id}"
            )
        normalized_overrides[stage_id] = override
        override_sources[stage_id] = raw_stage_id

    include_set = set(include_list)
    filtered: list[StageNodeSpec] = []
    for spec in stage_specs:
        if include_set and spec.stage_id not in include_set:
            continue
        if spec.stage_id in exclude_set:
            continue
        filtered.append(spec)

    if not filtered:
        raise ValueError(
            f"Resolved stage list is empty after applying prompt.stages include/exclude (plan={plan_name})"
        )

    resolved: list[StageNodeSpec] = []
    stage_overrides_summary: dict[str, dict[str, Any]] = {}

    for spec in filtered:
        override = normalized_overrides.get(spec.stage_id)
        if override is None:
            resolved.append(spec)
            continue
        if isinstance(spec, ActionStageSpec):
            raise ValueError(
                f"Stage override for {spec.stage_id} targets an action stage; "
                "action stages do not support temperature/params/refinement overrides"
            )

        next_temp = spec.temperature
        next_params = dict(spec.params)
        next_ref_policy = spec.refinement_policy

        override_summary: dict[str, Any] = {}
        if override.temperature is not None:
            next_temp = float(override.temperature)
            override_summary["temperature"] = next_temp
        if override.params is not None:
            if "temperature" in override.params:
                raise ValueError(
                    f"prompt.stages.overrides.{spec.stage_id}.params sets temperature; use .temperature instead"
                )
            next_params.update(dict(override.params))
            override_summary["params"] = dict(override.params)
        if override.refinement_policy is not None:
            next_ref_policy = RefinementPolicyRegistry.normalize(
                override.refinement_policy,
                path=f"prompt.stages.overrides.{spec.stage_id}.refinement_policy",
            )
            override_summary["refinement_policy"] = next_ref_policy

        if override_summary:
            stage_overrides_summary[spec.stage_id] = override_summary

        resolved.append(
            StageSpec(
                stage_id=spec.stage_id,
                prompt=spec.prompt,
                temperature=next_temp,
                params=next_params,
                allow_empty_prompt=spec.allow_empty_prompt,
                allow_empty_response=spec.allow_empty_response,
                tags=spec.tags,
                refinement_policy=next_ref_policy,
                is_default_capture=spec.is_default_capture,
                merge=spec.merge,
                output_key=spec.output_key,
                doc=spec.doc,
                source=spec.source,
            )
        )

    if capture_stage is not None:
        if not isinstance(capture_stage, str) or not capture_stage.strip():
            raise ValueError("prompt.output.capture_stage must be a non-empty string or null")
        capture = normalize_stage_id(capture_stage, path="prompt.output.capture_stage")
        if capture not in {spec.stage_id for spec in resolved}:
            raise ValueError(
                f"prompt.output.capture_stage={capture} is not in resolved stages: {', '.join(s.stage_id for s in resolved)}"
            )
        chosen_capture_stage = capture
    else:
        # Default capture: plan default if present, otherwise last resolved stage.
        plan_default = next((spec.stage_id for spec in stage_specs if spec.is_default_capture), None)
        chosen_capture_stage = (
            plan_default if plan_default in {spec.stage_id for spec in resolved} else resolved[-1].stage_id
        )

    metadata: dict[str, Any] = {
        "plan": plan_name,
        "include": list(include_list),
        "exclude": list(exclude_list),
        "capture_stage": chosen_capture_stage,
        "resolved_stages": [spec.stage_id for spec in resolved],
        "stage_overrides": stage_overrides_summary,
    }

    return ResolvedStages(stages=tuple(resolved), capture_stage=chosen_capture_stage, metadata=metadata)


def build_pipeline_block(
    stages: ResolvedStages,
    *,
    refinement_policy: str,
    refinement_registry: type[RefinementPolicyRegistry] = RefinementPolicyRegistry,
    capture_key: str = "image_prompt",
) -> Block:
    root_nodes: list[Block] = []
    for spec in stages.stages:
        effective_capture_key: str | None
        if spec.stage_id == stages.capture_stage:
            if spec.output_key is not None and spec.output_key != capture_key:
                raise ValueError(
                    "Capture stage output_key conflict: "
                    f"stage={spec.stage_id} output_key={spec.output_key} capture_key={capture_key}"
                )
            effective_capture_key = capture_key
        elif spec.output_key is not None:
            effective_capture_key = spec.output_key
        else:
            effective_capture_key = None

        stage_meta: dict[str, Any] = {}
        if spec.source:
            stage_meta["source"] = spec.source
        if spec.doc:
            stage_meta["doc"] = spec.doc

        if isinstance(spec, ActionStageSpec):
            root_nodes.append(
                Block(
                    name=spec.stage_id,
                    merge=spec.merge,
                    meta=stage_meta,
                    nodes=[
                        ActionStep(
                            name="action",
                            fn=spec.fn,
                            capture_key=effective_capture_key,
                        )
                    ],
                )
            )
            continue

        if spec.refinement_policy is not None:
            policy_name = refinement_registry.normalize(
                spec.refinement_policy,
                path=f"stage[{spec.stage_id}].refinement_policy",
            )
        else:
            policy_name = refinement_registry.normalize(
                refinement_policy, path="prompt.refinement.policy"
            )
        policy = refinement_registry.get(
            policy_name, path=f"resolved.stage[{spec.stage_id}].refinement_policy"
        )

        root_nodes.append(
            policy.stage(
                spec.stage_id,
                prompt=spec.prompt,
                temperature=spec.temperature,
                merge=spec.merge,
                allow_empty_prompt=spec.allow_empty_prompt,
                allow_empty_response=spec.allow_empty_response,
                params=spec.params,
                meta=stage_meta or None,
                capture_key=effective_capture_key,
            )
        )

    return Block(name="pipeline", merge="all_messages", nodes=root_nodes)


@dataclass(frozen=True)
class PlanInputs:
    cfg: RunConfig
    ai_text: Any
    prompt_data: pd.DataFrame
    user_profile: pd.DataFrame
    preferences_guidance: str
    context_guidance: str | None
    rng: random.Random
    draft_prompt: str | None = None


class PromptPlan(Protocol):
    name: str

    def stage_specs(self, inputs: PlanInputs) -> list[StageNodeSpec]:
        ...

    def execute(
        self, ctx: RunContext, runner: ChatRunner, resolved: ResolvedStages, inputs: PlanInputs
    ) -> None:
        ...


ContextInjectionMode = Literal["config", "disabled", "enabled"]


@dataclass(frozen=True)
class PlanMetadata:
    requires_scoring: bool | None = None
    context_injection: ContextInjectionMode = "config"
    required_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedPlan:
    requested_plan: str
    plan: PromptPlan
    metadata: PlanMetadata
    effective_context_enabled: bool


class LinearStagePlan:
    """Default executor for plans that are just a stage list."""

    def execute(
        self, ctx: RunContext, runner: ChatRunner, resolved: ResolvedStages, inputs: PlanInputs
    ) -> None:
        pipeline_root = build_pipeline_block(
            resolved,
            refinement_policy=ctx.cfg.prompt_refinement_policy,
        )
        runner.run(ctx, pipeline_root)


class SequencePromptPlan(LinearStagePlan):
    """Declarative plan: resolve a sequence of stage ids via the StageCatalog."""

    sequence: tuple[str, ...] = ()

    def stage_sequence(self, inputs: PlanInputs) -> tuple[str, ...]:
        return tuple(self.sequence)

    def stage_specs(self, inputs: PlanInputs) -> list[StageNodeSpec]:
        from stage_catalog import StageCatalog

        sequence = self.stage_sequence(inputs)
        if not sequence:
            raise ValueError(f"prompt.plan={self.name} produced an empty stage sequence")
        return [StageCatalog.build(stage_id, inputs) for stage_id in sequence]


_PLAN_REGISTRY: dict[str, type[PromptPlan]] = {}
_PLUGINS_DISCOVERED = False
_PLUGINS_IMPORT_ERROR: Exception | None = None


def register_plan(cls: type[PromptPlan]) -> type[PromptPlan]:
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise TypeError("Prompt plan must define a non-empty 'name' attribute")

    key = name.strip().lower()
    if key in _PLAN_REGISTRY:
        raise ValueError(f"Duplicate prompt plan name: {key}")

    _PLAN_REGISTRY[key] = cls
    return cls


def _discover_plugins() -> None:
    global _PLUGINS_DISCOVERED, _PLUGINS_IMPORT_ERROR
    if _PLUGINS_DISCOVERED:
        return

    try:
        import plan_plugins  # noqa: F401
    except ModuleNotFoundError as exc:
        _PLUGINS_IMPORT_ERROR = exc
        _PLUGINS_DISCOVERED = True
        return
    except Exception as exc:  # pragma: no cover - defensive
        _PLUGINS_IMPORT_ERROR = exc
        _PLUGINS_DISCOVERED = True
        raise

    import plan_plugins as _plan_plugins

    try:
        for module in pkgutil.iter_modules(
            _plan_plugins.__path__, prefix=_plan_plugins.__name__ + "."
        ):
            importlib.import_module(module.name)
    except Exception as exc:
        _PLUGINS_IMPORT_ERROR = exc
        _PLUGINS_DISCOVERED = True
        raise

    _PLUGINS_DISCOVERED = True


class PromptPlanManager:
    @classmethod
    def get(cls, name: str) -> PromptPlan:
        _discover_plugins()
        key = (name or "").strip().lower()
        if not key:
            raise ValueError("Unknown prompt plan: <empty>")
        plan_cls = _PLAN_REGISTRY.get(key)
        if plan_cls is None:
            available = ", ".join(sorted(_PLAN_REGISTRY.keys())) or "<none>"
            hint = ""
            if isinstance(_PLUGINS_IMPORT_ERROR, ModuleNotFoundError):
                missing = getattr(_PLUGINS_IMPORT_ERROR, "name", None)
                if missing == "plan_plugins":
                    hint = (
                        " (plan_plugins package not found; if you expected plugin plans, "
                        "ensure ./plan_plugins is present and importable)"
                    )
            elif _PLUGINS_IMPORT_ERROR is not None:
                hint = f" (plan plugin discovery failed: {_PLUGINS_IMPORT_ERROR})"
            raise ValueError(f"Unknown prompt plan: {name} (available: {available}){hint}")
        return plan_cls()

    @classmethod
    def available(cls) -> tuple[str, ...]:
        _discover_plugins()
        return tuple(sorted(_PLAN_REGISTRY.keys()))

    @classmethod
    def resolve(cls, cfg: RunConfig) -> ResolvedPlan:
        requested = (cfg.prompt_plan or "").strip().lower()
        if not requested:
            raise ValueError("Invalid config value for prompt.plan: must be a non-empty string")

        scoring_enabled = bool(cfg.prompt_scoring.enabled)

        if requested == "auto":
            plan_name = "blackbox" if scoring_enabled else "standard"
        else:
            plan_name = requested

        plan = cls.get(plan_name)
        metadata = cls._read_metadata(plan)

        if metadata.requires_scoring is True and not scoring_enabled:
            raise ValueError(f"prompt.plan={plan.name} requires prompt.scoring.enabled=true")
        if metadata.requires_scoring is False and scoring_enabled:
            raise ValueError(f"prompt.plan={plan.name} conflicts with prompt.scoring.enabled=true")

        if metadata.context_injection == "disabled":
            effective_context_enabled = False
        elif metadata.context_injection == "enabled":
            if not cfg.context_enabled:
                raise ValueError(f"Plan {plan.name} requires context.enabled=true")
            effective_context_enabled = True
        else:
            effective_context_enabled = bool(cfg.context_enabled)

        return ResolvedPlan(
            requested_plan=requested,
            plan=plan,
            metadata=metadata,
            effective_context_enabled=effective_context_enabled,
        )

    @classmethod
    def _read_metadata(cls, plan: PromptPlan) -> PlanMetadata:
        raw_context_injection = getattr(plan, "context_injection", "config")
        if raw_context_injection is None:
            raw_context_injection = "config"
        if not isinstance(raw_context_injection, str):
            raise TypeError(f"Plan {plan.name}.context_injection must be a string if set")
        context_injection = raw_context_injection.strip().lower() or "config"
        if context_injection not in ("config", "disabled", "enabled"):
            raise ValueError(
                f"Plan {plan.name}.context_injection must be one of: config, disabled, enabled"
            )

        requires_scoring = getattr(plan, "requires_scoring", None)
        if requires_scoring is not None and not isinstance(requires_scoring, bool):
            raise TypeError(f"Plan {plan.name}.requires_scoring must be a bool if set")

        required_inputs_raw = getattr(plan, "required_inputs", ())
        required_inputs: tuple[str, ...]
        if required_inputs_raw is None:
            required_inputs = ()
        elif isinstance(required_inputs_raw, (list, tuple)):
            items: list[str] = []
            for item in required_inputs_raw:
                if not isinstance(item, str) or not item.strip():
                    raise TypeError(
                        f"Plan {plan.name}.required_inputs must contain only non-empty strings"
                    )
                items.append(item.strip())
            required_inputs = tuple(items)
        else:
            raise TypeError(
                f"Plan {plan.name}.required_inputs must be a list[str] or tuple[str, ...] if set"
            )

        return PlanMetadata(
            requires_scoring=requires_scoring,
            context_injection=context_injection,  # type: ignore[arg-type]
            required_inputs=required_inputs,
        )


@register_plan
class CustomPromptPlan(SequencePromptPlan):
    name = "custom"

    def stage_sequence(self, inputs: PlanInputs) -> tuple[str, ...]:
        sequence = tuple(inputs.cfg.prompt_stages_sequence)
        if not sequence:
            raise ValueError("prompt.plan=custom requires prompt.stages.sequence")
        return sequence


@register_plan
class StandardPromptPlan(SequencePromptPlan):
    name = "standard"
    requires_scoring = False
    sequence = (
        "preprompt.select_concepts",
        "preprompt.filter_concepts",
        "standard.initial_prompt",
        "standard.section_2_choice",
        "standard.section_2b_title_and_story",
        "standard.section_3_message_focus",
        "standard.section_4_concise_description",
        "standard.image_prompt_creation",
    )


@register_plan
class BlackboxPromptPlan(SequencePromptPlan):
    name = "blackbox"
    requires_scoring = True

    def stage_sequence(self, inputs: PlanInputs) -> tuple[str, ...]:
        scoring_cfg = inputs.cfg.prompt_scoring
        sequence: list[str] = [
            "preprompt.select_concepts",
            "preprompt.filter_concepts",
            "blackbox.prepare",
        ]
        if scoring_cfg.generator_profile_abstraction:
            sequence.append("blackbox.profile_abstraction")
        sequence.extend(
            (
                "blackbox.idea_cards_generate",
                "blackbox.idea_cards_judge_score",
                "blackbox.select_idea_card",
                "blackbox.image_prompt_creation",
            )
        )
        return tuple(sequence)


@register_plan
class RefineOnlyPromptPlan(SequencePromptPlan):
    name = "refine_only"
    requires_scoring = False
    required_inputs = ("draft_prompt",)
    sequence = ("refine.image_prompt_refine",)

    def stage_specs(self, inputs: PlanInputs) -> list[StageNodeSpec]:
        draft_text = (inputs.draft_prompt or "").strip()
        if not draft_text:
            raise ValueError("prompt.plan=refine_only requires prompt.refine_only.draft or draft_path")
        return super().stage_specs(inputs)
