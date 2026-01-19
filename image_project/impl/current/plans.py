from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from image_project.framework.config import RunConfig
from image_project.framework.prompt_pipeline import (
    PlanInputs,
    PlanMetadata,
    PromptPlan,
    ResolvedPlan,
)
from image_project.impl.current.blackbox_idea_cards import build_blackbox_isolated_idea_card_instances
from image_project.stages.blackbox.idea_cards_judge_score import (
    STAGE as BLACKBOX_IDEA_CARDS_JUDGE_SCORE,
)
from image_project.stages.blackbox.image_prompt_creation import (
    STAGE as BLACKBOX_IMAGE_PROMPT_CREATION,
)
from image_project.stages.blackbox.prepare import STAGE as BLACKBOX_PREPARE
from image_project.stages.blackbox.profile_abstraction import (
    STAGE as BLACKBOX_PROFILE_ABSTRACTION,
)
from image_project.stages.blackbox.profile_hints_load import (
    STAGE as BLACKBOX_PROFILE_HINTS_LOAD,
)
from image_project.stages.blackbox.select_idea_card import STAGE as BLACKBOX_SELECT_IDEA_CARD
from image_project.stages.preprompt.filter_concepts import STAGE as PREPROMPT_FILTER_CONCEPTS
from image_project.stages.preprompt.select_concepts import STAGE as PREPROMPT_SELECT_CONCEPTS
from image_project.stages.refine.image_prompt_refine import STAGE as REFINE_IMAGE_PROMPT_REFINE
from image_project.stages.refine.tot_enclave import STAGE as REFINE_TOT_ENCLAVE
from image_project.stages.registry import get_stage_registry
from image_project.stages.standard.image_prompt_creation import (
    STAGE as STANDARD_IMAGE_PROMPT_CREATION,
)
from image_project.stages.standard.initial_prompt import STAGE as STANDARD_INITIAL_PROMPT
from image_project.stages.standard.section_2_choice import STAGE as STANDARD_SECTION_2_CHOICE
from image_project.stages.standard.section_2b_title_and_story import (
    STAGE as STANDARD_SECTION_2B_TITLE_AND_STORY,
)
from image_project.stages.standard.section_3_message_focus import (
    STAGE as STANDARD_SECTION_3_MESSAGE_FOCUS,
)
from image_project.stages.standard.section_4_concise_description import (
    STAGE as STANDARD_SECTION_4_CONCISE_DESCRIPTION,
)
from pipelinekit.stage_types import StageInstance, StageRef


class SequencePromptPlan:
    """Declarative plan: return a sequence of stage nodes."""

    name: str
    sequence: tuple[Any, ...] = ()

    def stage_sequence(self, inputs: PlanInputs) -> tuple[Any, ...]:
        return tuple(self.sequence)

    def stage_nodes(self, inputs: PlanInputs) -> list[StageInstance]:
        sequence = self.stage_sequence(inputs)
        if not sequence:
            raise ValueError(f"prompt.plan={self.name} produced an empty stage sequence")
        nodes: list[StageInstance] = []
        for idx, entry in enumerate(sequence):
            if isinstance(entry, StageInstance):
                nodes.append(entry)
                continue
            if isinstance(entry, StageRef):
                nodes.append(entry.instance())
                continue
            raise TypeError(
                f"prompt.plan={self.name} stage_sequence[{idx}] must be StageRef|StageInstance (type={type(entry).__name__})"
            )
        return nodes


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
        from image_project.impl.current import plan_plugins as _plan_plugins
    except ModuleNotFoundError as exc:
        _PLUGINS_IMPORT_ERROR = exc
        _PLUGINS_DISCOVERED = True
        return
    except Exception as exc:  # pragma: no cover - defensive
        _PLUGINS_IMPORT_ERROR = exc
        _PLUGINS_DISCOVERED = True
        raise

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
                hint = f" (plan plugin package not found: {missing})" if missing else ""
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


def list_plans() -> None:
    for name in PromptPlanManager.available():
        print(name)


@register_plan
class CustomPromptPlan:
    name = "custom"

    def stage_nodes(self, inputs: PlanInputs) -> list[StageInstance]:
        sequence = tuple(inputs.cfg.prompt_stages_sequence)
        if not sequence:
            raise ValueError("prompt.plan=custom requires prompt.stages.sequence")

        registry = get_stage_registry()
        nodes: list[StageInstance] = []

        def resolve_stage(stage_id: str, *, path: str) -> StageRef:
            try:
                return registry.resolve(stage_id)
            except ValueError as exc:
                suggestions = registry.suggest(stage_id, limit=8)
                hint = f" (did you mean: {', '.join(suggestions)})" if suggestions else ""
                raise ValueError(f"{path}={stage_id!r} is invalid{hint}") from exc

        for idx, entry in enumerate(sequence):
            if isinstance(entry, str):
                ref = resolve_stage(entry, path=f"prompt.stages.sequence[{idx}]")
                nodes.append(ref.instance())
                continue
            if isinstance(entry, dict):
                stage = entry.get("stage")
                name = entry.get("name")
                if not isinstance(stage, str) or not stage.strip():
                    raise ValueError(f"prompt.stages.sequence[{idx}].stage missing/invalid")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"prompt.stages.sequence[{idx}].name missing/invalid")
                ref = resolve_stage(stage, path=f"prompt.stages.sequence[{idx}].stage")
                nodes.append(ref.instance(name.strip()))
                continue
            raise TypeError(
                f"prompt.stages.sequence[{idx}] must be a string or mapping (type={type(entry).__name__})"
            )

        return nodes


@register_plan
class StandardPromptPlan(SequencePromptPlan):
    name = "standard"
    requires_scoring = False
    sequence = (
        PREPROMPT_SELECT_CONCEPTS,
        PREPROMPT_FILTER_CONCEPTS,
        STANDARD_INITIAL_PROMPT,
        STANDARD_SECTION_2_CHOICE,
        STANDARD_SECTION_2B_TITLE_AND_STORY,
        STANDARD_SECTION_3_MESSAGE_FOCUS,
        STANDARD_SECTION_4_CONCISE_DESCRIPTION,
        STANDARD_IMAGE_PROMPT_CREATION,
        REFINE_TOT_ENCLAVE,
    )


@register_plan
class BlackboxPromptPlan:
    name = "blackbox"
    requires_scoring = True

    def stage_nodes(self, inputs: PlanInputs) -> list[StageInstance]:
        scoring_cfg = inputs.cfg.prompt_scoring
        sequence: list[StageInstance] = [
            PREPROMPT_SELECT_CONCEPTS.instance(),
            PREPROMPT_FILTER_CONCEPTS.instance(),
            BLACKBOX_PREPARE.instance(),
        ]
        if scoring_cfg.generator_profile_hints_path:
            sequence.append(BLACKBOX_PROFILE_HINTS_LOAD.instance())
        elif scoring_cfg.generator_profile_abstraction:
            sequence.append(BLACKBOX_PROFILE_ABSTRACTION.instance())

        sequence.extend(build_blackbox_isolated_idea_card_instances(inputs))

        sequence.extend(
            [
                BLACKBOX_IDEA_CARDS_JUDGE_SCORE.instance(),
                BLACKBOX_SELECT_IDEA_CARD.instance(),
                BLACKBOX_IMAGE_PROMPT_CREATION.instance(),
            ]
        )
        return sequence


@register_plan
class RefineOnlyPromptPlan(SequencePromptPlan):
    name = "refine_only"
    requires_scoring = False
    required_inputs = ("draft_prompt",)
    sequence = (REFINE_IMAGE_PROMPT_REFINE,)

    def stage_nodes(self, inputs: PlanInputs) -> list[StageInstance]:
        draft_text = (inputs.draft_prompt or "").strip()
        if not draft_text:
            raise ValueError(
                "prompt.plan=refine_only requires prompt.refine_only.draft or draft_path"
            )
        return super().stage_nodes(inputs)
