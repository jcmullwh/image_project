from __future__ import annotations

"""Prompt pipeline authoring + compilation glue.

This module is about *pipeline structure* (plans, stage blocks, compilation, and
image-project conventions for selectors/overrides/capture), not prompt text
templates. Prompt policy and templates live under `image_project.prompts.*`.
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, TypeAlias

import pandas as pd

import pipelinekit.compiler as pipelinekit_compiler

from pipelinekit.compiler import StageOverrideLike
from pipelinekit.engine.pipeline import (
    ALLOWED_MERGE_MODES,
    ActionStep,
    Block,
    ChatStep,
    MergeMode,
)
from image_project.framework.config import RunConfig
from image_project.framework.runtime import RunContext
from pipelinekit.stage_registry import StageRegistry
from pipelinekit.stage_types import StageInstance

from .stage_policies import (
    IMAGE_PROJECT_CAPTURE_POLICY,
    IMAGE_PROJECT_OVERRIDE_POLICY,
    IMAGE_PROJECT_SELECTOR_RESOLVER,
)

StageNode: TypeAlias = StageInstance


def _normalized_optional_str(value: str | None, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string or None (type={type(value).__name__})")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"{path} cannot be empty")
    return trimmed


def make_chat_stage_block(
    stage_id: str,
    *,
    prompt: str | Callable[[RunContext], str],
    temperature: float,
    merge: MergeMode = "last_response",
    params: dict[str, Any] | None = None,
    allow_empty_prompt: bool = False,
    allow_empty_response: bool = False,
    step_capture_key: str | None = None,
    doc: str | None = None,
    source: str | None = None,
    tags: tuple[str, ...] = (),
) -> Block:
    if not isinstance(stage_id, str) or not stage_id.strip():
        raise TypeError("stage_id must be a non-empty string")
    stage_id = stage_id.strip()

    if not (isinstance(prompt, str) or callable(prompt)):
        raise TypeError("prompt must be a string or callable")

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a float")

    if merge not in ALLOWED_MERGE_MODES:
        raise ValueError(f"Invalid stage merge mode: {merge} (stage={stage_id})")

    if params is not None:
        if not isinstance(params, dict):
            raise TypeError(
                f"params must be a dict if provided (type={type(params).__name__})"
            )
        if "temperature" in params:
            raise ValueError(
                f"Stage {stage_id} sets params['temperature']; use the temperature argument instead"
            )

    stage_meta: dict[str, Any] = {}
    normalized_doc = _normalized_optional_str(doc, path=f"stage[{stage_id}].doc")
    if normalized_doc:
        stage_meta["doc"] = normalized_doc
    normalized_source = _normalized_optional_str(source, path=f"stage[{stage_id}].source")
    if normalized_source:
        stage_meta["source"] = normalized_source
    if tags:
        stage_meta["tags"] = list(tags)

    draft_step = ChatStep(
        name="draft",
        prompt=prompt,
        temperature=float(temperature),
        allow_empty_prompt=bool(allow_empty_prompt),
        allow_empty_response=bool(allow_empty_response),
        capture_key=step_capture_key,
        params=dict(params) if params else {},
        meta={"role": "primary"},
    )

    return Block(
        name=stage_id,
        merge=merge,
        nodes=[draft_step],
        meta=stage_meta,
    )


def make_action_stage_block(
    stage_id: str,
    *,
    fn: Callable[[RunContext], Any],
    merge: MergeMode = "none",
    step_capture_key: str | None = None,
    doc: str | None = None,
    source: str | None = None,
    tags: tuple[str, ...] = (),
) -> Block:
    if not isinstance(stage_id, str) or not stage_id.strip():
        raise TypeError("stage_id must be a non-empty string")
    stage_id = stage_id.strip()

    if not callable(fn):
        raise TypeError("fn must be callable")

    if merge not in ALLOWED_MERGE_MODES:
        raise ValueError(f"Invalid stage merge mode: {merge} (stage={stage_id})")

    stage_meta: dict[str, Any] = {}
    normalized_doc = _normalized_optional_str(doc, path=f"stage[{stage_id}].doc")
    if normalized_doc:
        stage_meta["doc"] = normalized_doc
    normalized_source = _normalized_optional_str(source, path=f"stage[{stage_id}].source")
    if normalized_source:
        stage_meta["source"] = normalized_source
    if tags:
        stage_meta["tags"] = list(tags)

    return Block(
        name=stage_id,
        merge=merge,
        nodes=[
            ActionStep(
                name="action",
                fn=fn,
                capture_key=step_capture_key,
            )
        ],
        meta=stage_meta,
    )


@dataclass(frozen=True)
class CompiledStageBlocks:
    blocks: tuple[Block, ...]
    metadata: dict[str, Any]
    overrides: Mapping[str, StageOverrideLike]


def compile_stage_nodes(
    stage_nodes: list[StageNode],
    *,
    plan_name: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    overrides: Mapping[str, StageOverrideLike],
    stage_configs_defaults: Mapping[str, Mapping[str, Any]],
    stage_configs_instances: Mapping[str, Mapping[str, Any]],
    initial_outputs: Iterable[str] = (),
    stage_registry: StageRegistry,
    inputs: PlanInputs,
) -> CompiledStageBlocks:
    compiled = pipelinekit_compiler.compile_stage_nodes(
        stage_nodes,
        plan_name=plan_name,
        include=include,
        exclude=exclude,
        overrides=overrides,
        stage_configs_defaults=stage_configs_defaults,
        stage_configs_instances=stage_configs_instances,
        initial_outputs=tuple(initial_outputs),
        stage_registry=stage_registry,
        inputs=inputs,
        selector_resolver=IMAGE_PROJECT_SELECTOR_RESOLVER,
    )
    return CompiledStageBlocks(
        blocks=compiled.blocks, metadata=compiled.metadata, overrides=compiled.overrides
    )


def _stage_id(block: Block, *, path: str) -> str:
    if not isinstance(block, Block):
        raise TypeError(f"{path} must be a Block (type={type(block).__name__})")
    if not isinstance(block.name, str) or not block.name.strip():
        raise ValueError(f"{path} must have a non-empty Block.name (stage id)")
    return block.name.strip()


@dataclass(frozen=True)
class ResolvedStages:
    stages: tuple[Block, ...]
    capture_stage: str
    metadata: dict[str, Any]

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(_stage_id(block, path="resolved_stages.stages[]") for block in self.stages)


def resolve_stage_blocks(
    stage_blocks: list[Block],
    *,
    plan_name: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    overrides: Mapping[str, StageOverrideLike],
    capture_stage: str | None,
    capture_key: str = "image_prompt",
) -> ResolvedStages:
    resolved = pipelinekit_compiler.resolve_stage_blocks(
        stage_blocks,
        plan_name=plan_name,
        include=include,
        exclude=exclude,
        overrides=overrides,
        capture_stage=capture_stage,
        capture_key=capture_key,
        selector_resolver=IMAGE_PROJECT_SELECTOR_RESOLVER,
        override_policy=IMAGE_PROJECT_OVERRIDE_POLICY,
        capture_policy=IMAGE_PROJECT_CAPTURE_POLICY,
    )
    return ResolvedStages(
        stages=resolved.stages, capture_stage=resolved.capture_stage, metadata=resolved.metadata
    )


def make_pipeline_root_block(resolved: ResolvedStages) -> Block:
    return Block(name="pipeline", merge="all_messages", nodes=list(resolved.stages))


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

    def stage_nodes(self, inputs: PlanInputs) -> list[StageNode]:
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
