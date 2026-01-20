from __future__ import annotations

"""Generic stage compilation and patching.

This module intentionally contains no application-specific conventions. Callers must
inject their conventions via explicit policy objects:

- selector resolution (include/exclude/overrides/capture-stage selectors)
- override application targeting + patch semantics
- capture-stage default selection and eligibility validation
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, TypeAlias

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.pipeline import ActionStep, Block, ChatStep
from pipelinekit.stage_registry import StageRegistry
from pipelinekit.stage_types import StageInstance

Node: TypeAlias = ChatStep | ActionStep | Block
StageNode: TypeAlias = StageInstance

SelectorContext = Literal["sequence_item", "mapping_key"]

logger = logging.getLogger(__name__)


class StageOverrideLike(Protocol):
    temperature: float | None
    params: dict[str, Any] | None


class StageSelectorResolver(Protocol):
    def normalize(
        self,
        raw: Any,
        *,
        available: set[str],
        available_list: str,
        plan_name: str,
        path: str,
        context: SelectorContext,
    ) -> str:
        """Resolve a user-provided selector to a concrete stage instance id."""


class StageOverridePolicy(Protocol):
    def apply(
        self,
        stage_block: Block,
        *,
        stage_id: str,
        override: StageOverrideLike,
        override_source: str,
    ) -> tuple[Block, dict[str, Any] | None]:
        """Apply an override to a stage block and return (updated_block, summary_or_none)."""


class CapturePolicy(Protocol):
    def choose_default_capture_stage(self, stage_blocks: tuple[Block, ...], *, plan_name: str) -> str:
        """Select a default capture stage id (when not configured)."""

    def validate_capture_stage(self, stage_block: Block, *, stage_id: str) -> None:
        """Raise if the stage block cannot be used as a capture stage."""

    def attach_capture(self, stage_block: Block, *, stage_id: str, capture_key: str) -> Block:
        """Return a stage block configured to capture its final output."""


def _stage_node_id(node: StageNode, *, path: str) -> str:
    stage_id = (node.instance_id or "").strip()
    if not stage_id:
        raise ValueError(f"{path} StageInstance.instance_id must be non-empty")
    return stage_id


def _deep_merge_stage_config(base: Any, overlay: Any, *, path: str) -> Any:
    if overlay is None:
        return None
    if base is None:
        return overlay
    if isinstance(base, Mapping):
        if not isinstance(overlay, Mapping):
            raise ValueError(
                f"Invalid stage config merge at {path}: base is mapping but overlay is {type(overlay).__name__}"
            )
        merged: dict[str, Any] = dict(base)
        for key, overlay_value in overlay.items():
            next_path = f"{path}.{key}"
            if key in base:
                merged[key] = _deep_merge_stage_config(base[key], overlay_value, path=next_path)
            else:
                merged[key] = overlay_value
        return merged
    if isinstance(base, (list, tuple)):
        if not isinstance(overlay, (list, tuple)):
            raise ValueError(
                f"Invalid stage config merge at {path}: base is list but overlay is {type(overlay).__name__}"
            )
        return list(overlay)
    if isinstance(overlay, (Mapping, list, tuple)):
        raise ValueError(
            f"Invalid stage config merge at {path}: base is {type(base).__name__} but overlay is {type(overlay).__name__}"
        )
    return overlay


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
    initial_outputs: tuple[str, ...] | list[str] | set[str] = (),
    stage_registry: StageRegistry,
    inputs: Any,
    selector_resolver: StageSelectorResolver,
) -> CompiledStageBlocks:
    """Compile stage nodes into concrete stage blocks with strict IO + config validation."""

    for idx, node in enumerate(stage_nodes):
        if not isinstance(node, StageInstance):
            raise TypeError(
                f"prompt.plan={plan_name} stage_nodes[{idx}] must be a StageInstance "
                f"(type={type(node).__name__}, value={node!r})"
            )

    plan_stage_ids = [_stage_node_id(node, path="plan.stage_nodes[]") for node in stage_nodes]
    duplicates: set[str] = set()
    seen_stage_ids: set[str] = set()
    for stage_id in plan_stage_ids:
        if stage_id in seen_stage_ids:
            duplicates.add(stage_id)
        seen_stage_ids.add(stage_id)
    if duplicates:
        raise ValueError(f"Duplicate stage id: {sorted(duplicates)[0]}")

    available = set(plan_stage_ids)
    available_list = ", ".join(plan_stage_ids) or "<none>"

    include_list = [
        selector_resolver.normalize(
            raw,
            available=available,
            available_list=available_list,
            plan_name=plan_name,
            path="prompt.stages.include",
            context="sequence_item",
        )
        for raw in include
    ]
    include_set = set(include_list)

    exclude_list: list[str] = []
    exclude_set: set[str] = set()
    for raw in exclude:
        normalized = selector_resolver.normalize(
            raw,
            available=available,
            available_list=available_list,
            plan_name=plan_name,
            path="prompt.stages.exclude",
            context="sequence_item",
        )
        exclude_list.append(normalized)
        exclude_set.add(normalized)

    normalized_overrides: dict[str, StageOverrideLike] = {}
    override_sources: dict[str, str] = {}
    for raw_stage_id, override in overrides.items():
        stage_id = selector_resolver.normalize(
            raw_stage_id,
            available=available,
            available_list=available_list,
            plan_name=plan_name,
            path="prompt.stages.overrides",
            context="sequence_item",
        )
        if stage_id in normalized_overrides:
            raise ValueError(
                "Duplicate stage override after stage id normalization: "
                f"{override_sources.get(stage_id, stage_id)} and {raw_stage_id} both map to {stage_id}"
            )
        normalized_overrides[stage_id] = override
        override_sources[stage_id] = str(raw_stage_id)

    filtered_nodes: list[StageNode] = []
    filtered_stage_ids: list[str] = []
    for node in stage_nodes:
        stage_id = _stage_node_id(node, path="plan.stage_nodes[]")
        if include_set and stage_id not in include_set:
            continue
        if stage_id in exclude_set:
            continue
        filtered_nodes.append(node)
        filtered_stage_ids.append(stage_id)

    if not filtered_nodes:
        raise ValueError(
            f"Resolved stage list is empty after applying prompt.stages include/exclude (plan={plan_name})"
        )

    included_set = set(filtered_stage_ids)
    filtered_overrides: dict[str, StageOverrideLike] = {
        stage_id: override
        for stage_id, override in normalized_overrides.items()
        if stage_id in included_set
    }

    # Stage config defaults: validate kind ids via the registry, but allow unused kinds.
    normalized_defaults: dict[str, dict[str, Any]] = {}
    default_sources: dict[str, str] = {}
    for raw_kind_id, raw_cfg in stage_configs_defaults.items():
        try:
            kind = stage_registry.resolve(raw_kind_id).id
        except ValueError as exc:
            suggestions = stage_registry.suggest(raw_kind_id)
            hint = f" (did you mean: {', '.join(suggestions)})" if suggestions else ""
            raise ValueError(
                f"Unknown stage kind id in prompt.stage_configs.defaults: {raw_kind_id!r}{hint}"
            ) from exc
        if kind in normalized_defaults:
            raise ValueError(
                "Duplicate stage config default after normalization: "
                f"{default_sources.get(kind, kind)} and {raw_kind_id} both map to {kind}"
            )
        normalized_defaults[kind] = dict(raw_cfg)
        default_sources[kind] = str(raw_kind_id)

    # Stage config instances: must all apply to included stage ids (no silent leftovers).
    included_available = set(filtered_stage_ids)
    included_available_list = ", ".join(filtered_stage_ids) or "<none>"

    normalized_instances: dict[str, dict[str, Any]] = {}
    instance_sources: dict[str, str] = {}
    for raw_instance_id, raw_cfg in stage_configs_instances.items():
        instance_id = selector_resolver.normalize(
            raw_instance_id,
            available=included_available,
            available_list=included_available_list,
            plan_name=plan_name,
            path="prompt.stage_configs.instances",
            context="mapping_key",
        )
        if instance_id in normalized_instances:
            raise ValueError(
                "Duplicate stage instance config after stage id normalization: "
                f"{instance_sources.get(instance_id, instance_id)} and {raw_instance_id} both map to {instance_id}"
            )
        normalized_instances[instance_id] = dict(raw_cfg)
        instance_sources[instance_id] = str(raw_instance_id)

    provided = {str(item).strip() for item in initial_outputs if str(item).strip()}
    stage_io_summary: dict[str, dict[str, Any]] = {}
    stage_io_effective: dict[str, dict[str, Any]] = {}
    stage_instances_out: list[dict[str, str]] = []
    stage_configs_summary: dict[str, dict[str, Any]] = {}
    stage_configs_effective: dict[str, dict[str, Any]] = {}

    discovered_capture_key_owner: dict[str, str] = {}
    discovered_capture_key_collisions: dict[str, set[str]] = {}

    blocks: list[Block] = []
    for node in filtered_nodes:
        stage_id = _stage_node_id(node, path="plan.stage_nodes[]")
        ref = node.stage
        kind_id = ref.id

        stage_instances_out.append({"instance": stage_id, "kind": kind_id})
        stage_io_summary.setdefault(
            kind_id,
            {
                "requires": list(ref.io.requires),
                "provides": list(ref.io.provides),
                "captures": list(ref.io.captures),
            },
        )

        missing = [key for key in ref.io.requires if key not in provided]
        if missing:
            raise ValueError(
                "Stage IO validation failed: "
                f"stage={stage_id} kind={kind_id} missing_required_outputs={', '.join(missing)}"
            )

        defaults_cfg = normalized_defaults.get(kind_id, {})
        instance_cfg = normalized_instances.get(stage_id, {})
        merged_cfg = _deep_merge_stage_config(
            defaults_cfg, instance_cfg, path=f"prompt.stage_configs.{stage_id}"
        )
        if merged_cfg is None:
            merged_cfg = {}
        if not isinstance(merged_cfg, Mapping):
            raise ValueError(
                "Stage config merge produced non-mapping: "
                f"stage={stage_id} kind={kind_id} type={type(merged_cfg).__name__}"
            )

        cfg_ns = ConfigNamespace(
            dict(merged_cfg),
            path=f"prompt.stage_configs.resolved.{stage_id}",
        )

        if defaults_cfg or instance_cfg:
            stage_configs_summary[stage_id] = {
                "kind": kind_id,
                "defaults_keys": sorted(defaults_cfg.keys()),
                "instance_keys": sorted(instance_cfg.keys()),
            }

        try:
            block = ref.build(inputs, instance_id=stage_id, cfg=cfg_ns)
            cfg_ns.assert_consumed()
            effective = cfg_ns.effective_values()
        except Exception as exc:
            raise ValueError(
                f"Stage config/build failed: stage={stage_id} kind={kind_id}: {exc}"
            ) from exc

        if effective:
            stage_configs_effective[stage_id] = effective

        blocks.append(block)
        capture_keys: dict[str, str] = {}
        _collect_capture_keys(block, path_segments=["pipeline", stage_id], seen=capture_keys)
        discovered = sorted(capture_keys.keys())
        for key in discovered:
            owner = discovered_capture_key_owner.get(key)
            if owner is None:
                discovered_capture_key_owner[key] = stage_id
                continue
            if owner == stage_id:
                continue
            collided = discovered_capture_key_collisions.setdefault(key, {owner})
            collided.add(stage_id)

        stage_io_effective[stage_id] = {
            "requires": list(ref.io.requires),
            "provides_declared": list(ref.io.provides),
            "captures_declared": list(ref.io.captures),
            "captures_discovered": list(discovered),
        }
        provided.update(ref.io.provides)
        provided.update(ref.io.captures)
        provided.update(capture_keys.keys())

    if discovered_capture_key_collisions:
        collisions = []
        for key, stage_ids in sorted(discovered_capture_key_collisions.items()):
            collisions.append(f"{key} (stages: {', '.join(sorted(stage_ids))})")
        raise ValueError("Capture key collisions across stages: " + "; ".join(collisions))

    metadata: dict[str, Any] = {
        "plan": plan_name,
        "stages_include": list(include_list),
        "stages_exclude": list(exclude_list),
        "stage_instances": stage_instances_out,
        "stage_io": stage_io_summary,
    }
    if stage_io_effective:
        metadata["stage_io_effective"] = stage_io_effective
    if stage_configs_summary:
        metadata["stage_configs"] = stage_configs_summary
    if stage_configs_effective:
        metadata["stage_configs_effective"] = stage_configs_effective

    return CompiledStageBlocks(blocks=tuple(blocks), metadata=metadata, overrides=filtered_overrides)


def _stage_id(block: Block, *, path: str) -> str:
    if not isinstance(block, Block):
        raise TypeError(f"{path} must be a Block (type={type(block).__name__})")
    if not isinstance(block.name, str) or not block.name.strip():
        raise ValueError(f"{path} must have a non-empty Block.name (stage id)")
    return block.name.strip()


def _effective_child_name(node: Node, *, index: int) -> str:
    if isinstance(node, ChatStep):
        return node.name or f"step_{index + 1:02d}"
    if isinstance(node, ActionStep):
        return node.name or f"action_{index + 1:02d}"
    return node.name or f"block_{index + 1:02d}"


def _collect_capture_keys(
    node: Node,
    *,
    path_segments: list[str],
    seen: dict[str, str],
) -> None:
    """Collect capture keys under a node, raising on duplicates within the subtree."""

    if isinstance(node, ChatStep):
        if node.capture_key:
            key = node.capture_key
            path = "/".join(path_segments)
            existing = seen.get(key)
            if existing is not None:
                raise ValueError(f"Duplicate capture_key: {key} (at {existing} and {path})")
            seen[key] = path
        return

    if isinstance(node, ActionStep):
        if node.capture_key:
            key = node.capture_key
            path = "/".join(path_segments)
            existing = seen.get(key)
            if existing is not None:
                raise ValueError(f"Duplicate capture_key: {key} (at {existing} and {path})")
            seen[key] = path
        return

    if node.capture_key:
        key = node.capture_key
        path = "/".join(path_segments)
        existing = seen.get(key)
        if existing is not None:
            raise ValueError(f"Duplicate capture_key: {key} (at {existing} and {path})")
        seen[key] = path

    for idx, child in enumerate(node.nodes):
        child_name = _effective_child_name(child, index=idx)
        _collect_capture_keys(child, path_segments=[*path_segments, child_name], seen=seen)


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
    capture_key: str,
    selector_resolver: StageSelectorResolver,
    override_policy: StageOverridePolicy,
    capture_policy: CapturePolicy,
) -> ResolvedStages:
    """Resolve stage blocks by applying selectors, overrides, and capture policy."""

    stage_ids = [_stage_id(block, path="plan.stage_blocks[]") for block in stage_blocks]
    duplicates: set[str] = set()
    seen_stage_ids: set[str] = set()
    for stage_id in stage_ids:
        if stage_id in seen_stage_ids:
            duplicates.add(stage_id)
        seen_stage_ids.add(stage_id)
    if duplicates:
        raise ValueError(f"Duplicate stage id: {sorted(duplicates)[0]}")

    available = set(stage_ids)
    available_list = ", ".join(stage_ids) or "<none>"

    include_list = [
        selector_resolver.normalize(
            raw,
            available=available,
            available_list=available_list,
            plan_name=plan_name,
            path="prompt.stages.include",
            context="sequence_item",
        )
        for raw in include
    ]
    include_set = set(include_list)

    exclude_list: list[str] = []
    exclude_set: set[str] = set()
    for raw in exclude:
        normalized = selector_resolver.normalize(
            raw,
            available=available,
            available_list=available_list,
            plan_name=plan_name,
            path="prompt.stages.exclude",
            context="sequence_item",
        )
        exclude_list.append(normalized)
        exclude_set.add(normalized)

    normalized_overrides: dict[str, StageOverrideLike] = {}
    override_sources: dict[str, str] = {}
    for raw_stage_id, override in overrides.items():
        stage_id = selector_resolver.normalize(
            raw_stage_id,
            available=available,
            available_list=available_list,
            plan_name=plan_name,
            path="prompt.stages.overrides",
            context="sequence_item",
        )
        if stage_id in normalized_overrides:
            raise ValueError(
                "Duplicate stage override after stage id normalization: "
                f"{override_sources.get(stage_id, stage_id)} and {raw_stage_id} both map to {stage_id}"
            )
        normalized_overrides[stage_id] = override
        override_sources[stage_id] = str(raw_stage_id)

    filtered: list[Block] = []
    for block in stage_blocks:
        stage_id = _stage_id(block, path="plan.stage_blocks[]")
        if include_set and stage_id not in include_set:
            continue
        if stage_id in exclude_set:
            continue
        filtered.append(block)

    if not filtered:
        raise ValueError(
            f"Resolved stage list is empty after applying prompt.stages include/exclude (plan={plan_name})"
        )

    resolved: list[Block] = []
    stage_overrides_summary: dict[str, dict[str, Any]] = {}

    for block in filtered:
        stage_id = _stage_id(block, path="plan.stage_blocks[]")
        override = normalized_overrides.get(stage_id)
        if override is None:
            resolved.append(block)
            continue

        updated, summary = override_policy.apply(
            block,
            stage_id=stage_id,
            override=override,
            override_source=override_sources.get(stage_id, stage_id),
        )
        resolved.append(updated)
        if summary:
            stage_overrides_summary[stage_id] = summary

    resolved_stage_ids = [_stage_id(block, path="resolved_stages") for block in resolved]
    stage_by_id = {_stage_id(block, path="resolved_stages"): block for block in resolved}

    chosen_capture_stage: str
    if capture_stage is not None:
        if not isinstance(capture_stage, str) or not capture_stage.strip():
            raise ValueError("prompt.output.capture_stage must be a non-empty string or null")
        capture = selector_resolver.normalize(
            capture_stage,
            available=set(resolved_stage_ids),
            available_list=", ".join(resolved_stage_ids) or "<none>",
            plan_name=plan_name,
            path="prompt.output.capture_stage",
            context="sequence_item",
        )
        if capture not in set(resolved_stage_ids):
            raise ValueError(
                f"prompt.output.capture_stage={capture} is not in resolved stages: {', '.join(resolved_stage_ids)}"
            )
        chosen_capture_stage = capture
    else:
        chosen_capture_stage = capture_policy.choose_default_capture_stage(
            tuple(resolved), plan_name=plan_name
        )

    capture_block = stage_by_id.get(chosen_capture_stage)
    if capture_block is None:  # pragma: no cover - defensive
        raise AssertionError(f"capture_stage missing from resolved list: {chosen_capture_stage}")

    capture_policy.validate_capture_stage(capture_block, stage_id=chosen_capture_stage)

    updated_capture: list[Block] = []
    for block in resolved:
        stage_id = _stage_id(block, path="resolved_stages")
        if stage_id == chosen_capture_stage:
            updated_capture.append(
                capture_policy.attach_capture(block, stage_id=stage_id, capture_key=capture_key)
            )
        else:
            updated_capture.append(block)

    pipeline_preview = Block(name="pipeline", merge="all_messages", nodes=list(updated_capture))
    capture_keys: dict[str, str] = {}
    _collect_capture_keys(pipeline_preview, path_segments=["pipeline"], seen=capture_keys)

    metadata: dict[str, Any] = {
        "plan": plan_name,
        "stages_include": list(include_list),
        "stages_exclude": list(exclude_list),
        "capture_stage": chosen_capture_stage,
        "resolved_stages": [_stage_id(block, path="resolved_stages") for block in updated_capture],
        "stage_overrides": stage_overrides_summary,
    }
    stage_instances: dict[str, str] = {}
    for block in updated_capture:
        stage_id = _stage_id(block, path="resolved_stages")
        stage_kind = block.meta.get("stage_kind")
        if isinstance(stage_kind, str) and stage_kind.strip():
            kind = stage_kind.strip()
            if kind != stage_id:
                stage_instances[stage_id] = kind
    if stage_instances:
        metadata["stage_instances"] = stage_instances

    return ResolvedStages(
        stages=tuple(updated_capture),
        capture_stage=chosen_capture_stage,
        metadata=metadata,
    )

