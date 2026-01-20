from __future__ import annotations

"""Image-project conventions for stage compilation.

`pipelinekit` provides the generic compiler and stage system. This module supplies
the image-project-specific conventions as explicit policy objects:

- Suffix selector resolution for stage instance ids.
- Override targeting ("primary draft step") and patch semantics.
- Default capture stage selection and capture eligibility validation.

These policies must remain free of `image_project.stages` and `image_project.impl`
imports (framework boundary).
"""

import logging
from typing import Any

from pipelinekit.compiler import (
    CapturePolicy,
    SelectorContext,
    StageOverrideLike,
    StageOverridePolicy,
    StageSelectorResolver,
)
from pipelinekit.engine.pipeline import ActionStep, Block, ChatStep

Node = ChatStep | ActionStep | Block

logger = logging.getLogger(__name__)


class SuffixStageSelectorResolver:
    """Resolve stage selectors using image-project suffix rules.

    Rules (kept compatible with the pre-kernel compiler):

    - Exact match always wins.
    - If the selector contains a dot and is not an exact match, it is treated as a
      fully-qualified id and is rejected (no suffix matching).
    - If the selector contains no dot, treat it as a suffix and match any available
      id that ends with `.<suffix>`.
    - Ambiguous suffix matches raise.
    """

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
        if context == "mapping_key":
            if not isinstance(raw, str):
                raise TypeError(f"{path} keys must be strings")
            stage_id = raw.strip()
            if not stage_id:
                raise ValueError(f"{path} keys must be non-empty strings")
        else:
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
            resolved = matches[0]
            logger.info("Resolved stage selector %s=%r to %r (plan=%s)", path, stage_id, resolved, plan_name)
            return resolved
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous stage id in {path}: {stage_id} (matches: {', '.join(matches)})"
            )
        raise ValueError(
            f"Unknown stage id in {path}: {stage_id} (plan={plan_name}, available={available_list})"
        )


def _block_contains_chat_step(node: Node) -> bool:
    """Return True if any ChatStep exists in the node subtree."""

    if isinstance(node, ChatStep):
        return True
    if isinstance(node, Block):
        return any(_block_contains_chat_step(child) for child in node.nodes)
    return False


def _node_produces_assistant_output(node: Node) -> bool:
    """Return True if the node type could produce assistant output (ignoring merge)."""

    if isinstance(node, ChatStep):
        return True
    if isinstance(node, ActionStep):
        return False
    return _block_produces_assistant_output(node)


def _block_produces_assistant_output(block: Block) -> bool:
    """Return True if a block contains any chat-producing child with merge != 'none'."""

    for child in block.nodes:
        if child.merge == "none":
            continue
        if _node_produces_assistant_output(child):
            return True
    return False


class ImageProjectCapturePolicy:
    """Capture semantics for image_project prompt pipelines."""

    def choose_default_capture_stage(self, stage_blocks: tuple[Block, ...], *, plan_name: str) -> str:
        candidates = [block for block in stage_blocks if _block_produces_assistant_output(block)]
        if not candidates:
            raise ValueError(
                f"Resolved stage list contains no chat-producing stages; cannot infer capture stage (plan={plan_name})"
            )
        chosen = candidates[-1]
        if not isinstance(chosen.name, str) or not chosen.name.strip():
            raise ValueError("Default capture stage selection requires named stage blocks")
        stage_id = chosen.name.strip()
        logger.info("Selected default capture stage %s (plan=%s)", stage_id, plan_name)
        return stage_id

    def validate_capture_stage(self, stage_block: Block, *, stage_id: str) -> None:
        if not _block_contains_chat_step(stage_block):
            raise ValueError(
                f"capture_stage={stage_id!r} must be chat-producing stage; no ChatStep found"
            )
        if not _block_produces_assistant_output(stage_block):
            raise ValueError(
                f"capture_stage={stage_id!r} must be chat-producing stage; stage produces no assistant output"
            )

    def attach_capture(self, stage_block: Block, *, stage_id: str, capture_key: str) -> Block:
        if stage_block.capture_key is not None and stage_block.capture_key != capture_key:
            raise ValueError(
                f"capture_stage={stage_id!r} already has capture_key={stage_block.capture_key!r}; "
                f"cannot set {capture_key!r}"
            )
        return Block(
            name=stage_block.name,
            merge=stage_block.merge,
            nodes=list(stage_block.nodes),
            capture_key=capture_key,
            meta=dict(stage_block.meta),
        )


def _find_primary_draft_steps(node: Node) -> list[ChatStep]:
    """Locate the single, image-project-convention "primary draft step" in a stage block."""

    matches: list[ChatStep] = []
    if isinstance(node, ChatStep):
        if (
            node.name == "draft"
            and isinstance(node.meta, dict)
            and node.meta.get("role") == "primary"
        ):
            matches.append(node)
        return matches

    if isinstance(node, Block):
        for child in node.nodes:
            matches.extend(_find_primary_draft_steps(child))
    return matches


class PrimaryDraftOverridePolicy:
    """Override policy: apply overrides to the stage's single primary draft ChatStep."""

    def apply(
        self,
        stage_block: Block,
        *,
        stage_id: str,
        override: StageOverrideLike,
        override_source: str,
    ) -> tuple[Block, dict[str, Any] | None]:
        matches = _find_primary_draft_steps(stage_block)
        if not matches:
            raise ValueError(
                f"prompt.stages.overrides[{override_source!r}] targets non-overridable stage (no primary draft step)"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Stage override for {stage_id} is ambiguous; multiple primary draft steps found"
            )

        target = matches[0]

        override_summary: dict[str, Any] = {}
        next_temperature = target.temperature
        next_params = dict(target.params)

        if override.temperature is not None:
            next_temperature = float(override.temperature)
            override_summary["temperature"] = next_temperature
        if override.params is not None:
            if "temperature" in override.params:
                raise ValueError(
                    f"prompt.stages.overrides.{stage_id}.params sets temperature; use .temperature instead"
                )
            next_params.update(dict(override.params))
            override_summary["params"] = dict(override.params)

        if not override_summary:
            return stage_block, None

        def _apply(node: Node) -> tuple[Node, bool]:
            if isinstance(node, ChatStep):
                if node is target:
                    return (
                        ChatStep(
                            name=node.name,
                            prompt=node.prompt,
                            temperature=next_temperature,
                            merge=node.merge,
                            allow_empty_prompt=node.allow_empty_prompt,
                            allow_empty_response=node.allow_empty_response,
                            capture_key=node.capture_key,
                            params=next_params,
                            meta=dict(node.meta),
                        ),
                        True,
                    )
                return node, False

            if isinstance(node, Block):
                changed = False
                next_nodes: list[Node] = []
                for child in node.nodes:
                    updated, child_changed = _apply(child)
                    changed = changed or child_changed
                    next_nodes.append(updated)
                if not changed:
                    return node, False
                return (
                    Block(
                        name=node.name,
                        merge=node.merge,
                        nodes=next_nodes,
                        capture_key=node.capture_key,
                        meta=dict(node.meta),
                    ),
                    True,
                )

            return node, False

        updated, changed = _apply(stage_block)
        if not changed or not isinstance(updated, Block):
            raise AssertionError("Override application did not update the stage block as expected")

        logger.info(
            "Applied stage override (stage=%s source=%s changes=%s)",
            stage_id,
            override_source,
            sorted(override_summary.keys()),
        )
        return updated, override_summary


IMAGE_PROJECT_SELECTOR_RESOLVER: StageSelectorResolver = SuffixStageSelectorResolver()
IMAGE_PROJECT_OVERRIDE_POLICY: StageOverridePolicy = PrimaryDraftOverridePolicy()
IMAGE_PROJECT_CAPTURE_POLICY: CapturePolicy = ImageProjectCapturePolicy()
