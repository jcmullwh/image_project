from __future__ import annotations

"""Reusable Block composition helpers.

These helpers are intentionally generic (no `image_project.*` dependencies) and work
for any Block/Step pipeline built on `pipelinekit.engine.pipeline`.
"""

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pipelinekit.engine.pipeline import ActionStep, Block, ChatStep, MergeMode

Node = ChatStep | ActionStep | Block
T = TypeVar("T")


def fanout_then_reduce(
    name: str,
    *,
    fanout: Sequence[Node],
    reduce: Sequence[Node],
    fanout_name: str = "fanout",
    reduce_name: str = "reduce",
    merge: MergeMode = "all_messages",
    meta: dict[str, Any] | None = None,
) -> Block:
    """Pattern: run N independent nodes, then one or more reducer nodes."""

    if not isinstance(name, str) or not name.strip():
        raise TypeError("name must be a non-empty string")
    if not isinstance(fanout_name, str) or not fanout_name.strip():
        raise TypeError("fanout_name must be a non-empty string")
    if not isinstance(reduce_name, str) or not reduce_name.strip():
        raise TypeError("reduce_name must be a non-empty string")

    return Block(
        name=name.strip(),
        merge=merge,
        nodes=[
            Block(name=fanout_name.strip(), merge="none", nodes=list(fanout)),
            Block(name=reduce_name.strip(), merge="all_messages", nodes=list(reduce)),
        ],
        meta=dict(meta) if meta else {},
    )


def generate_then_select(
    name: str,
    *,
    generate: Sequence[Node],
    select: Node,
    generate_name: str = "generate",
    select_name: str = "select",
    merge: MergeMode = "all_messages",
    meta: dict[str, Any] | None = None,
) -> Block:
    """Pattern: generate candidates (fanout-ish), then select a winner (reducer)."""

    if not isinstance(name, str) or not name.strip():
        raise TypeError("name must be a non-empty string")
    if not isinstance(generate_name, str) or not generate_name.strip():
        raise TypeError("generate_name must be a non-empty string")
    if not isinstance(select_name, str) or not select_name.strip():
        raise TypeError("select_name must be a non-empty string")

    select_block = (
        select
        if isinstance(select, Block)
        else Block(name=select_name.strip(), merge="all_messages", nodes=[select])
    )

    return Block(
        name=name.strip(),
        merge=merge,
        nodes=[
            Block(name=generate_name.strip(), merge="none", nodes=list(generate)),
            select_block,
        ],
        meta=dict(meta) if meta else {},
    )


def iterate(
    name: str,
    *,
    items: Sequence[T],
    build: Callable[[T, int], Node],
    iteration_name: Callable[[T, int], str] | None = None,
    merge: MergeMode = "none",
    meta: dict[str, Any] | None = None,
) -> Block:
    """Pattern: deterministically repeat a node builder across a sequence."""

    if not isinstance(name, str) or not name.strip():
        raise TypeError("name must be a non-empty string")

    nodes: list[Node] = []
    for idx, item in enumerate(items, start=1):
        child = build(item, idx)
        if iteration_name is None:
            child_name = f"iter_{idx:02d}"
        else:
            child_name = iteration_name(item, idx)
            if not isinstance(child_name, str) or not child_name.strip():
                raise ValueError("iteration_name must return a non-empty string")
            child_name = child_name.strip()

        if isinstance(child, Block):
            nodes.append(
                Block(
                    name=child_name,
                    merge="none",
                    nodes=list(child.nodes),
                    capture_key=child.capture_key,
                    meta=dict(child.meta),
                )
            )
        else:
            nodes.append(Block(name=child_name, merge="none", nodes=[child]))

    return Block(name=name.strip(), merge=merge, nodes=nodes, meta=dict(meta) if meta else {})

