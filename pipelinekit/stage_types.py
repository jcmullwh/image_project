from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.pipeline import Block

StageKind = Literal["chat", "action", "composite"]


@dataclass(frozen=True)
class StageIO:
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    captures: tuple[str, ...] = ()


class StageBuilder(Protocol):
    def __call__(self, inputs: Any, *, instance_id: str, cfg: ConfigNamespace) -> Block:
        ...


@dataclass(frozen=True)
class StageRef:
    id: str
    builder: StageBuilder
    doc: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = ()
    kind: StageKind | None = None
    io: StageIO = field(default_factory=StageIO)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise TypeError("StageRef.id must be a non-empty string")
        object.__setattr__(self, "id", self.id.strip())

        if self.doc is not None and (not isinstance(self.doc, str) or not self.doc.strip()):
            raise TypeError("StageRef.doc must be a non-empty string or None")
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source.strip()
        ):
            raise TypeError("StageRef.source must be a non-empty string or None")

        if self.tags:
            object.__setattr__(
                self, "tags", tuple(str(tag).strip() for tag in self.tags if str(tag).strip())
            )

        if self.kind is not None:
            normalized = str(self.kind).strip().lower()
            if normalized not in ("chat", "action", "composite"):
                raise ValueError(
                    f"StageRef.kind must be one of: chat, action, composite (got {self.kind!r})"
                )
            object.__setattr__(self, "kind", normalized)  # type: ignore[arg-type]

    def instance(self, instance_id: str | None = None) -> "StageInstance":
        return StageInstance(stage=self, instance_id=instance_id or self.id)

    def build(self, inputs: Any, *, instance_id: str, cfg: ConfigNamespace) -> Block:
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id must be a non-empty string")
        normalized_instance_id = instance_id.strip()

        block = self.builder(inputs, instance_id=normalized_instance_id, cfg=cfg)
        if not isinstance(block, Block):
            raise TypeError(
                f"Stage builder returned non-Block (stage={self.id}, type={type(block).__name__})"
            )
        if block.name != normalized_instance_id:
            raise ValueError(
                "Stage builder returned mismatched Block.name: "
                f"expected={normalized_instance_id} got={block.name}"
            )

        next_meta = dict(block.meta)

        existing_kind = next_meta.get("stage_kind")
        if existing_kind is None:
            next_meta["stage_kind"] = self.id
        elif not isinstance(existing_kind, str) or existing_kind.strip() != self.id:
            raise ValueError(
                "Stage builder returned conflicting meta.stage_kind: "
                f"expected={self.id} got={existing_kind!r}"
            )

        existing_instance = next_meta.get("stage_instance")
        if existing_instance is None:
            next_meta["stage_instance"] = normalized_instance_id
        elif not isinstance(existing_instance, str) or existing_instance.strip() != normalized_instance_id:
            raise ValueError(
                "Stage builder returned conflicting meta.stage_instance: "
                f"expected={normalized_instance_id} got={existing_instance!r}"
            )

        if "doc" not in next_meta and self.doc:
            next_meta["doc"] = self.doc
        if "source" not in next_meta and self.source:
            next_meta["source"] = self.source
        if "tags" not in next_meta and self.tags:
            next_meta["tags"] = list(self.tags)

        if next_meta == block.meta:
            return block

        return Block(
            name=block.name,
            merge=block.merge,
            nodes=list(block.nodes),
            capture_key=block.capture_key,
            meta=next_meta,
        )


@dataclass(frozen=True)
class StageInstance:
    stage: StageRef
    instance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise TypeError("StageInstance.instance_id must be a non-empty string")
        object.__setattr__(self, "instance_id", self.instance_id.strip())
