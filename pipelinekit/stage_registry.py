from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from pipelinekit.stage_types import StageRef



@dataclass(frozen=True)
class StageRegistry:
    _by_id: dict[str, StageRef]

    @classmethod
    def from_refs(cls, refs: Iterable[StageRef]) -> "StageRegistry":
        entries: dict[str, StageRef] = {}
        for ref in refs:
            if ref.id in entries:
                raise ValueError(f"Duplicate stage kind id: {ref.id}")
            entries[ref.id] = ref
        return cls(_by_id=entries)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id.keys()))

    def describe(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for ref in sorted(self._by_id.values(), key=lambda r: r.id):
            rows.append(
                {
                    "stage_id": ref.id,
                    "doc": ref.doc,
                    "source": ref.source,
                    "tags": list(ref.tags),
                    "kind": ref.kind,
                    "io": {
                        "requires": list(ref.io.requires),
                        "provides": list(ref.io.provides),
                        "captures": list(ref.io.captures),
                    },
                }
            )
        return tuple(rows)

    def get(self, stage_id: str) -> StageRef:
        ref = self._by_id.get((stage_id or "").strip())
        if ref is None:
            raise ValueError(f"Unknown stage kind id: {stage_id}")
        return ref

    def resolve(self, stage_id: str) -> StageRef:
        if not isinstance(stage_id, str) or not stage_id.strip():
            raise ValueError("stage_id must be a non-empty string")
        key = stage_id.strip()

        direct = self._by_id.get(key)
        if direct is not None:
            return direct

        if "." in key:
            available = ", ".join(self.available()) or "<none>"
            raise ValueError(f"Unknown stage kind id: {stage_id} (available: {available})")

        matches = sorted(s for s in self._by_id.keys() if s.endswith("." + key))
        if len(matches) == 1:
            return self._by_id[matches[0]]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous stage kind id: {stage_id} (matches: {', '.join(matches)})"
            )

        available = ", ".join(self.available()) or "<none>"
        raise ValueError(f"Unknown stage kind id: {stage_id} (available: {available})")

    def suggest(self, stage_id: str, *, limit: int = 3) -> tuple[str, ...]:
        key = (stage_id or "").strip()
        if not key:
            return ()

        available = self.available()
        if not available:
            return ()

        suffix_to_full: dict[str, list[str]] = defaultdict(list)
        for full in available:
            suffix_to_full[full.split(".", 1)[-1]].append(full)

        suggestions = list(difflib.get_close_matches(key, list(suffix_to_full.keys()), n=limit))
        expanded: list[str] = []
        for suggestion in suggestions:
            expanded.extend(suffix_to_full.get(suggestion, []))
        if expanded:
            return tuple(expanded[:limit])
        return tuple(difflib.get_close_matches(key, list(available), n=limit))
