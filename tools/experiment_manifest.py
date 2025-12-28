from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class PlannedRunLike(Protocol):
    variant_id: str
    run_index: int
    generation_id: str


def build_pairs_payload(
    plan: Sequence[PlannedRunLike],
    *,
    experiment_id: str,
    run_mode: str,
) -> dict[str, Any]:
    from image_project.framework.artifacts import utc_now_iso8601  # noqa: PLC0415

    by_index: dict[int, dict[str, str]] = {}
    for entry in plan:
        idx = int(entry.run_index)
        by_index.setdefault(idx, {})[str(entry.variant_id)] = str(entry.generation_id)

    pairs: list[dict[str, Any]] = []
    for run_index in sorted(by_index.keys()):
        variants = by_index[run_index]
        if "A" not in variants or "B" not in variants:
            raise ValueError(
                "A/B experiment plan must include both variants for every run_index "
                f"(run_index={run_index}, present={sorted(variants.keys())})"
            )
        pairs.append(
            {
                "run_index": run_index,
                "a_generation_id": variants["A"],
                "b_generation_id": variants["B"],
                "metadata": {"mode": run_mode},
            }
        )

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at": utc_now_iso8601(),
        "pairs": pairs,
    }


def _find_pair(payload: Mapping[str, Any], *, run_index: int) -> dict[str, Any]:
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise TypeError("pairs payload missing 'pairs' list")
    for pair in pairs:
        if isinstance(pair, dict) and pair.get("run_index") == run_index:
            return pair
    raise KeyError(f"pairs payload missing run_index={run_index}")


def record_pair_error(
    payload: Mapping[str, Any],
    *,
    run_index: int,
    variant_id: str,
    error: Mapping[str, Any],
) -> None:
    pair = _find_pair(payload, run_index=run_index)
    vid = (variant_id or "").strip().upper()
    if vid == "A":
        pair["a_error"] = dict(error)
        return
    if vid == "B":
        pair["b_error"] = dict(error)
        return
    raise ValueError(f"Unknown variant_id for A/B pairs manifest: {variant_id!r}")


def write_pairs_manifest(experiment_dir: str, payload: Mapping[str, Any]) -> str:
    if not isinstance(experiment_dir, str) or not experiment_dir.strip():
        raise ValueError("experiment_dir must be a non-empty string")
    output_path = os.path.join(os.path.abspath(experiment_dir), "pairs.json")

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    data = dict(payload)
    pairs = data.get("pairs")
    if isinstance(pairs, list):
        data["pairs"] = sorted(
            (pair for pair in pairs if isinstance(pair, dict)),
            key=lambda pair: int(pair.get("run_index", 0)),
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return output_path

