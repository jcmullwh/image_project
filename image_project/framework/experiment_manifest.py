from __future__ import annotations

"""Experiment helper artifacts (pairing manifests, etc.).

This module lives in `framework/` so it can be shared by:

- the canonical experiment runner (`image_project.app.experiment_runner`)
- any maintenance tooling under `tools/`

It must remain independent of `image_project.stages` and `image_project.impl`.
"""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from image_project.framework.artifacts import utc_now_iso8601


class PlannedRunLike(Protocol):
    """Minimal run interface required for pairing manifests."""

    variant: str
    run: int
    generation_id: str


def build_pairs_payload(
    plan: Sequence[PlannedRunLike],
    *,
    experiment_id: str,
    run_mode: str,
    variant_a: str = "A",
    variant_b: str = "B",
) -> dict[str, Any]:
    """Build a simple A/B pairing manifest payload from a plan.

    Args:
        plan: Planned runs. Each run must provide `variant`, `run`, and `generation_id`.
        experiment_id: Experiment id recorded in artifacts/transcripts.
        run_mode: Run mode string ("full" or "prompt_only").
        variant_a: Variant key representing "A" (default "A").
        variant_b: Variant key representing "B" (default "B").

    Returns:
        JSON-serializable payload suitable for writing to `pairs.json`.

    Raises:
        ValueError: If any run index is missing either the A or B variant.
    """

    a_key = str(variant_a).strip()
    b_key = str(variant_b).strip()
    if not a_key or not b_key:
        raise ValueError("variant_a and variant_b must be non-empty strings")
    if a_key == b_key:
        raise ValueError("variant_a and variant_b must be distinct")

    by_index: dict[int, dict[str, str]] = {}
    for entry in plan:
        idx = int(entry.run)
        by_index.setdefault(idx, {})[str(entry.variant)] = str(entry.generation_id)

    pairs: list[dict[str, Any]] = []
    for run_index in sorted(by_index.keys()):
        variants = by_index[run_index]
        if a_key not in variants or b_key not in variants:
            raise ValueError(
                "A/B experiment plan must include both variants for every run index "
                f"(run={run_index}, present={sorted(variants.keys())})"
            )
        pairs.append(
            {
                "run_index": run_index,
                "a_generation_id": variants[a_key],
                "b_generation_id": variants[b_key],
                "metadata": {"mode": str(run_mode)},
            }
        )

    return {
        "schema_version": 1,
        "experiment_id": str(experiment_id),
        "created_at": utc_now_iso8601(),
        "pairs": pairs,
    }


def _find_pair(payload: Mapping[str, Any], *, run_index: int) -> dict[str, Any]:
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise TypeError("pairs payload missing 'pairs' list")
    for pair in pairs:
        if isinstance(pair, dict) and int(pair.get("run_index", 0)) == int(run_index):
            return pair
    raise KeyError(f"pairs payload missing run_index={run_index}")


def record_pair_error(
    payload: Mapping[str, Any],
    *,
    run_index: int,
    variant: str,
    error: Mapping[str, Any],
    variant_a: str = "A",
    variant_b: str = "B",
) -> None:
    """Record a per-run per-variant error inside a pairing payload."""

    pair = _find_pair(payload, run_index=run_index)
    vid = (variant or "").strip()
    a_key = str(variant_a).strip()
    b_key = str(variant_b).strip()

    if vid == a_key:
        pair["a_error"] = dict(error)
        return
    if vid == b_key:
        pair["b_error"] = dict(error)
        return
    raise ValueError(f"Unknown variant for pairs manifest: {variant!r} (expected {a_key!r} or {b_key!r})")


def write_pairs_manifest(experiment_dir: str, payload: Mapping[str, Any]) -> str:
    """Write a `pairs.json` file under an experiment directory."""

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
