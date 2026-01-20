"""Backward-compatible wrapper for experiment pairing manifests.

New code should import from `image_project.framework.experiment_manifest`.
"""

from __future__ import annotations

from image_project.framework.experiment_manifest import (  # noqa: F401
    PlannedRunLike,
    build_pairs_payload,
    record_pair_error,
    write_pairs_manifest,
)
