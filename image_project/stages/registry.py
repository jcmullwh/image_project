from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from pipelinekit.stage_registry import StageRegistry
from pipelinekit.stage_types import StageRef


@lru_cache(maxsize=1)
def get_stage_registry() -> StageRegistry:
    # Import side-effect: stage modules define `STAGE` symbols collected here.
    # This function is the single import point for tools/docs generation.
    from image_project.stages import (  # noqa: PLC0415
        ab,
        blackbox,
        blackbox_refine,
        direct,
        postprompt,
        preprompt,
        refine,
        standard,
    )

    refs: list[StageRef] = []
    for pkg in (ab, blackbox, blackbox_refine, direct, postprompt, preprompt, refine, standard):
        exported = getattr(pkg, "__all_stages__", None)
        if isinstance(exported, (list, tuple)):
            refs.extend(exported)

    return StageRegistry.from_refs(refs)
