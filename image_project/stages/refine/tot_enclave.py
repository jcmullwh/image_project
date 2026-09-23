from __future__ import annotations

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.pipeline import Block
from image_project.framework.prompt_pipeline import PlanInputs
from image_project.stages.refine.tot_enclave_prompts import ENCLAVE_ARTISTS, make_tot_enclave_block
from pipelinekit.stage_types import StageIO, StageRef

KIND_ID = "refine.tot_enclave"


def _build(_inputs: PlanInputs, *, instance_id: str, cfg: ConfigNamespace):
    default_critics = [key for key, _label, _persona in ENCLAVE_ARTISTS]
    critics = cfg.get_list_str("critics", default=default_critics)
    max_critics_raw = cfg.get_int("max_critics", default=len(critics), min_value=1)
    reduce_style = cfg.get_str(
        "reduce_style",
        default="best_of",
        choices=("best_of", "consensus"),
    )
    capture_prefix = cfg.get_str("capture_prefix", default=instance_id)
    if reduce_style is None:
        raise ValueError(f"{instance_id}.reduce_style cannot be null")
    if capture_prefix is None:
        raise ValueError(f"{instance_id}.capture_prefix cannot be null")

    allowed = {key for key, _label, _persona in ENCLAVE_ARTISTS}
    seen: set[str] = set()
    dupes: list[str] = []
    for critic_id in critics:
        if critic_id in seen and critic_id not in dupes:
            dupes.append(critic_id)
        seen.add(critic_id)
    if dupes:
        raise ValueError(f"Duplicate critic id(s) for {instance_id}: {', '.join(dupes)}")

    invalid = [item for item in critics if item not in allowed]
    if invalid:
        allowed_list = ", ".join(sorted(allowed)) or "<none>"
        raise ValueError(
            f"Invalid critic id(s) for {instance_id}: {', '.join(invalid)} (allowed: {allowed_list})"
        )

    max_critics = min(int(max_critics_raw), len(critics)) if critics else 0
    if max_critics <= 0:
        raise ValueError(f"{instance_id} critics list is empty after max_critics clamp")

    critics_effective = list(critics[:max_critics])
    cfg.set_effective("critics", critics_effective)
    cfg.set_effective("max_critics", int(max_critics))

    cfg.assert_consumed()
    return Block(
        name=instance_id,
        merge="last_response",
        nodes=[
            make_tot_enclave_block(
                instance_id,
                critics=critics_effective,
                reduce_style=reduce_style,
                capture_prefix=capture_prefix,
            )
        ],
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build,
    doc="Refine the latest assistant output via a ToT enclave critique+consensus pass.",
    source="stages.refine.tot_enclave_prompts.make_tot_enclave_block",
    tags=("refine",),
    kind="composite",
    io=StageIO(),
)
