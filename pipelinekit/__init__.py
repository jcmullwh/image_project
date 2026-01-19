"""Reusable pipeline kernel (engine primitives + stage authoring kit).

This package is intentionally independent of `image_project.*`. Any project-specific
conventions (selector syntax, override targeting rules, capture semantics, artifact
schemas) must live in the consuming application.
"""

from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.messages import MessageHandler
from pipelinekit.engine.pipeline import (
    ALLOWED_MERGE_MODES,
    ActionStep,
    Block,
    ChatRunner,
    ChatStep,
    DefaultStepRecorder,
    FlowContext,
    MergeMode,
    Node,
    NullStepRecorder,
    StepRecorder,
    utc_now_iso8601,
)
from pipelinekit.stage_registry import StageRegistry
from pipelinekit.stage_types import StageBuilder, StageIO, StageInstance, StageKind, StageRef

__all__ = [
    "ALLOWED_MERGE_MODES",
    "ActionStep",
    "Block",
    "ChatRunner",
    "ChatStep",
    "ConfigNamespace",
    "DefaultStepRecorder",
    "FlowContext",
    "MergeMode",
    "MessageHandler",
    "Node",
    "NullStepRecorder",
    "StageBuilder",
    "StageIO",
    "StageInstance",
    "StageKind",
    "StageRef",
    "StageRegistry",
    "StepRecorder",
    "utc_now_iso8601",
]

