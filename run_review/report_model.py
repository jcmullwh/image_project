from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RunInputs:
    generation_id: str
    oplog_path: Optional[str] = None
    transcript_path: Optional[str] = None


@dataclass
class OplogEvent:
    timestamp: datetime
    level: str
    type: str
    message: str
    path: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class TranscriptStep:
    name: str
    path: str
    prompt: Optional[str]
    response: Optional[str]
    prompt_chars: Optional[int] = None
    input_chars: Optional[int] = None
    context_chars: Optional[int] = None
    response_chars: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepTiming:
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    duration_ms: Optional[float] = None


@dataclass
class StepReport:
    path: str
    name: str
    step_index: Optional[int] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    prompt_chars: Optional[int] = None
    input_chars: Optional[int] = None
    context_chars: Optional[int] = None
    response_chars: Optional[int] = None
    oplog_prompt_chars: Optional[int] = None
    oplog_input_chars: Optional[int] = None
    oplog_context_chars: Optional[int] = None
    oplog_response_chars: Optional[int] = None
    timing: StepTiming = field(default_factory=StepTiming)
    issues: List["Issue"] = field(default_factory=list)


@dataclass
class RunMetadata:
    generation_id: str
    seed: Optional[int] = None
    created_at: Optional[str] = None
    selected_concepts: Optional[List[str]] = None
    image_path: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    title_generation: Optional[Any] = None
    concept_filter_log: Optional[Any] = None
    oplog_stats: Optional[Dict[str, Any]] = None
    artifact_paths: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    path: Optional[str] = None
    artifact_path: Optional[str] = None


@dataclass
class SideEffect:
    type: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class RunReport:
    metadata: RunMetadata
    steps: List[StepReport]
    side_effects: List[SideEffect]
    issues: List[Issue]
    parser_version: str
    tool_version: str
    run_start_ts: Optional[datetime] = None
    run_end_ts: Optional[datetime] = None
    runtime_ms: Optional[float] = None
    unknown_events: List[str] = field(default_factory=list)


@dataclass
class CompareResult:
    run_a: RunReport
    run_b: RunReport
    added_steps: List[str]
    removed_steps: List[str]
    metadata_changes: Dict[str, Dict[str, Any]]
    injector_diffs: List[str]
    post_processing_diffs: List[str] = field(default_factory=list)

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RunInputs:
    generation_id: str
    oplog_path: Optional[str] = None
    transcript_path: Optional[str] = None


@dataclass
class OplogEvent:
    timestamp: datetime
    level: str
    type: str
    message: str
    path: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class TranscriptStep:
    name: str
    path: str
    prompt: Optional[str]
    response: Optional[str]
    prompt_chars: Optional[int] = None
    input_chars: Optional[int] = None
    context_chars: Optional[int] = None
    response_chars: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepTiming:
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    duration_ms: Optional[float] = None


@dataclass
class StepReport:
    path: str
    name: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    prompt_chars: Optional[int] = None
    input_chars: Optional[int] = None
    context_chars: Optional[int] = None
    response_chars: Optional[int] = None
    oplog_prompt_chars: Optional[int] = None
    oplog_input_chars: Optional[int] = None
    oplog_context_chars: Optional[int] = None
    oplog_response_chars: Optional[int] = None
    timing: StepTiming = field(default_factory=StepTiming)
    issues: List["Issue"] = field(default_factory=list)


@dataclass
class RunMetadata:
    generation_id: str
    seed: Optional[int] = None
    created_at: Optional[str] = None
    selected_concepts: Optional[List[str]] = None
    image_path: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    title_generation: Optional[Any] = None
    concept_filter_log: Optional[Any] = None
    artifact_paths: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    path: Optional[str] = None
    artifact_path: Optional[str] = None


@dataclass
class SideEffect:
    type: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class RunReport:
    metadata: RunMetadata
    steps: List[StepReport]
    side_effects: List[SideEffect]
    issues: List[Issue]
    parser_version: str
    tool_version: str
    run_start_ts: Optional[datetime] = None
    run_end_ts: Optional[datetime] = None
    runtime_ms: Optional[float] = None
    unknown_events: List[str] = field(default_factory=list)


@dataclass
class CompareResult:
    run_a: RunReport
    run_b: RunReport
    added_steps: List[str]
    removed_steps: List[str]
    metadata_changes: Dict[str, Dict[str, Any]]
    injector_diffs: List[str]
    post_processing_diffs: List[str] = field(default_factory=list)

