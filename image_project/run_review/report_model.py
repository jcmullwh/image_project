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
    transcript_index: Optional[int] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
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
    transcript_index: Optional[int] = None
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
    experiment: Optional[Dict[str, Any]] = None
    prompt_pipeline: Optional[Dict[str, Any]] = None
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
    evolution: Optional["EvolutionReport"] = None
    run_start_ts: Optional[datetime] = None
    run_end_ts: Optional[datetime] = None
    runtime_ms: Optional[float] = None
    unknown_events: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvolutionThresholds:
    candidate_redundancy_seq_ratio: float = 0.975
    candidate_redundancy_jaccard: float = 0.92
    merge_near_single_candidate_seq_ratio: float = 0.96
    merge_near_single_candidate_margin: float = 0.05
    merge_near_single_candidate_other_max: float = 0.92
    low_delta_merge_seq_ratio: float = 0.985
    low_delta_merge_token_change_ratio: float = 0.05

    suggestion_adopted_seq_ratio: float = 0.82
    suggestion_adopted_jaccard: float = 0.35

    provenance_origin_min_score: float = 0.62

    max_segment_preview_chars: int = 240
    max_term_index: int = 300
    max_node_introduced_terms: int = 24


@dataclass(frozen=True)
class EvolutionConfig:
    thresholds: EvolutionThresholds = field(default_factory=EvolutionThresholds)


@dataclass(frozen=True)
class DeltaMetrics:
    from_path: str
    to_path: str
    from_chars: int
    to_chars: int
    from_tokens: int
    to_tokens: int
    from_lines: int
    to_lines: int
    from_sentences: int
    to_sentences: int
    tokens_added: int
    tokens_removed: int
    token_change_ratio: float
    seq_ratio: float
    jaccard: float


@dataclass(frozen=True)
class PairSimilarity:
    a_path: str
    b_path: str
    seq_ratio: float
    jaccard: float


@dataclass(frozen=True)
class EvolutionFinding:
    type: str
    severity: str
    message: str
    support: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuggestionMatch:
    target_path: str
    target_kind: str
    segment_preview: str
    seq_ratio: float
    jaccard: float


@dataclass(frozen=True)
class CritiqueSuggestion:
    snippet: str
    source_bullet: str
    extracted_via: str
    adopted_in: List[SuggestionMatch] = field(default_factory=list)


@dataclass(frozen=True)
class CritiqueUptake:
    critic_path: str
    suggestions_total: int
    adopted_in_candidates_count: int
    adopted_in_merge_count: int
    parse_status: str
    suggestions: List[CritiqueSuggestion] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentAttribution:
    segment_index: int
    segment_preview: str
    origin_kind: str
    origin_path: Optional[str]
    score: float


@dataclass(frozen=True)
class TermFirstSeen:
    term: str
    first_seen_path: str
    first_seen_kind: str
    first_seen_step_index: Optional[int] = None
    injected: bool = False


@dataclass(frozen=True)
class EvolutionSection:
    section_key: str
    ordering: str
    base_path: Optional[str]
    merge_path: Optional[str]
    base_step_index: Optional[int] = None
    merge_step_index: Optional[int] = None
    candidate_paths: List[str] = field(default_factory=list)
    candidate_step_indices: List[Optional[int]] = field(default_factory=list)
    critique_paths: List[str] = field(default_factory=list)
    critique_step_indices: List[Optional[int]] = field(default_factory=list)
    node_paths_in_order: List[str] = field(default_factory=list)
    node_step_indices_in_order: List[Optional[int]] = field(default_factory=list)

    deltas: List[DeltaMetrics] = field(default_factory=list)
    candidate_similarity: List[PairSimilarity] = field(default_factory=list)
    findings: List[EvolutionFinding] = field(default_factory=list)

    critiques: List[CritiqueUptake] = field(default_factory=list)
    merge_provenance: List[SegmentAttribution] = field(default_factory=list)
    merge_branch_contributions: Dict[str, float] = field(default_factory=dict)

    critique_provenance_targets: List[str] = field(default_factory=list)
    critique_provenance_target_step_indices: List[Optional[int]] = field(default_factory=list)
    critique_provenance_by_critic: Dict[str, Dict[str, float]] = field(default_factory=dict)
    critique_provenance_unattributed_by_target: Dict[str, float] = field(default_factory=dict)
    critique_provenance_segments_total_by_target: Dict[str, int] = field(default_factory=dict)

    term_first_seen: List[TermFirstSeen] = field(default_factory=list)
    node_introduced_terms: Dict[str, List[str]] = field(default_factory=dict)

    issues: List[Issue] = field(default_factory=list)


@dataclass(frozen=True)
class EvolutionReport:
    config: EvolutionConfig = field(default_factory=EvolutionConfig)
    sections: List[EvolutionSection] = field(default_factory=list)


@dataclass
class CompareResult:
    run_a: RunReport
    run_b: RunReport
    added_steps: List[str]
    removed_steps: List[str]
    metadata_changes: Dict[str, Dict[str, Any]]
    injector_diffs: List[str]
    post_processing_diffs: List[str] = field(default_factory=list)

