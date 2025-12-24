from __future__ import annotations

import dataclasses
import difflib
import re
from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Mapping

from .report_model import (
    CritiqueSuggestion,
    CritiqueUptake,
    DeltaMetrics,
    EvolutionConfig,
    EvolutionFinding,
    EvolutionReport,
    EvolutionSection,
    EvolutionThresholds,
    Issue,
    PairSimilarity,
    SegmentAttribution,
    StepReport,
    SuggestionMatch,
    TermFirstSeen,
)

_CANDIDATE_RE = re.compile(r"^consensus(?:_\d+)?$", re.IGNORECASE)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "over",
        "the",
        "this",
        "to",
        "with",
        "without",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "that",
        "these",
        "those",
    }
)

_HEADER_RE = re.compile(r"^\s*#{1,6}\s*(?P<title>[^#]+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(?P<body>.+?)\s*$")
_QUOTE_RE = re.compile(r'["\'](?P<q>[^"\']{5,})["\']')


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").replace("\r\n", "\n").replace("\t", " ").split()).strip()


def _normalize_curly_quotes(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2026", "...")
    )


def _tokens_all(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _tokens_content(text: str) -> list[str]:
    return [t for t in _tokens_all(text) if len(t) >= 3 and t not in _STOPWORDS]


def _count_sentences(text: str) -> int:
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return 0
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return len([p for p in (part.strip() for part in parts) if p])


def _count_nonempty_lines(text: str) -> int:
    return len([ln for ln in (text or "").replace("\r\n", "\n").split("\n") if ln.strip()])


def _token_diff_counts(a_tokens: list[str], b_tokens: list[str]) -> tuple[int, int]:
    matcher = difflib.SequenceMatcher(a=a_tokens, b=b_tokens)
    added = 0
    removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += (j2 - j1)
        elif tag == "delete":
            removed += (i2 - i1)
        elif tag == "replace":
            removed += (i2 - i1)
            added += (j2 - j1)
    return added, removed


def _jaccard(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / float(len(a_set | b_set))


def _combined_similarity(seq_ratio: float, jaccard: float) -> float:
    return 0.65 * seq_ratio + 0.35 * jaccard


def _safe_preview(text: str, limit: int) -> str:
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _segment_text(text: str) -> list[str]:
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    bullet_lines = sum(1 for ln in lines if _BULLET_RE.match(ln))

    if len(lines) >= 12 or (len(lines) >= 6 and bullet_lines / max(1, len(lines)) >= 0.25):
        return lines

    segments: list[str] = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            seg = part.strip()
            if seg:
                segments.append(seg)
    return segments


def _best_match_for_segment_index(
    segment: str,
    *,
    origin_segments: list[str],
) -> tuple[float, int] | None:
    seg_norm = _normalize_ws(_normalize_curly_quotes(segment)).lower()
    if not seg_norm:
        return None
    seg_tokens = _tokens_content(seg_norm)

    best: tuple[float, float, float, int] | None = None
    best_idx: int | None = None
    for idx, origin in enumerate(origin_segments):
        origin_norm = _normalize_ws(_normalize_curly_quotes(origin)).lower()
        if not origin_norm:
            continue
        seq_ratio = difflib.SequenceMatcher(a=seg_norm, b=origin_norm).ratio()
        jacc = _jaccard(seg_tokens, _tokens_content(origin_norm))
        score = _combined_similarity(seq_ratio, jacc)
        candidate = (score, seq_ratio, jacc, -idx)
        if best is None or candidate > best:
            best = candidate
            best_idx = idx

    if best is None or best_idx is None:
        return None
    return best[0], best_idx


def _best_critic_for_segment(
    segment: str,
    *,
    critic_snippets: list[tuple[str, str]],
    thresholds: EvolutionThresholds,
) -> tuple[str, float] | None:
    seg_norm = _normalize_ws(_normalize_curly_quotes(segment)).lower()
    if not seg_norm:
        return None
    seg_tokens = _tokens_content(seg_norm)

    best: tuple[float, float, float, str] | None = None
    for critic_path, snippet in critic_snippets:
        sn_norm = _normalize_ws(_normalize_curly_quotes(snippet)).lower()
        if not sn_norm:
            continue
        seq_ratio = difflib.SequenceMatcher(a=seg_norm, b=sn_norm).ratio()
        jacc = _jaccard(seg_tokens, _tokens_content(sn_norm))
        score = _combined_similarity(seq_ratio, jacc)
        candidate = (score, seq_ratio, jacc, critic_path)
        if best is None or candidate > best:
            best = candidate

    if best is None or best[0] < thresholds.provenance_origin_min_score:
        return None
    return best[3], best[0]


def _extract_snippet_from_text(text: str) -> tuple[str, str]:
    cleaned = _normalize_ws(text)
    if not cleaned:
        return "", "empty"

    q = _QUOTE_RE.search(cleaned)
    if q and q.group("q").strip():
        return _normalize_ws(q.group("q")), "quote"

    if "->" in cleaned:
        return _normalize_ws(cleaned.split("->", 1)[1]), "arrow"
    if "\u2192" in cleaned:
        return _normalize_ws(cleaned.split("\u2192", 1)[1]), "arrow"

    if ":" in cleaned:
        after = cleaned.split(":", 1)[1].strip()
        if after:
            return _normalize_ws(after), "colon"
        return "", "colon_empty"

    return cleaned, "text"


def _extract_label_value(line: str) -> tuple[str, str] | None:
    cleaned = _normalize_ws(_normalize_curly_quotes(line))
    if not cleaned:
        return None

    # Support "Replace: ..." and "**Replace:** ..."
    match = re.match(r"^(?:\*\*)?(replace|with)(?:\*\*)?\s*:\s*(.+)$", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    label = match.group(1).lower()
    value = match.group(2).strip()
    return label, value


def _extract_edits_suggestions(text: str) -> tuple[list[CritiqueSuggestion], list[str]]:
    raw = _normalize_curly_quotes((text or "").replace("\r\n", "\n"))
    if not raw.strip():
        return [], ["empty critique response"]

    lines = raw.split("\n")

    edits_start: int | None = None
    for idx, line in enumerate(lines):
        match = _HEADER_RE.match(line)
        if not match:
            continue
        title = match.group("title").strip().lower()
        if title == "edits" or title.startswith("edits "):
            edits_start = idx + 1
            break

    if edits_start is None:
        return [], ["missing '## Edits' section"]

    warnings: list[str] = []
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines[edits_start:]:
        header = _HEADER_RE.match(line)
        if header:
            break

        m = _BULLET_RE.match(line)
        if m:
            if current:
                blocks.append(current)
            body = m.group("body").strip()
            current = [body] if body else []
            continue

        if current is not None and line.strip():
            current.append(line.strip())

    if current:
        blocks.append(current)

    if not blocks:
        return [], ["no bullets found under '## Edits'"]

    suggestions: list[CritiqueSuggestion] = []
    for block_lines in blocks:
        if not block_lines:
            continue

        source_block = _normalize_ws("\n".join(block_lines))

        labelled: list[tuple[str, str]] = []
        for ln in block_lines:
            parsed = _extract_label_value(ln)
            if parsed:
                labelled.append(parsed)

        # Prefer "With:" suggestions; skip "Replace:" snippets (old text).
        with_values = [value for label, value in labelled if label == "with"]
        replace_values = [value for label, value in labelled if label == "replace"]
        if replace_values and not with_values:
            continue

        if with_values:
            for value in with_values:
                snippet, method = _extract_snippet_from_text(value)
                if not snippet:
                    warnings.append(f"empty snippet extracted from With: {source_block!r}")
                    continue
                suggestions.append(
                    CritiqueSuggestion(
                        snippet=snippet,
                        source_bullet=source_block,
                        extracted_via=f"with_{method}",
                    )
                )
            continue

        # Fallback: treat the bullet body itself as a suggestion.
        bullet_body = block_lines[0]
        snippet, method = _extract_snippet_from_text(bullet_body)
        if (not snippet or bullet_body.rstrip().endswith(":")) and len(block_lines) > 1:
            # Instruction-style bullets often put the actual rewrite on the next line.
            snippet2, method2 = _extract_snippet_from_text(block_lines[1])
            if snippet2:
                snippet, method = snippet2, f"continuation_{method2}"

        if not snippet:
            warnings.append(f"empty snippet extracted from bullet: {source_block!r}")
            continue

        suggestions.append(
            CritiqueSuggestion(
                snippet=snippet,
                source_bullet=source_block,
                extracted_via=method,
            )
        )

    if not suggestions:
        return [], warnings or ["no suggestions extracted"]

    return suggestions, warnings


def _best_match_for_snippet(
    snippet: str,
    *,
    target_segments: list[str],
    thresholds: EvolutionThresholds,
) -> tuple[float, float, str] | None:
    best: tuple[float, float, str] | None = None
    snippet_norm = _normalize_ws(snippet).lower()
    snippet_tokens = _tokens_content(snippet_norm)

    for seg in target_segments:
        seg_norm = _normalize_ws(seg).lower()
        if not seg_norm:
            continue
        seq_ratio = difflib.SequenceMatcher(a=snippet_norm, b=seg_norm).ratio()
        jacc = _jaccard(snippet_tokens, _tokens_content(seg_norm))
        if (
            seq_ratio >= thresholds.suggestion_adopted_seq_ratio
            and jacc >= thresholds.suggestion_adopted_jaccard
        ):
            if best is None or (seq_ratio, jacc) > (best[0], best[1]):
                best = (seq_ratio, jacc, seg)

    return best


def _best_match_for_segment(
    segment: str,
    *,
    origin_segments: list[str],
) -> tuple[float, str] | None:
    seg_norm = _normalize_ws(segment).lower()
    if not seg_norm:
        return None
    seg_tokens = _tokens_content(seg_norm)

    best: tuple[float, float, float, str] | None = None
    for origin in origin_segments:
        origin_norm = _normalize_ws(origin).lower()
        if not origin_norm:
            continue
        seq_ratio = difflib.SequenceMatcher(a=seg_norm, b=origin_norm).ratio()
        jacc = _jaccard(seg_tokens, _tokens_content(origin_norm))
        score = _combined_similarity(seq_ratio, jacc)
        candidate = (score, seq_ratio, jacc, origin)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return None
    return best[0], best[3]


def _compute_delta(from_text: str, to_text: str) -> DeltaMetrics:
    a = (from_text or "").strip()
    b = (to_text or "").strip()

    a_tokens = _tokens_all(a)
    b_tokens = _tokens_all(b)
    tokens_added, tokens_removed = _token_diff_counts(a_tokens, b_tokens)

    denom = max(1, len(a_tokens))
    token_change_ratio = (tokens_added + tokens_removed) / float(denom)

    seq_ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
    jacc = _jaccard(_tokens_content(a), _tokens_content(b))

    return DeltaMetrics(
        from_path="",
        to_path="",
        from_chars=len(a),
        to_chars=len(b),
        from_tokens=len(a_tokens),
        to_tokens=len(b_tokens),
        from_lines=_count_nonempty_lines(a),
        to_lines=_count_nonempty_lines(b),
        from_sentences=_count_sentences(a),
        to_sentences=_count_sentences(b),
        tokens_added=tokens_added,
        tokens_removed=tokens_removed,
        token_change_ratio=token_change_ratio,
        seq_ratio=seq_ratio,
        jaccard=jacc,
    )


def _delta_between(from_step: StepReport, to_step: StepReport) -> DeltaMetrics:
    if not isinstance(from_step.response, str) or not from_step.response.strip():
        raise ValueError(f"missing response for {from_step.path}")
    if not isinstance(to_step.response, str) or not to_step.response.strip():
        raise ValueError(f"missing response for {to_step.path}")

    delta = _compute_delta(from_step.response, to_step.response)
    return dataclasses.replace(delta, from_path=from_step.path, to_path=to_step.path)


def _pair_similarity(a_step: StepReport, b_step: StepReport) -> PairSimilarity:
    a_text = (a_step.response or "").strip()
    b_text = (b_step.response or "").strip()
    seq_ratio = difflib.SequenceMatcher(a=a_text, b=b_text).ratio()
    jacc = _jaccard(_tokens_content(a_text), _tokens_content(b_text))
    return PairSimilarity(a_path=a_step.path, b_path=b_step.path, seq_ratio=seq_ratio, jaccard=jacc)


def _section_key(path: str) -> str:
    parts = (path or "").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return path or "unknown"


def _classify_step(step: StepReport) -> str:
    parts = (step.path or "").split("/")
    leaf = (parts[-1] if parts else "").strip()
    leaf_lower = leaf.lower()
    if leaf_lower == "draft":
        return "base"
    if leaf_lower == "final_consensus":
        return "merge"
    if _CANDIDATE_RE.match(leaf):
        return "candidate"
    if "/tot_enclave/" in (step.path or ""):
        return "critique"
    return "other"


def _order_steps_for_section(steps: list[StepReport]) -> tuple[str, list[StepReport]]:
    if not steps:
        return "empty", []

    has_any_start_ts = any(step.timing.start_ts is not None for step in steps)
    if has_any_start_ts:
        ordering = "oplog_start_ts" if all(step.timing.start_ts is not None for step in steps) else "oplog_start_ts_partial"
        return ordering, sorted(
            steps,
            key=lambda s: (
                0 if s.timing.start_ts is not None else 1,
                s.timing.start_ts if s.timing.start_ts is not None else datetime.max,
                s.transcript_index if s.transcript_index is not None else 1_000_000,
                s.step_index if s.step_index is not None else 1_000_000,
                s.path,
            ),
        )

    if any(step.transcript_index is not None for step in steps):
        return "transcript_index", sorted(
            steps,
            key=lambda s: (
                s.transcript_index if s.transcript_index is not None else 1_000_000,
                s.step_index if s.step_index is not None else 1_000_000,
                s.path,
            ),
        )

    if any(step.step_index is not None for step in steps):
        return "step_index", sorted(
            steps,
            key=lambda s: (
                s.step_index if s.step_index is not None else 1_000_000,
                s.path,
            ),
        )

    return "path", sorted(steps, key=lambda s: s.path)


def _extract_injected_terms(context: Mapping[str, Any] | None) -> set[str]:
    if not context:
        return set()

    strings: list[str] = []
    for value in context.values():
        if isinstance(value, str):
            strings.append(value)
            continue
        if isinstance(value, Mapping):
            for sub in value.values():
                if isinstance(sub, str) and sub.strip():
                    strings.append(sub)

    injected: set[str] = set()
    for s in strings:
        injected.update(_tokens_content(s))
    return injected


def analyze_evolution(
    *,
    metadata_context: Mapping[str, Any] | None,
    steps: list[StepReport],
    thresholds: EvolutionThresholds | None = None,
) -> tuple[EvolutionReport, list[Issue]]:
    thresholds = thresholds or EvolutionThresholds()
    config = EvolutionConfig(thresholds=thresholds)

    issues: list[Issue] = []
    sections: dict[str, list[StepReport]] = {}
    for step in steps:
        key = _section_key(step.path)
        if not key.startswith("pipeline/"):
            continue
        sections.setdefault(key, []).append(step)

    injected_terms = _extract_injected_terms(metadata_context)

    out_sections: list[EvolutionSection] = []
    for section_key, section_steps in sorted(sections.items(), key=lambda kv: kv[0]):
        section_issues: list[Issue] = []
        ordering, ordered_steps = _order_steps_for_section(section_steps)
        node_paths_in_order: list[str] = []
        node_step_indices_in_order: list[int | None] = []
        for step in ordered_steps:
            if not step.path:
                continue
            node_paths_in_order.append(step.path)
            node_step_indices_in_order.append(step.step_index)

        def unique_last_by_path(items: list[StepReport]) -> list[StepReport]:
            seen: set[str] = set()
            out_rev: list[StepReport] = []
            for item in reversed(items):
                if item.path and item.path not in seen:
                    out_rev.append(item)
                    seen.add(item.path)
            return list(reversed(out_rev))

        classified: dict[str, list[StepReport]] = {
            "base": [],
            "candidate": [],
            "merge": [],
            "critique": [],
            "other": [],
        }
        for step in ordered_steps:
            kind = _classify_step(step)
            classified[kind].append(step)

        base = classified["base"][-1] if classified["base"] else None
        merge = classified["merge"][-1] if classified["merge"] else None
        candidates_raw = list(classified["candidate"])
        critiques_raw = list(classified["critique"])
        candidates = unique_last_by_path(candidates_raw)
        critiques = unique_last_by_path(critiques_raw)

        if len(classified["base"]) > 1 and base:
            section_issues.append(
                Issue(
                    "info",
                    "evolution_multiple_base_steps",
                    f"Multiple /draft steps found; using last by ordering: {base.path}",
                    path=section_key,
                )
            )
        if len(classified["merge"]) > 1 and merge:
            section_issues.append(
                Issue(
                    "info",
                    "evolution_multiple_merge_steps",
                    f"Multiple /final_consensus steps found; using last by ordering: {merge.path}",
                    path=section_key,
                )
            )
        if len(candidates_raw) > len(candidates):
            section_issues.append(
                Issue(
                    "info",
                    "evolution_deduped_candidates",
                    f"Collapsed {len(candidates_raw)} candidate steps to {len(candidates)} unique paths (kept last occurrence per path).",
                    path=section_key,
                )
            )
        if len(critiques_raw) > len(critiques):
            section_issues.append(
                Issue(
                    "info",
                    "evolution_deduped_critiques",
                    f"Collapsed {len(critiques_raw)} critique steps to {len(critiques)} unique paths (kept last occurrence per path).",
                    path=section_key,
                )
            )

        if base is None:
            section_issues.append(
                Issue(
                    "warn",
                    "evolution_incomplete",
                    "Evolution base node missing (no /draft step found).",
                    path=section_key,
                )
            )
        if merge is None:
            section_issues.append(
                Issue(
                    "warn",
                    "evolution_incomplete",
                    "Evolution merge node missing (no /final_consensus step found).",
                    path=section_key,
                )
            )
        if not candidates:
            section_issues.append(
                Issue(
                    "info",
                    "evolution_incomplete",
                    "No candidate nodes found (no /consensus(_N) steps).",
                    path=section_key,
                )
            )

        deltas: list[DeltaMetrics] = []
        similarity: list[PairSimilarity] = []
        findings: list[EvolutionFinding] = []
        critique_uptake: list[CritiqueUptake] = []
        merge_provenance: list[SegmentAttribution] = []
        merge_branch_contributions: dict[str, float] = {}
        critique_provenance_targets: list[str] = []
        critique_provenance_target_step_indices: list[int | None] = []
        critique_provenance_by_critic: dict[str, dict[str, float]] = {}
        critique_provenance_unattributed_by_target: dict[str, float] = {}
        critique_provenance_segments_total_by_target: dict[str, int] = {}
        term_first_seen: list[TermFirstSeen] = []
        node_introduced_terms: dict[str, list[str]] = {}

        if base and merge:
            try:
                deltas.append(_delta_between(base, merge))
            except Exception as exc:
                section_issues.append(
                    Issue(
                        "warn",
                        "evolution_delta_failed",
                        f"Failed base->merge delta: {exc}",
                        path=section_key,
                    )
                )
        if base:
            for cand in candidates:
                try:
                    deltas.append(_delta_between(base, cand))
                except Exception as exc:
                    section_issues.append(
                        Issue(
                            "warn",
                            "evolution_delta_failed",
                            f"Failed base->candidate delta ({cand.path}): {exc}",
                            path=section_key,
                        )
                    )
        if merge:
            for cand in candidates:
                try:
                    deltas.append(_delta_between(cand, merge))
                except Exception as exc:
                    section_issues.append(
                        Issue(
                            "warn",
                            "evolution_delta_failed",
                            f"Failed candidate->merge delta ({cand.path}): {exc}",
                            path=section_key,
                        )
                    )

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                similarity.append(_pair_similarity(candidates[i], candidates[j]))

        if similarity:
            near_pairs = [
                p
                for p in similarity
                if p.seq_ratio >= thresholds.candidate_redundancy_seq_ratio
                and p.jaccard >= thresholds.candidate_redundancy_jaccard
            ]
            if near_pairs:
                top = max(near_pairs, key=lambda p: (p.seq_ratio, p.jaccard))
                findings.append(
                    EvolutionFinding(
                        type="candidate_redundancy",
                        severity="warn",
                        message="Candidate drafts appear near-identical (likely redundant refinement).",
                        support={
                            "example_pair": {"a": top.a_path, "b": top.b_path},
                            "seq_ratio": top.seq_ratio,
                            "jaccard": top.jaccard,
                        },
                    )
                )

        if base and merge:
            base_merge_delta = next(
                (d for d in deltas if d.from_path == base.path and d.to_path == merge.path),
                None,
            )
            if base_merge_delta and (
                base_merge_delta.seq_ratio >= thresholds.low_delta_merge_seq_ratio
                or base_merge_delta.token_change_ratio <= thresholds.low_delta_merge_token_change_ratio
            ):
                findings.append(
                    EvolutionFinding(
                        type="low_delta_merge",
                        severity="info",
                        message="Merge output is very close to base output (low-value refinement).",
                        support={
                            "base": base.path,
                            "merge": merge.path,
                            "seq_ratio": base_merge_delta.seq_ratio,
                            "token_change_ratio": base_merge_delta.token_change_ratio,
                        },
                    )
                )

        if merge and candidates and isinstance(merge.response, str):
            merge_sims: list[tuple[str, float, float]] = []
            for cand in candidates:
                if not isinstance(cand.response, str):
                    continue
                seq_ratio = difflib.SequenceMatcher(
                    a=merge.response.strip(), b=cand.response.strip()
                ).ratio()
                jacc = _jaccard(_tokens_content(merge.response), _tokens_content(cand.response))
                merge_sims.append((cand.path, seq_ratio, jacc))

            merge_sims.sort(key=lambda t: (t[1], t[2]), reverse=True)
            if len(merge_sims) >= 2:
                best_path, best_seq, best_jacc = merge_sims[0]
                _second_path, second_seq, _second_jacc = merge_sims[1]
                if (
                    best_seq >= thresholds.merge_near_single_candidate_seq_ratio
                    and (best_seq - second_seq) >= thresholds.merge_near_single_candidate_margin
                    and second_seq <= thresholds.merge_near_single_candidate_other_max
                ):
                    findings.append(
                        EvolutionFinding(
                            type="merge_near_single_candidate",
                            severity="info",
                            message="Merge output closely matches a single candidate (other branches may be unused).",
                            support={
                                "best_candidate": best_path,
                                "best_seq_ratio": best_seq,
                                "best_jaccard": best_jacc,
                                "second_seq_ratio": second_seq,
                            },
                        )
                    )

        candidate_segments_by_path: dict[str, list[str]] = {}
        for cand in candidates:
            candidate_segments_by_path[cand.path] = (
                _segment_text(cand.response) if isinstance(cand.response, str) else []
            )
        merge_segments: list[str] = (
            _segment_text(merge.response) if merge and isinstance(merge.response, str) else []
        )

        if critiques:
            for critic in critiques:
                suggestions: list[CritiqueSuggestion] = []
                warnings: list[str] = []
                if isinstance(critic.response, str) and critic.response.strip():
                    suggestions, warnings = _extract_edits_suggestions(critic.response)
                else:
                    warnings = ["missing critique response"]

                parse_status = "ok" if suggestions else "unparseable"
                if parse_status != "ok":
                    section_issues.append(
                        Issue(
                            "warn",
                            "evolution_unparseable_critique",
                            f"Could not parse critique edits suggestions for {critic.path}: "
                            + (warnings[0] if warnings else "unparseable critique format"),
                            path=critic.path,
                        )
                    )

                adopted_in_candidates = 0
                adopted_in_merge = 0
                suggestions_out: list[CritiqueSuggestion] = []
                for suggestion in suggestions:
                    adopted: list[SuggestionMatch] = []

                    for cand in candidates:
                        segments = candidate_segments_by_path.get(cand.path, [])
                        if not segments:
                            continue
                        match = _best_match_for_snippet(
                            suggestion.snippet, target_segments=segments, thresholds=thresholds
                        )
                        if match:
                            seq_ratio, jacc, seg = match
                            adopted.append(
                                SuggestionMatch(
                                    target_path=cand.path,
                                    target_kind="candidate",
                                    segment_preview=_safe_preview(seg, thresholds.max_segment_preview_chars),
                                    seq_ratio=seq_ratio,
                                    jaccard=jacc,
                                )
                            )

                    if merge_segments:
                        match = _best_match_for_snippet(
                            suggestion.snippet, target_segments=merge_segments, thresholds=thresholds
                        )
                        if match:
                            seq_ratio, jacc, seg = match
                            adopted.append(
                                SuggestionMatch(
                                    target_path=merge.path,
                                    target_kind="merge",
                                    segment_preview=_safe_preview(seg, thresholds.max_segment_preview_chars),
                                    seq_ratio=seq_ratio,
                                    jaccard=jacc,
                                )
                            )

                    suggestions_out.append(dataclasses.replace(suggestion, adopted_in=adopted))
                    if any(m.target_kind == "candidate" for m in adopted):
                        adopted_in_candidates += 1
                    if any(m.target_kind == "merge" for m in adopted):
                        adopted_in_merge += 1

                critique_uptake.append(
                    CritiqueUptake(
                        critic_path=critic.path,
                        suggestions_total=len(suggestions_out),
                        adopted_in_candidates_count=adopted_in_candidates,
                        adopted_in_merge_count=adopted_in_merge,
                        parse_status=parse_status,
                        suggestions=suggestions_out,
                        warnings=warnings,
                    )
                )

        if merge_segments and (base or candidates):
            try:
                base_segments = (
                    _segment_text(base.response) if base and isinstance(base.response, str) else []
                )
                suggestion_snippets: list[tuple[str, str]] = []
                for cu in critique_uptake:
                    if cu.parse_status != "ok":
                        continue
                    for s in cu.suggestions:
                        suggestion_snippets.append((cu.critic_path, s.snippet))

                origin_priority = {"candidate": 2, "base": 1, "critic_suggestion": 0}
                for idx, segment in enumerate(merge_segments):
                    seg_norm = _normalize_ws(segment).lower()
                    seg_tokens = _tokens_content(seg_norm)

                    base_best: tuple[float, str] | None = None
                    if base_segments:
                        base_match = _best_match_for_segment(segment, origin_segments=base_segments)
                        if base_match:
                            base_best = (base_match[0], base.path if base else "")

                    cand_best: tuple[float, str] | None = None
                    for cand in candidates:
                        segments = candidate_segments_by_path.get(cand.path, [])
                        if not segments:
                            continue
                        cand_match = _best_match_for_segment(segment, origin_segments=segments)
                        if not cand_match:
                            continue
                        candidate = (cand_match[0], cand.path)
                        if cand_best is None or candidate > cand_best:
                            cand_best = candidate

                    sugg_best_full: tuple[float, float, float, str] | None = None
                    for critic_path, snippet in suggestion_snippets:
                        snippet_norm = _normalize_ws(snippet).lower()
                        if not snippet_norm:
                            continue
                        seq_ratio = difflib.SequenceMatcher(a=seg_norm, b=snippet_norm).ratio()
                        jacc = _jaccard(seg_tokens, _tokens_content(snippet_norm))
                        score = _combined_similarity(seq_ratio, jacc)
                        candidate = (score, seq_ratio, jacc, critic_path)
                        if sugg_best_full is None or candidate > sugg_best_full:
                            sugg_best_full = candidate

                    sugg_best: tuple[float, str] | None = None
                    if sugg_best_full is not None:
                        sugg_best = (sugg_best_full[0], sugg_best_full[3])

                    candidates_scored: list[tuple[str, str | None, float]] = []
                    if cand_best:
                        candidates_scored.append(("candidate", cand_best[1], cand_best[0]))
                    if base_best:
                        candidates_scored.append(("base", base_best[1], base_best[0]))
                    if sugg_best:
                        candidates_scored.append(("critic_suggestion", sugg_best[1], sugg_best[0]))

                    if not candidates_scored:
                        origin_kind = "merge_new"
                        origin_path = None
                        best_score = 0.0
                    else:
                        origin_kind, origin_path, best_score = max(
                            candidates_scored,
                            key=lambda t: (
                                t[2],
                                origin_priority.get(t[0], -1),
                                t[1] or "",
                            ),
                        )
                        if best_score < thresholds.provenance_origin_min_score:
                            origin_kind = "merge_new"
                            origin_path = None

                    merge_provenance.append(
                        SegmentAttribution(
                            segment_index=idx,
                            segment_preview=_safe_preview(segment, thresholds.max_segment_preview_chars),
                            origin_kind=origin_kind,
                            origin_path=origin_path,
                            score=best_score,
                        )
                    )

                if merge_provenance and candidates:
                    total = len(merge_provenance)
                    counts = Counter(
                        a.origin_path
                        for a in merge_provenance
                        if a.origin_kind == "candidate" and a.origin_path
                    )
                    merge_branch_contributions = {
                        c.path: (counts.get(c.path, 0) / float(total)) for c in candidates
                    }
            except Exception as exc:
                section_issues.append(
                    Issue(
                        "warn",
                        "evolution_provenance_failed",
                        f"Failed merge provenance attribution: {exc}",
                        path=section_key,
                    )
                )

        try:
            # Critique provenance matrix: estimate which critic most likely influenced each candidate/merge segment.
            targets: list[StepReport] = []
            targets.extend(candidates)
            if merge:
                targets.append(merge)
            critique_provenance_targets = []
            critique_provenance_target_step_indices = []
            for target in targets:
                if not target.path:
                    continue
                critique_provenance_targets.append(target.path)
                critique_provenance_target_step_indices.append(target.step_index)

            critic_steps = list(critiques)
            critic_snippets: list[tuple[str, str]] = []
            for cu in critique_uptake:
                if cu.parse_status != "ok":
                    continue
                for s in cu.suggestions:
                    if s.snippet:
                        critic_snippets.append((cu.critic_path, s.snippet))

            if critic_steps and targets and critic_snippets:
                candidate_segment_best_critic: dict[str, list[str | None]] = {}
                for cand in candidates:
                    segs = candidate_segments_by_path.get(cand.path, [])
                    mapping: list[str | None] = []
                    for seg in segs:
                        best = _best_critic_for_segment(seg, critic_snippets=critic_snippets, thresholds=thresholds)
                        mapping.append(best[0] if best else None)
                    candidate_segment_best_critic[cand.path] = mapping

                merge_attr_by_index = {a.segment_index: a for a in merge_provenance} if merge_provenance else {}

                for target in targets:
                    if not target.path:
                        continue

                    if merge and target.path == merge.path:
                        segs = merge_segments
                    else:
                        segs = candidate_segments_by_path.get(target.path, [])

                    total = len(segs)
                    critique_provenance_segments_total_by_target[target.path] = total
                    if total == 0:
                        critique_provenance_unattributed_by_target[target.path] = 1.0
                        continue

                    counts: Counter[str] = Counter()
                    unattributed = 0
                    for seg_idx, seg in enumerate(segs):
                        critic: str | None = None

                        if merge and target.path == merge.path:
                            direct = _best_critic_for_segment(
                                seg, critic_snippets=critic_snippets, thresholds=thresholds
                            )
                            if direct:
                                critic = direct[0]
                            else:
                                origin = merge_attr_by_index.get(seg_idx)
                                if origin and origin.origin_kind == "candidate" and origin.origin_path:
                                    cand_path = origin.origin_path
                                    cand_segs = candidate_segments_by_path.get(cand_path, [])
                                    cand_map = candidate_segment_best_critic.get(cand_path, [])
                                    if cand_segs and cand_map:
                                        best_idx = _best_match_for_segment_index(seg, origin_segments=cand_segs)
                                        if best_idx:
                                            _score, cand_seg_idx = best_idx
                                            if 0 <= cand_seg_idx < len(cand_map) and cand_map[cand_seg_idx]:
                                                critic = cand_map[cand_seg_idx]
                        else:
                            mapping = candidate_segment_best_critic.get(target.path, [])
                            if 0 <= seg_idx < len(mapping):
                                critic = mapping[seg_idx]

                        if critic:
                            counts[critic] += 1
                        else:
                            unattributed += 1

                    critique_provenance_unattributed_by_target[target.path] = unattributed / float(total)
                    for critic_path, count in counts.items():
                        critique_provenance_by_critic.setdefault(critic_path, {})[target.path] = count / float(total)
        except Exception as exc:
            section_issues.append(
                Issue(
                    "warn",
                    "evolution_critique_provenance_failed",
                    f"Failed critique provenance matrix: {exc}",
                    path=section_key,
                )
            )

        try:
            base_counter = (
                Counter(_tokens_content(base.response)) if base and isinstance(base.response, str) else Counter()
            )
            candidate_counters = {
                c.path: (Counter(_tokens_content(c.response)) if isinstance(c.response, str) else Counter())
                for c in candidates
            }
            merge_counter = (
                Counter(_tokens_content(merge.response))
                if merge and isinstance(merge.response, str)
                else Counter()
            )

            base_terms = set(base_counter)
            candidate_terms_union: set[str] = set()
            for c in candidate_counters.values():
                candidate_terms_union.update(c.keys())

            for cand in candidates:
                counter = candidate_counters.get(cand.path, Counter())
                introduced = [t for t in counter.keys() if t not in base_terms]
                introduced.sort(key=lambda t: (-counter[t], t))
                node_introduced_terms[cand.path] = introduced[: thresholds.max_node_introduced_terms]

            if merge and merge_counter:
                merge_introduced = [
                    t
                    for t in merge_counter.keys()
                    if t not in base_terms and t not in candidate_terms_union
                ]
                merge_introduced.sort(key=lambda t: (-merge_counter[t], t))
                node_introduced_terms[merge.path] = merge_introduced[: thresholds.max_node_introduced_terms]

            primary_counter: Counter[str]
            if merge_counter:
                primary_counter = merge_counter
            elif candidate_counters:
                combined: Counter[str] = Counter()
                for c in candidate_counters.values():
                    combined.update(c)
                primary_counter = combined
            else:
                primary_counter = base_counter

            primary_terms = [t for t, _count in sorted(primary_counter.items(), key=lambda kv: (-kv[1], kv[0]))]
            primary_terms = primary_terms[: thresholds.max_term_index]

            for term in primary_terms:
                first_path: str | None = None
                first_kind: str | None = None
                if base and term in base_counter:
                    first_path = base.path
                    first_kind = "base"
                else:
                    for cand in candidates:
                        if term in candidate_counters.get(cand.path, Counter()):
                            first_path = cand.path
                            first_kind = "candidate"
                            break
                if first_path is None and merge and term in merge_counter:
                    first_path = merge.path
                    first_kind = "merge"
                if first_path is None or first_kind is None:
                    continue

                first_seen_step_index: int | None = None
                if first_kind == "base" and base:
                    first_seen_step_index = base.step_index
                elif first_kind == "candidate":
                    first_seen_step_index = next((c.step_index for c in candidates if c.path == first_path), None)
                elif first_kind == "merge" and merge:
                    first_seen_step_index = merge.step_index
                term_first_seen.append(
                    TermFirstSeen(
                        term=term,
                        first_seen_path=first_path,
                        first_seen_kind=first_kind,
                        first_seen_step_index=first_seen_step_index,
                        injected=term in injected_terms,
                    )
                )
        except Exception as exc:
            section_issues.append(
                Issue(
                    "warn",
                    "evolution_terms_failed",
                    f"Failed term provenance analysis: {exc}",
                    path=section_key,
                )
            )

        out_sections.append(
            EvolutionSection(
                section_key=section_key,
                ordering=ordering,
                base_path=base.path if base else None,
                base_step_index=base.step_index if base else None,
                merge_path=merge.path if merge else None,
                merge_step_index=merge.step_index if merge else None,
                candidate_paths=[c.path for c in candidates],
                candidate_step_indices=[c.step_index for c in candidates],
                critique_paths=[c.path for c in critiques],
                critique_step_indices=[c.step_index for c in critiques],
                node_paths_in_order=node_paths_in_order,
                node_step_indices_in_order=node_step_indices_in_order,
                deltas=deltas,
                candidate_similarity=similarity,
                findings=findings,
                critiques=critique_uptake,
                merge_provenance=merge_provenance,
                merge_branch_contributions=merge_branch_contributions,
                critique_provenance_targets=critique_provenance_targets,
                critique_provenance_target_step_indices=critique_provenance_target_step_indices,
                critique_provenance_by_critic=critique_provenance_by_critic,
                critique_provenance_unattributed_by_target=critique_provenance_unattributed_by_target,
                critique_provenance_segments_total_by_target=critique_provenance_segments_total_by_target,
                term_first_seen=term_first_seen,
                node_introduced_terms=node_introduced_terms,
                issues=section_issues,
            )
        )
        issues.extend(section_issues)

    return EvolutionReport(config=config, sections=out_sections), issues


def thresholds_from_overrides(overrides: Mapping[str, Any]) -> EvolutionThresholds:
    if not isinstance(overrides, Mapping):
        raise TypeError("evolution thresholds override must be a JSON object")

    allowed = {field.name for field in dataclasses.fields(EvolutionThresholds)}
    updates: dict[str, Any] = {}
    defaults = EvolutionThresholds()

    for key, value in overrides.items():
        if key not in allowed:
            known = ", ".join(sorted(allowed))
            raise ValueError(f"Unknown evolution threshold key {key!r}. Known keys: {known}")
        if isinstance(value, bool):
            raise ValueError(f"Invalid evolution threshold {key!r}: bool is not allowed")

        expected_type = type(getattr(defaults, key))
        if expected_type is int:
            if not isinstance(value, int):
                raise TypeError(f"Invalid evolution threshold {key!r}: expected int, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"Invalid evolution threshold {key!r}: must be > 0")
            updates[key] = value
        elif expected_type is float:
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Invalid evolution threshold {key!r}: expected float, got {type(value).__name__}"
                )
            as_float = float(value)
            if not (0.0 <= as_float <= 1.0):
                raise ValueError(f"Invalid evolution threshold {key!r}: must be in [0, 1]")
            updates[key] = as_float
        else:
            raise TypeError(f"Invalid evolution threshold schema for {key!r}: {expected_type!r}")

    return dataclasses.replace(EvolutionThresholds(), **updates)
