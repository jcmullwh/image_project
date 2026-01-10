from __future__ import annotations

import html
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .report_model import CompareResult, RunReport, StepReport

_ID_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_id(value: str) -> str:
    if not value:
        return "unknown"
    cleaned = _ID_SAFE_RE.sub("-", value.strip()).strip("-").lower()
    return cleaned or "unknown"


def _format_duration_ms(duration_ms: Optional[float]) -> str:
    if duration_ms is None:
        return "n/a"
    if duration_ms >= 60_000:
        return f"{duration_ms/60_000:.1f} min"
    if duration_ms >= 1_000:
        return f"{duration_ms/1_000:.2f} s"
    return f"{duration_ms:.0f} ms"


def _escape_json(value: Any) -> str:
    try:
        return html.escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    except TypeError:
        return html.escape(str(value))


def _format_meta_table(report: RunReport) -> str:
    meta = report.metadata
    rows: List[Tuple[str, str]] = []
    rows.append(("Generation ID", html.escape(meta.generation_id)))
    if meta.seed is not None:
        rows.append(("Seed", html.escape(str(meta.seed))))
    if meta.created_at:
        rows.append(("Created at", html.escape(meta.created_at)))
    if meta.experiment:
        rows.append(("Experiment", _escape_json(meta.experiment)))
    if meta.prompt_pipeline:
        requested = meta.prompt_pipeline.get("requested_plan") if isinstance(meta.prompt_pipeline, dict) else None
        resolved = meta.prompt_pipeline.get("plan") if isinstance(meta.prompt_pipeline, dict) else None
        capture = meta.prompt_pipeline.get("capture_stage") if isinstance(meta.prompt_pipeline, dict) else None
        stages = meta.prompt_pipeline.get("resolved_stages") if isinstance(meta.prompt_pipeline, dict) else None
        if requested or resolved:
            rows.append(("Prompt plan", html.escape(f"{requested or 'n/a'} -> {resolved or 'n/a'}")))
        if isinstance(stages, list) and stages:
            rows.append(("Resolved stages", html.escape(", ".join(str(s) for s in stages))))
        if capture:
            rows.append(("Capture stage", html.escape(str(capture))))
        rows.append(("Prompt pipeline", _escape_json(meta.prompt_pipeline)))
    if meta.selected_concepts:
        rows.append(("Selected concepts", html.escape(", ".join(meta.selected_concepts))))
    if meta.context:
        rows.append(("Context", _escape_json(meta.context)))
    if meta.title_generation:
        rows.append(("Title", _escape_json(meta.title_generation)))
    if meta.image_path:
        rows.append(("Image path", html.escape(meta.image_path)))
    if meta.artifact_paths:
        rows.append(("Artifacts", _escape_json(meta.artifact_paths)))
    if report.run_start_ts or report.run_end_ts or report.runtime_ms is not None:
        rows.append(("Run start", html.escape(report.run_start_ts.isoformat() if report.run_start_ts else "n/a")))
        rows.append(("Run end", html.escape(report.run_end_ts.isoformat() if report.run_end_ts else "n/a")))
        rows.append(("Runtime", html.escape(_format_duration_ms(report.runtime_ms))))
    return "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)


def _render_issues(report: RunReport) -> str:
    if not report.issues:
        return "<p>No issues detected.</p>"

    severity_order = {"error": 0, "warn": 1, "info": 2}
    ordered = sorted(
        report.issues,
        key=lambda issue: (
            severity_order.get(issue.severity.lower(), 99),
            issue.code,
            issue.path or "",
            issue.message,
        ),
    )
    items = []
    for issue in ordered:
        sev = issue.severity.lower()
        items.append(
            "<li>"
            f"<span class='sev sev-{html.escape(sev)}'>{html.escape(sev.upper())}</span> "
            f"<span class='code'>[{html.escape(issue.code)}]</span> "
            f"{html.escape(issue.message)}"
            + (f" <span class='path'>({html.escape(issue.path)})</span>" if issue.path else "")
            + "</li>"
        )
    return f"<ul class='issues'>{''.join(items)}</ul>"


def _group_steps(report: RunReport) -> Dict[str, List[StepReport]]:
    groups: Dict[str, List[StepReport]] = {}
    for step in report.steps:
        parts = step.path.split("/")
        key = "/".join(parts[:2]) if len(parts) >= 2 else step.path
        groups.setdefault(key, []).append(step)
    return groups


def _group_steps_tree(report: RunReport) -> Dict[str, Dict[str, List[StepReport]]]:
    tree: Dict[str, Dict[str, List[StepReport]]] = {}
    for step in report.steps:
        parts = step.path.split("/")
        group2 = "/".join(parts[:2]) if len(parts) >= 2 else step.path
        group3 = parts[2] if len(parts) >= 3 else ""
        tree.setdefault(group2, {}).setdefault(group3, []).append(step)
    return tree


def _step_anchor_id(step: StepReport) -> str:
    if step.step_index is None:
        raise ValueError(f"StepReport.step_index missing for path={step.path!r}")
    return f"step-{step.step_index:04d}-{_safe_id(step.path)}"


def _path_to_anchor_map(report: RunReport) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for step in report.steps:
        mapping[step.path] = _step_anchor_id(step)
    return mapping


def _truncate_preview(text: str, *, limit: int = 180) -> str:
    stripped = text.replace("\r\n", "\n").strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"


def _render_parser_health_banner(report: RunReport) -> str:
    health_issues = [
        issue
        for issue in report.issues
        if issue.code in {"oplog_parse_failed", "oplog_low_coverage"}
    ]
    if not health_issues:
        return ""

    severity = "error" if any(i.severity.lower() == "error" for i in health_issues) else "warn"
    stats = report.metadata.oplog_stats or {}

    detected_format = stats.get("detected_format") or "unknown"
    total = stats.get("total_lines")
    parsed = stats.get("parsed_lines")
    coverage = stats.get("coverage")
    event_count = stats.get("event_count")

    details = []
    if total is not None and parsed is not None and coverage is not None:
        details.append(f"coverage {coverage:.1%} ({parsed}/{total})")
    if event_count is not None:
        details.append(f"events {event_count}")
    details.append(f"format {detected_format}")
    detail_text = " | ".join(details)

    headline = "Oplog parsing degraded" if severity == "warn" else "Oplog parsing failed"
    issue_text = "; ".join(sorted({i.message for i in health_issues}))
    return (
        f"<div class='banner banner-{html.escape(severity)}'>"
        f"<div><strong>{html.escape(headline)}</strong> <span class='muted'>{html.escape(detail_text)}</span></div>"
        f"<div class='muted'>{html.escape(issue_text)}</div>"
        "</div>"
    )


def _step_card(step: StepReport) -> str:
    dur = _format_duration_ms(step.timing.duration_ms)

    def metric(label: str, primary: Optional[int], oplog: Optional[int]) -> str | None:
        if primary is None and oplog is None:
            return None
        if primary is None:
            return f"{label} {oplog} (oplog)"
        if oplog is None or oplog == primary:
            return f"{label} {primary}"
        return f"{label} {primary} (oplog {oplog})"

    size_bits = [
        metric("prompt", step.prompt_chars, step.oplog_prompt_chars),
        metric("context", step.context_chars, step.oplog_context_chars),
        metric("input", step.input_chars, step.oplog_input_chars),
        metric("response", step.response_chars, step.oplog_response_chars),
    ]
    size_txt = ", ".join(bit for bit in size_bits if bit) or "no size counters"

    issues_html = "".join(f"<li>[{html.escape(i.code)}] {html.escape(i.message)}</li>" for i in step.issues)
    issues_block = f"<ul class='step-issues'>{issues_html}</ul>" if issues_html else "<em>No step-specific issues</em>"

    prompt = step.prompt or ""
    response = step.response or ""

    prompt_preview = html.escape(_truncate_preview(prompt) if prompt else "<missing>")
    response_preview = html.escape(_truncate_preview(response) if response else "<missing>")

    prompt_full = html.escape(prompt or "<missing>")
    response_full = html.escape(response or "<missing>")

    start_ts = step.timing.start_ts.isoformat() if step.timing.start_ts else "n/a"
    end_ts = step.timing.end_ts.isoformat() if step.timing.end_ts else "n/a"

    anchor = _step_anchor_id(step)
    return f"""
    <details class="step-card" id="{html.escape(anchor)}">
      <summary>
        <span class="step-name">{html.escape(step.name)}</span>
        <span class="muted">{html.escape(step.path)}</span>
        <span class="pill">{html.escape(dur)}</span>
        <span class="muted">{html.escape(size_txt)}</span>
      </summary>
      <div class="step-body">
        <div class="step-meta">
          <div><strong>Start</strong> {html.escape(start_ts)}</div>
          <div><strong>End</strong> {html.escape(end_ts)}</div>
        </div>
        <div class="step-block">
          <details class="text-block">
            <summary><strong>Prompt</strong> <span class="muted">({len(prompt)} chars)</span> <span class="muted">{prompt_preview}</span></summary>
            <pre>{prompt_full}</pre>
          </details>
          <details class="text-block">
            <summary><strong>Response</strong> <span class="muted">({len(response)} chars)</span> <span class="muted">{response_preview}</span></summary>
            <pre>{response_full}</pre>
          </details>
        </div>
        <div class="step-block"><strong>Issues</strong> {issues_block}</div>
      </div>
    </details>
    """


def _guess_final_prompt_step(steps: List[StepReport]) -> StepReport | None:
    candidates = [s for s in steps if s.response and str(s.response).strip()]
    if not candidates:
        return None

    def score(step: StepReport) -> Tuple[int, datetime, str]:
        path = step.path.lower()
        base = 0
        if "image_prompt_creation" in path or "dalle_prompt_creation" in path:
            base += 2
        if re.search(r"/final_consensus$", path):
            base += 4
        elif re.search(r"/consensus(_\d+)?$", path):
            base += 3
        elif "consensus" in path:
            base += 1
        ts = step.timing.start_ts or datetime.min
        return (base, ts, step.path)

    return max(candidates, key=score)


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _render_evolution(report: RunReport) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    evolution = report.evolution
    if evolution is None:
        evo_issues = [i for i in report.issues if i.code.startswith("evolution_") or i.code == "evolution_failed"]
        issues_html = ""
        if evo_issues:
            items = "".join(
                f"<li><span class='sev sev-{html.escape(i.severity)}'>{html.escape(i.severity.upper())}</span> "
                f"<span class='code'>[{html.escape(i.code)}]</span> "
                f"{html.escape(i.message)}"
                + (f" <span class='path'>({html.escape(i.path)})</span>" if i.path else "")
                + "</li>"
                for i in evo_issues
            )
            issues_html = f"<ul class='issues'>{items}</ul>"
        return (
            "<section id='evolution'><h2>Evolution</h2><p class='muted'>Evolution unavailable.</p>"
            + issues_html
            + "</section>",
            {},
        )

    path_to_anchor = _path_to_anchor_map(report)

    def link(path: Optional[str], step_index: Optional[int] = None, *, label: Optional[str] = None) -> str:
        if not path:
            return "<em class='muted'>missing</em>"
        text = label if label is not None else path
        if step_index is not None:
            anchor = f"step-{step_index:04d}-{_safe_id(path)}"
            return f"<a href='#{html.escape(anchor)}'>{html.escape(text)}</a>"
        anchor = path_to_anchor.get(path)
        if anchor:
            return f"<a href='#{html.escape(anchor)}'>{html.escape(text)}</a>"
        return f"<code>{html.escape(text)}</code>"

    term_index: Dict[str, Dict[str, Any]] = {}
    parts: List[str] = []
    parts.append("<section id='evolution'>")
    parts.append("<h2>Evolution</h2>")
    parts.append(f"<p class='muted'>Sections analyzed: {len(evolution.sections)}</p>")
    parts.append(
        "<details class='evo-help' open><summary><strong>How to read this</strong></summary>"
        "<ul class='muted'>"
        "<li><strong>Model</strong>: base is the section's last <code>/draft</code>; candidates are <code>/consensus</code> (and <code>/consensus_N</code>); merge is the last <code>/final_consensus</code>; critiques are steps under <code>/tot_enclave/</code>.</li>"
        "<li><strong>Deterministic</strong>: all metrics are offline heuristics (no LLM calls). Thresholds are in <code>evolution.config.thresholds</code> and overridable via <code>--evolution-thresholds</code>.</li>"
        "<li><strong>Ordering</strong>: steps are ordered primarily by oplog <code>start_ts</code> when present, otherwise transcript order.</li>"
        "<li><strong>Warnings</strong>: missing nodes or parse failures emit Issues; analysis continues per section (no silent omission).</li>"
        "</ul></details>"
    )

    if not evolution.sections:
        parts.append("<p class='muted'>No pipeline sections found for evolution analysis.</p>")
        parts.append("</section>")
        return "".join(parts), term_index

    toc_items = []
    for section in evolution.sections:
        sec_id = f"evo-sec-{_safe_id(section.section_key)}"
        toc_items.append(f"<li><a href='#{html.escape(sec_id)}'>{html.escape(section.section_key)}</a></li>")
    parts.append("<details class='evo-toc' open><summary><strong>Sections</strong></summary>")
    parts.append(f"<ul class='evo-toc-list'>{''.join(toc_items)}</ul></details>")

    for section in evolution.sections:
        sec_dom_id = f"evo-sec-{_safe_id(section.section_key)}"
        sec_key_id = _safe_id(section.section_key)

        candidate_index_by_path = {
            p: idx
            for p, idx in zip(section.candidate_paths, section.candidate_step_indices)
            if p and idx is not None
        }
        critic_index_by_path = {
            p: idx
            for p, idx in zip(section.critique_paths, section.critique_step_indices)
            if p and idx is not None
        }
        node_index_by_path: Dict[str, Optional[int]] = {}
        if section.base_path:
            node_index_by_path[section.base_path] = section.base_step_index
        if section.merge_path:
            node_index_by_path[section.merge_path] = section.merge_step_index
        for p, idx in zip(section.candidate_paths, section.candidate_step_indices):
            node_index_by_path[p] = idx

        term_index[sec_key_id] = {}
        for t in section.term_first_seen:
            if t.first_seen_step_index is not None:
                anchor = f"step-{t.first_seen_step_index:04d}-{_safe_id(t.first_seen_path)}"
            else:
                anchor = path_to_anchor.get(t.first_seen_path)
            if not anchor or not t.first_seen_path:
                continue
            term_index[sec_key_id][t.term] = {
                "anchor": anchor,
                "path": t.first_seen_path,
                "kind": t.first_seen_kind,
                "injected": bool(t.injected),
            }

        issues_block = ""
        if section.issues:
            items = "".join(
                "<li>"
                f"<span class='sev sev-{html.escape(i.severity.lower())}'>{html.escape(i.severity.upper())}</span> "
                f"<span class='code'>[{html.escape(i.code)}]</span> {html.escape(i.message)}"
                + (f" <span class='path'>({html.escape(i.path)})</span>" if i.path else "")
                + "</li>"
                for i in section.issues
            )
            issues_block = (
                "<details class='evo-issues' open><summary>Evolution issues</summary>"
                f"<ul class='issues'>{items}</ul></details>"
            )

        parts.append(f"<details class='evo-section' open id='{html.escape(sec_dom_id)}'>")
        parts.append(
            f"<summary><strong>{html.escape(section.section_key)}</strong> "
            f"<span class='muted'>ordering={html.escape(section.ordering)}</span></summary>"
        )
        parts.append("<div class='evo-body'>")
        parts.append(issues_block)

        parts.append("<div class='evo-graph'>")
        parts.append(f"<div><strong>Base</strong>: {link(section.base_path, section.base_step_index)}</div>")
        parts.append(f"<div><strong>Merge</strong>: {link(section.merge_path, section.merge_step_index)}</div>")
        if section.candidate_paths:
            cand_links = ", ".join(
                link(p, idx) for p, idx in zip(section.candidate_paths, section.candidate_step_indices)
            )
            parts.append(f"<div><strong>Candidates</strong>: {cand_links}</div>")
        else:
            parts.append("<div><strong>Candidates</strong>: <em class='muted'>none</em></div>")
        if section.critique_paths:
            parts.append(f"<div><strong>Critiques</strong>: {len(section.critique_paths)}</div>")
        parts.append("</div>")

        if section.deltas:
            rows = []
            for d in section.deltas:
                rows.append(
                    "<tr>"
                    f"<td>{link(d.from_path, node_index_by_path.get(d.from_path))}</td>"
                    f"<td>{link(d.to_path, node_index_by_path.get(d.to_path))}</td>"
                    f"<td>{html.escape(_fmt_ratio(d.seq_ratio))}</td>"
                    f"<td>{html.escape(_fmt_ratio(d.jaccard))}</td>"
                    f"<td>{html.escape(f'{d.token_change_ratio*100:.1f}%')}</td>"
                    f"<td>{d.tokens_added}</td>"
                    f"<td>{d.tokens_removed}</td>"
                    "</tr>"
                )
            parts.append("<h4>Delta metrics</h4>")
            parts.append(
                "<p class='muted'>Seq = difflib.SequenceMatcher ratio on raw text (1.0 identical). "
                "Jacc = content-token Jaccard similarity. Tok change = (added+removed tokens) / source token count.</p>"
            )
            parts.append(
                "<div class='evo-table-wrap'><table class='evo-table'>"
                "<tr><th align='left'>From</th><th align='left'>To</th><th>Seq</th><th>Jacc</th><th>Tok change</th><th>+Tok</th><th>-Tok</th></tr>"
                + "".join(rows)
                + "</table></div>"
            )
        else:
            parts.append("<p class='muted'>No delta metrics available for this section.</p>")

        if section.candidate_paths and len(section.candidate_paths) >= 2:
            score_map: Dict[Tuple[str, str], Tuple[float, float]] = {}
            for p in section.candidate_similarity:
                score_map[(p.a_path, p.b_path)] = (p.seq_ratio, p.jaccard)
                score_map[(p.b_path, p.a_path)] = (p.seq_ratio, p.jaccard)

            header_cells = "".join(
                f"<th title='{html.escape(path)}'>{html.escape(path.split('/')[-1])}</th>"
                for path in section.candidate_paths
            )
            body_rows = []
            for a in section.candidate_paths:
                row_cells = []
                for b in section.candidate_paths:
                    if a == b:
                        row_cells.append("<td class='evo-diag'>1.000<br><span class='muted'>1.000</span></td>")
                        continue
                    seq, jacc = score_map.get((a, b), (None, None))
                    if seq is None:
                        row_cells.append("<td class='muted'>n/a</td>")
                    else:
                        row_cells.append(
                            f"<td>{html.escape(_fmt_ratio(seq))}<br><span class='muted'>{html.escape(_fmt_ratio(jacc))}</span></td>"
                        )
                body_rows.append(
                    f"<tr><th title='{html.escape(a)}'>{html.escape(a.split('/')[-1])}</th>{''.join(row_cells)}</tr>"
                )

            parts.append("<h4>Candidate similarity</h4>")
            parts.append(
                "<p class='muted'>Matrix values are Seq (top) and Jacc (bottom). Near-1.0 pairs indicate redundant refinement branches.</p>"
            )
            parts.append(
                "<div class='evo-table-wrap'><table class='evo-table evo-matrix'>"
                f"<tr><th></th>{header_cells}</tr>"
                + "".join(body_rows)
                + "</table></div>"
            )
        elif section.candidate_paths:
            parts.append("<p class='muted'>Candidate similarity unavailable (need at least 2 candidates).</p>")

        if section.findings:
            parts.append("<p class='muted'>Findings are heuristic flags derived from the metrics above.</p>")
            items = "".join(
                "<li>"
                f"<span class='sev sev-{html.escape(f.severity.lower())}'>{html.escape(f.severity.upper())}</span> "
                f"<span class='code'>[{html.escape(f.type)}]</span> {html.escape(f.message)}"
                "</li>"
                for f in section.findings
            )
            parts.append("<h4>Findings</h4>")
            parts.append(f"<ul class='evo-findings'>{items}</ul>")

        if section.critiques:
            parts.append(
                "<p class='muted'>Critique uptake extracts suggestion snippets from each critic's <code>## Edits</code> section (supports Replace/With). "
                "A suggestion is counted as adopted if it matches any candidate/merge segment above thresholds.</p>"
            )
            rows = []
            for c in section.critiques:
                detail_bits = []
                for s in c.suggestions[:12]:
                    adopted = ", ".join(
                        f"{html.escape(m.target_kind)}:{html.escape(m.target_path.split('/')[-1])} ({_fmt_ratio(m.seq_ratio)}/{_fmt_ratio(m.jaccard)})"
                        for m in s.adopted_in[:3]
                    )
                    if adopted:
                        detail_bits.append(
                            f"<li><code>{html.escape(s.snippet)}</code> <span class='muted'>-> {adopted}</span></li>"
                        )
                    else:
                        detail_bits.append(
                            f"<li><code>{html.escape(s.snippet)}</code> <span class='muted'>-> not detected</span></li>"
                        )
                details_html = ""
                if detail_bits:
                    details_html = (
                        "<details class='evo-inline'><summary class='muted'>details</summary>"
                        f"<ul class='evo-suggestions'>{''.join(detail_bits)}</ul></details>"
                    )
                rows.append(
                    "<tr>"
                    f"<td>{link(c.critic_path, critic_index_by_path.get(c.critic_path))}</td>"
                    f"<td>{html.escape(c.parse_status)}</td>"
                    f"<td>{c.suggestions_total}</td>"
                    f"<td>{c.adopted_in_candidates_count}</td>"
                    f"<td>{c.adopted_in_merge_count}</td>"
                    f"<td>{details_html}</td>"
                    "</tr>"
                )

            parts.append("<h4>Critique uptake</h4>")
            parts.append(
                "<div class='evo-table-wrap'><table class='evo-table'>"
                "<tr><th align='left'>Critic</th><th>Parse</th><th>Suggestions</th><th>Adopted in candidates</th><th>Adopted in merge</th><th></th></tr>"
                + "".join(rows)
                + "</table></div>"
            )

        if section.critique_provenance_segments_total_by_target:
            parts.append("<h4>Critique provenance</h4>")
            parts.append(
                "<p class='muted'>Critique provenance estimates how much of each consensus output can be attributed to critics' suggested edits. "
                "Each consensus output is segmented (lines or sentences). Each segment is assigned to the best-matching critic snippet "
                "(from <code>## Edits</code>) if similarity is at least <code>provenance_origin_min_score</code>. "
                "For the merge (final consensus), the tool first tries direct matches; if none, it traces through merge provenance when a segment "
                "originated from a candidate.</p>"
            )

            targets: list[tuple[str, Optional[int]]] = [
                (p, idx)
                for p, idx in zip(
                    section.critique_provenance_targets, section.critique_provenance_target_step_indices
                )
                if p
            ]
            header_cells = "".join(
                "<th title='{full}'>{link}<br><span class='muted'>n={n}</span></th>".format(
                    full=html.escape(path),
                    link=link(path, idx, label=path.split("/")[-1]),
                    n=section.critique_provenance_segments_total_by_target.get(path, 0),
                )
                for path, idx in targets
            )

            uptake_by_path = {c.critic_path: c for c in section.critiques}
            body_rows: list[str] = []
            for critic_path in section.critique_paths:
                uptake = uptake_by_path.get(critic_path)
                parse_status = uptake.parse_status if uptake else "unknown"
                excluded = parse_status != "ok"

                if excluded:
                    cells = "".join("<td class='muted'>n/a</td>" for _ in targets)
                else:
                    per_target = section.critique_provenance_by_critic.get(critic_path, {})
                    cells = "".join(
                        f"<td>{html.escape(_fmt_pct(per_target.get(path, 0.0)))}</td>" for path, _ in targets
                    )

                label_bits = link(critic_path, critic_index_by_path.get(critic_path))
                if excluded and uptake:
                    label_bits += f" <span class='muted'>({html.escape(parse_status)})</span>"
                body_rows.append(f"<tr><th align='left'>{label_bits}</th>{cells}</tr>")

            unattributed_cells = "".join(
                f"<td>{html.escape(_fmt_pct(section.critique_provenance_unattributed_by_target.get(path, 0.0)))}</td>"
                for path, _ in targets
            )
            body_rows.append(
                f"<tr><th align='left'><span class='muted'>Unattributed</span></th>{unattributed_cells}</tr>"
            )

            parts.append(
                "<div class='evo-table-wrap'><table class='evo-table evo-matrix'>"
                f"<tr><th align='left'>Critic</th>{header_cells}</tr>"
                + "".join(body_rows)
                + "</table></div>"
            )
        elif section.critique_paths and (section.candidate_paths or section.merge_path):
            parts.append("<h4>Critique provenance</h4>")
            parts.append(
                "<p class='muted'>Critique provenance unavailable (no parsed critic snippets or no consensus targets in this section).</p>"
            )

        if section.merge_provenance:
            parts.append(
                "<p class='muted'>Merge provenance attributes each merge segment to the best-matching origin among base/candidates/critic suggestions. "
                "<code>merge_new</code> means no origin exceeded the minimum similarity score.</p>"
            )
            contrib_rows = []
            if section.merge_branch_contributions:
                for path, frac in sorted(section.merge_branch_contributions.items(), key=lambda kv: (-kv[1], kv[0])):
                    contrib_rows.append(
                        f"<tr><td>{link(path, candidate_index_by_path.get(path))}</td><td>{html.escape(_fmt_pct(frac))}</td></tr>"
                    )
            contrib_table = (
                "<div class='evo-table-wrap'><table class='evo-table'>"
                "<tr><th align='left'>Candidate</th><th>Share of merge segments</th></tr>"
                + "".join(contrib_rows)
                + "</table></div>"
                if contrib_rows
                else ""
            )

            prov_rows = []
            for a in section.merge_provenance[:250]:
                origin = (
                    link(a.origin_path, node_index_by_path.get(a.origin_path)) if a.origin_path else "<span class='muted'>merge_new</span>"
                )
                prov_rows.append(
                    "<tr>"
                    f"<td>{a.segment_index}</td>"
                    f"<td><code>{html.escape(a.segment_preview)}</code></td>"
                    f"<td>{html.escape(a.origin_kind)}</td>"
                    f"<td>{origin}</td>"
                    f"<td>{html.escape(_fmt_ratio(a.score))}</td>"
                    "</tr>"
                )

            parts.append("<h4>Merge provenance</h4>")
            parts.append(contrib_table)
            parts.append(
                "<details class='evo-provenance'><summary class='muted'>merge segment attribution (first 250)</summary>"
                "<div class='evo-table-wrap'><table class='evo-table'>"
                "<tr><th>#</th><th align='left'>Segment</th><th>Origin kind</th><th align='left'>Origin</th><th>Score</th></tr>"
                + "".join(prov_rows)
                + "</table></div></details>"
            )

        if section.term_first_seen:
            parts.append("<h4>Term tracing</h4>")
            parts.append(
                "<p class='muted'>Content terms are tokenized (stopwords removed). "
                "The index shows where each term first appears among base/candidates/merge. "
                "<span class='pill'>injected</span> means the term is present in run <code>metadata.context</code>.</p>"
            )
            parts.append(
                "<div class='evo-term-search'>"
                f"<label>Term <input id='evo-term-input-{html.escape(sec_key_id)}' placeholder='e.g., christmas' /></label> "
                f"<button type='button' onclick=\"evoFindTerm('{html.escape(sec_key_id)}')\">Find</button> "
                f"<span class='muted' id='evo-term-out-{html.escape(sec_key_id)}'></span>"
                "</div>"
            )
            term_rows = []
            for t in section.term_first_seen[:250]:
                inj = " <span class='pill'>injected</span>" if t.injected else ""
                term_rows.append(
                    "<tr>"
                    f"<td><code>{html.escape(t.term)}</code>{inj}</td>"
                    f"<td>{html.escape(t.first_seen_kind)}</td>"
                    f"<td>{link(t.first_seen_path, t.first_seen_step_index)}</td>"
                    "</tr>"
                )
            parts.append(
                "<details class='evo-terms' open><summary class='muted'>first seen index (first 250)</summary>"
                "<div class='evo-table-wrap'><table class='evo-table'>"
                "<tr><th align='left'>Term</th><th>Kind</th><th align='left'>First seen</th></tr>"
                + "".join(term_rows)
                + "</table></div></details>"
            )

        if section.node_introduced_terms:
            intro_rows = []
            for path, terms in section.node_introduced_terms.items():
                if not terms:
                    continue
                intro_rows.append(
                    f"<tr><td>{link(path, node_index_by_path.get(path))}</td><td class='muted'>{html.escape(', '.join(terms))}</td></tr>"
                )
            if intro_rows:
                parts.append(
                    "<details class='evo-terms'><summary class='muted'>introduced terms</summary>"
                    "<div class='evo-table-wrap'><table class='evo-table'>"
                    "<tr><th align='left'>Node</th><th align='left'>Terms</th></tr>"
                    + "".join(intro_rows)
                    + "</table></div></details>"
                )

        parts.append("</div></details>")

    parts.append("</section>")
    return "".join(parts), term_index


def render_html(report: RunReport) -> str:
    anchors_seen: set[str] = set()
    anchor_duplicates: set[str] = set()
    for step in report.steps:
        anchor = _step_anchor_id(step)
        if anchor in anchors_seen:
            anchor_duplicates.add(anchor)
        anchors_seen.add(anchor)
    if anchor_duplicates:
        raise ValueError(f"Duplicate step anchor IDs detected: {sorted(anchor_duplicates)}")

    grouped = _group_steps(report)
    tree = _group_steps_tree(report)

    total_runtime_ms = report.runtime_ms
    if total_runtime_ms is None:
        starts = [s.timing.start_ts for s in report.steps if s.timing.start_ts]
        ends = [s.timing.end_ts for s in report.steps if s.timing.end_ts]
        if starts and ends:
            total_runtime_ms = (max(ends) - min(starts)).total_seconds() * 1000

    slowest: List[StepReport] = sorted(
        [s for s in report.steps if s.timing.duration_ms is not None],
        key=lambda s: (s.timing.duration_ms or 0),
        reverse=True,
    )[:10]
    slowest_rows = "".join(
        f"<tr><td><a href='#{html.escape(_step_anchor_id(s))}'>{html.escape(s.path)}</a></td><td>{html.escape(_format_duration_ms(s.timing.duration_ms))}</td></tr>"
        for s in slowest
    ) or "<tr><td colspan='2'>No durations available</td></tr>"

    section_rows = []
    for section, steps in grouped.items():
        durations = [s.timing.duration_ms for s in steps if s.timing.duration_ms is not None]
        total = sum(durations) if durations else None
        section_rows.append(
            f"<tr><td>{html.escape(section)}</td><td>{len(steps)}</td><td>{html.escape(_format_duration_ms(total))}</td></tr>"
        )
    sections_table = "".join(section_rows) or "<tr><td colspan='3'>No steps</td></tr>"

    side_effect_items = []
    for se in report.side_effects:
        try:
            preview_raw = json.dumps(se.data, ensure_ascii=False, sort_keys=True)
        except TypeError:
            preview_raw = str(se.data)
        data_preview = html.escape(_truncate_preview(preview_raw, limit=140))
        side_effect_items.append(
            "<li>"
            f"<span class='pill'>{html.escape(se.type)}</span> "
            f"<span class='muted'>{html.escape(se.timestamp.isoformat())}</span> "
            f"<details class='inline-details'><summary class='muted'>data {data_preview}</summary>"
            f"<pre>{_escape_json(se.data)}</pre><pre>{html.escape(se.raw)}</pre></details>"
            "</li>"
        )
    side_effects_html = "".join(side_effect_items) or "<li>No side effects captured</li>"

    unknown_html = ""
    if report.unknown_events:
        unknown_lines = "\n".join(report.unknown_events[:500])
        truncated_note = (
            f"\n\n... truncated; showing first 500 of {len(report.unknown_events)} lines ..."
            if len(report.unknown_events) > 500
            else ""
        )
        unknown_html = (
            "<details class='unknown'><summary>Unknown oplog lines "
            f"({len(report.unknown_events)})</summary>"
            f"<pre>{html.escape(unknown_lines + truncated_note)}</pre></details>"
        )

    guessed_final = _guess_final_prompt_step(report.steps)
    highlight_html = ""
    if guessed_final:
        highlight_parts = [
            "<div class='highlights'>",
            "<h2>Highlights</h2>",
            "<ul>",
            (
                f"<li>Likely final prompt step: <a href='#{html.escape(_step_anchor_id(guessed_final))}'>"
                f"{html.escape(guessed_final.path)}</a></li>"
            ),
        ]
        if report.metadata.image_path:
            highlight_parts.append(f"<li>Image path: <code>{html.escape(report.metadata.image_path)}</code></li>")
        highlight_parts.append("</ul></div>")
        highlight_html = "".join(highlight_parts)

    banner_html = _render_parser_health_banner(report)

    tree_html_bits = []
    for group2, sub in tree.items():
        tree_html_bits.append(f"<details open><summary>{html.escape(group2)}</summary><div class='tree'>")
        for group3, steps in sub.items():
            if group3:
                tree_html_bits.append(f"<details><summary>{html.escape(group3)}</summary><ul>")
            else:
                tree_html_bits.append("<ul>")
            for step in steps:
                tree_html_bits.append(
                    f"<li><a href='#{html.escape(_step_anchor_id(step))}' title='{html.escape(step.path)}'>{html.escape(step.name)}</a></li>"
                )
            tree_html_bits.append("</ul>" + ("</details>" if group3 else ""))
        tree_html_bits.append("</div></details>")
    tree_html = "".join(tree_html_bits) or "<p class='muted'>No steps</p>"

    evolution_html, evolution_term_index = _render_evolution(report)
    evolution_term_json = json.dumps(evolution_term_index, ensure_ascii=False, sort_keys=True).replace(
        "</", "<\\/"
    )
    evo_script = (
        "<script>\n"
        f"window.__EVOLUTION_TERM_INDEX__ = {evolution_term_json};\n"
        "function evoFindTerm(sectionId) {\n"
        "  try {\n"
        "    var input = document.getElementById('evo-term-input-' + sectionId);\n"
        "    var out = document.getElementById('evo-term-out-' + sectionId);\n"
        "    if (!input || !out) return;\n"
        "    var term = (input.value || '').toLowerCase().trim();\n"
        "    if (!term) { out.textContent = 'Enter a term.'; return; }\n"
        "    var sec = (window.__EVOLUTION_TERM_INDEX__ || {})[sectionId] || {};\n"
        "    var hit = sec[term];\n"
        "    if (!hit) { out.textContent = 'Not found in index.'; return; }\n"
        "    var injected = hit.injected ? ' injected' : '';\n"
        "    while (out.firstChild) { out.removeChild(out.firstChild); }\n"
        "    out.appendChild(document.createTextNode('First seen: '));\n"
        "    var a = document.createElement('a');\n"
        "    a.setAttribute('href', '#' + hit.anchor);\n"
        "    a.textContent = hit.path;\n"
        "    out.appendChild(a);\n"
        "    out.appendChild(document.createTextNode(' (' + hit.kind + injected + ')'));\n"
        "  } catch (e) { }\n"
        "}\n"
        "</script>\n"
    )

    html_doc = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        :root {{
          --bg: #ffffff;
          --fg: #111827;
          --muted: #6b7280;
          --border: #e5e7eb;
          --panel: #f9fafb;
          --codebg: #0b1220;
          --codefg: #e5e7eb;
          --warn: #f59e0b;
          --error: #ef4444;
          --info: #3b82f6;
        }}
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; color: var(--fg); background: var(--bg); }}
        header {{ padding: 1rem 1.25rem; border-bottom: 1px solid var(--border); background: var(--panel); }}
        h1 {{ margin: 0; font-size: 1.25rem; }}
        .layout {{ display: grid; grid-template-columns: 320px 1fr; gap: 1rem; padding: 1rem 1.25rem; }}
        nav {{ position: sticky; top: 1rem; align-self: start; max-height: calc(100vh - 2rem); overflow: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); padding: 0.75rem; }}
        main {{ min-width: 0; }}
        a {{ color: #2563eb; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        table.meta {{ border-collapse: collapse; width: 100%; max-width: 920px; }}
        table.meta th {{ text-align: left; color: var(--muted); padding: 0.3rem 0.4rem; width: 180px; vertical-align: top; }}
        table.meta td {{ padding: 0.3rem 0.4rem; }}
        .muted {{ color: var(--muted); }}
        code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; }}
        pre {{ background: var(--codebg); color: var(--codefg); padding: 0.75rem; border-radius: 10px; overflow: auto; white-space: pre-wrap; }}
        details > summary {{ cursor: pointer; }}
        .pill {{ display: inline-block; padding: 0.1rem 0.45rem; border: 1px solid var(--border); border-radius: 999px; background: #fff; font-size: 0.8rem; color: var(--muted); margin-left: 0.35rem; }}
        .sev {{ font-weight: 700; padding: 0.05rem 0.35rem; border-radius: 6px; border: 1px solid var(--border); background: #fff; }}
        .sev-error {{ border-color: rgba(239,68,68,.35); color: var(--error); }}
        .sev-warn {{ border-color: rgba(245,158,11,.35); color: var(--warn); }}
        .sev-info {{ border-color: rgba(59,130,246,.35); color: var(--info); }}
        .issues {{ padding-left: 1.25rem; }}
        .step-card {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.35rem 0.6rem; margin: 0.6rem 0; background: #fff; }}
        .step-card summary {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }}
        .step-name {{ font-weight: 700; }}
        .step-body {{ padding: 0.6rem 0.2rem 0.2rem 0.2rem; }}
        .step-meta {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
        .step-block {{ margin: 0.6rem 0 0.75rem 0; }}
        .text-block {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.35rem 0.6rem; background: var(--panel); margin: 0.5rem 0; }}
        .inline-details summary {{ display: inline; }}
        .tree details {{ margin: 0.25rem 0; }}
        .tree ul {{ margin: 0.25rem 0 0.5rem 1rem; padding: 0; }}
        .tree li {{ list-style: none; margin: 0.15rem 0; }}
        .unknown {{ margin-top: 1rem; }}
        .highlights {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 1rem; background: var(--panel); max-width: 920px; }}
        .banner {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 1rem; }}
        .banner-warn {{ border-color: rgba(245,158,11,.35); background: rgba(245,158,11,.08); }}
        .banner-error {{ border-color: rgba(239,68,68,.35); background: rgba(239,68,68,.08); }}
        .evo-toc-list {{ padding-left: 1.25rem; }}
        .evo-section {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.35rem 0.6rem; margin: 0.6rem 0; background: #fff; }}
        .evo-body {{ padding: 0.6rem 0.2rem 0.2rem 0.2rem; }}
        .evo-graph div {{ margin: 0.15rem 0; }}
        .evo-table-wrap {{ overflow: auto; border: 1px solid var(--border); border-radius: 10px; background: #fff; max-width: 920px; }}
        .evo-table {{ border-collapse: collapse; width: 100%; }}
        .evo-table th, .evo-table td {{ padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
        .evo-table th {{ text-align: left; color: var(--muted); }}
        .evo-table td {{ font-size: 0.92rem; }}
        .evo-matrix td {{ text-align: center; }}
        .evo-diag {{ color: var(--muted); }}
        .evo-findings {{ padding-left: 1.25rem; }}
        .evo-term-search {{ margin: 0.4rem 0 0.75rem 0; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }}
        .evo-term-search input {{ padding: 0.2rem 0.35rem; border: 1px solid var(--border); border-radius: 6px; }}
        .evo-term-search button {{ padding: 0.25rem 0.5rem; border: 1px solid var(--border); border-radius: 6px; background: var(--panel); cursor: pointer; }}
        .evo-inline summary {{ display: inline; }}
        .evo-suggestions {{ padding-left: 1.25rem; }}
      </style>
      <title>run_review - {html.escape(report.metadata.generation_id)}</title>
    </head>
    <body>
      <header>
        <h1>run_review: {html.escape(report.metadata.generation_id)}</h1>
        <div class="muted">Offline run analyzer report</div>
      </header>

      <div class="layout">
        <nav>
          <div><strong>Jump to</strong></div>
          <ul>
            <li><a href="#summary">Summary</a></li>
            <li><a href="#issues">Issues</a></li>
            <li><a href="#timeline">Timeline</a></li>
            <li><a href="#evolution">Evolution</a></li>
            <li><a href="#steps">Steps</a></li>
            <li><a href="#side-effects">Side effects</a></li>
          </ul>
          <div style="margin-top: 0.75rem;"><strong>Pipeline tree</strong></div>
          {tree_html}
        </nav>

        <main>
          {banner_html}
          <section id="summary">
            <h2>Summary</h2>
            <table class="meta">{_format_meta_table(report)}</table>
            <p class="muted">Steps: {len(report.steps)} | Side effects: {len(report.side_effects)} | Unknown lines: {len(report.unknown_events)}</p>
            {highlight_html}
          </section>

          <section id="issues">
            <h2>Issues</h2>
            {_render_issues(report)}
          </section>

          <section id="timeline">
            <h2>Timeline</h2>
            <p><strong>Total runtime</strong>: {html.escape(_format_duration_ms(total_runtime_ms))}</p>
            <h3>Slowest steps</h3>
            <table>
              <tr><th align="left">Path</th><th align="left">Duration</th></tr>
              {slowest_rows}
            </table>
            <h3>Counts by section</h3>
            <table>
              <tr><th align="left">Section</th><th align="left">Steps</th><th align="left">Total duration</th></tr>
              {sections_table}
            </table>
          </section>

          {evolution_html}

          <section id="steps">
            <h2>Steps</h2>
            {''.join(f"<h3>{html.escape(group)}</h3>{''.join(_step_card(s) for s in steps)}" for group, steps in grouped.items())}
          </section>

          <section id="side-effects">
            <h2>Side effects</h2>
            <ul>{side_effects_html}</ul>
          </section>

          {unknown_html}
        </main>
      </div>
      {evo_script}
    </body>
    </html>
    """
    return html_doc


def render_compare_html(diff: CompareResult) -> str:
    added = "".join(f"<li>{html.escape(path)}</li>" for path in diff.added_steps) or "<li>None</li>"
    removed = "".join(f"<li>{html.escape(path)}</li>" for path in diff.removed_steps) or "<li>None</li>"
    metadata_changes = "".join(
        f"<li>{html.escape(key)} changed: {html.escape(str(val['run_a']))} -> {html.escape(str(val['run_b']))}</li>"
        for key, val in diff.metadata_changes.items()
    ) or "<li>No metadata presence changes</li>"
    injector_block = "".join(f"<li>{html.escape(msg)}</li>" for msg in diff.injector_diffs) or "<li>No injector differences</li>"
    post_block = "".join(f"<li>{html.escape(msg)}</li>" for msg in diff.post_processing_diffs) or "<li>No post-processing differences</li>"
    return f"""
    <!doctype html>
    <html>
    <head><meta charset="utf-8" /><title>run_review - compare</title></head>
    <body>
      <h1>run_review: compare</h1>
      <h2>Added steps</h2><ul>{added}</ul>
      <h2>Removed steps</h2><ul>{removed}</ul>
      <h2>Metadata changes</h2><ul>{metadata_changes}</ul>
      <h2>Injector differences</h2><ul>{injector_block}</ul>
      <h2>Post-processing differences</h2><ul>{post_block}</ul>
    </body>
    </html>
    """
