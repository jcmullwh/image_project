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
        elif re.search(r"/consensus(_\\d+)?$", path):
            base += 3
        elif "consensus" in path:
            base += 1
        ts = step.timing.start_ts or datetime.min
        return (base, ts, step.path)

    return max(candidates, key=score)


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
    for steps in groups.values():
        steps.sort(key=lambda s: (s.timing.start_ts or datetime.min, s.path))
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def _group_steps_tree(report: RunReport) -> Dict[str, Dict[str, List[StepReport]]]:
    tree: Dict[str, Dict[str, List[StepReport]]] = {}
    for step in report.steps:
        parts = step.path.split("/")
        group2 = "/".join(parts[:2]) if len(parts) >= 2 else step.path
        group3 = parts[2] if len(parts) >= 3 else ""
        tree.setdefault(group2, {}).setdefault(group3, []).append(step)
    for sub in tree.values():
        for steps in sub.values():
            steps.sort(key=lambda s: (s.timing.start_ts or datetime.min, s.path))
    return dict(sorted(tree.items(), key=lambda kv: kv[0]))


def _step_anchor_id(step: StepReport) -> str:
    return f"step-{_safe_id(step.path)}"


def _truncate_preview(text: str, *, limit: int = 180) -> str:
    stripped = text.replace("\r\n", "\n").strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"


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
        elif re.search(r"/consensus(_\\d+)?$", path):
            base += 3
        elif "consensus" in path:
            base += 1
        ts = step.timing.start_ts or datetime.min
        return (base, ts, step.path)

    return max(candidates, key=score)


def render_html(report: RunReport) -> str:
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

    tree_html_bits = []
    for group2, sub in tree.items():
        tree_html_bits.append(f"<details open><summary>{html.escape(group2)}</summary><div class='tree'>")
        for group3, steps in sorted(sub.items(), key=lambda kv: kv[0] or ""):
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
            <li><a href="#steps">Steps</a></li>
            <li><a href="#side-effects">Side effects</a></li>
          </ul>
          <div style="margin-top: 0.75rem;"><strong>Pipeline tree</strong></div>
          {tree_html}
        </nav>

        <main>
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
