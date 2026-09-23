from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_number(value: float | None, *, decimals: int = 3) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


@dataclass(frozen=True)
class JudgeScores:
    judge_id: str
    rubric: str | None
    weight: float | None
    scores: dict[str, float]
    winner: str | None


@dataclass(frozen=True)
class IterationRow:
    iteration: int
    selection_mode: str | None
    selected_ids: tuple[str, ...]
    scores: dict[str, float]
    winner: str | None
    runner_up: str | None
    margin: float | None
    best_generated_vs_a_delta: float | None
    judge_scores: tuple[JudgeScores, ...]


@dataclass(frozen=True)
class RunScores:
    generation_id: str
    seed: int | None
    transcript_path: Path
    iterations: tuple[IterationRow, ...]

    @property
    def has_prompt_refine(self) -> bool:
        return bool(self.iterations)


def _stable_id_sort(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(items, key=lambda item: (-float(item[1]), str(item[0])))


def _resolve_relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _extract_iteration_row(raw: Mapping[str, Any]) -> IterationRow:
    iteration_raw = raw.get("iteration")
    iteration = int(iteration_raw) if _as_float(iteration_raw) is not None else 0

    selection = raw.get("selection") if isinstance(raw.get("selection"), dict) else {}
    selection_mode = selection.get("selection_mode")
    selection_mode = str(selection_mode) if isinstance(selection_mode, str) and selection_mode.strip() else None

    selected_ids_raw = selection.get("selected_ids")
    if isinstance(selected_ids_raw, list):
        selected_ids = tuple(str(v).strip() for v in selected_ids_raw if str(v).strip())
    elif isinstance(selected_ids_raw, str) and selected_ids_raw.strip():
        selected_ids = (selected_ids_raw.strip(),)
    else:
        selected_ids = ()

    judge_scores_raw = raw.get("judge_outputs") if isinstance(raw.get("judge_outputs"), list) else None
    judge_scores: list[JudgeScores] = []
    if isinstance(judge_scores_raw, list):
        for entry in judge_scores_raw:
            if not isinstance(entry, dict):
                continue
            judge_id = str(entry.get("judge_id") or "").strip()
            if not judge_id:
                continue
            rubric_raw = entry.get("rubric")
            rubric = str(rubric_raw).strip() if isinstance(rubric_raw, str) and rubric_raw.strip() else None
            weight = _as_float(entry.get("weight"))

            scores_in = entry.get("parsed_scores")
            scores_by_id: dict[str, float] = {}
            if isinstance(scores_in, dict):
                for cid, score_raw in scores_in.items():
                    score = _as_float(score_raw)
                    if score is None:
                        continue
                    key = str(cid).strip()
                    if not key:
                        continue
                    scores_by_id[key] = float(score)

            winner_id: str | None = None
            if scores_by_id:
                ranked = _stable_id_sort(list(scores_by_id.items()))
                winner_id = ranked[0][0] if ranked else None

            judge_scores.append(
                JudgeScores(
                    judge_id=judge_id,
                    rubric=rubric,
                    weight=weight,
                    scores=scores_by_id,
                    winner=winner_id,
                )
            )
        judge_scores.sort(key=lambda j: j.judge_id)

    scores: dict[str, float] = {}
    aggregate = raw.get("aggregate") if isinstance(raw.get("aggregate"), dict) else {}
    agg_scores = aggregate.get("scores") if isinstance(aggregate.get("scores"), dict) else None
    if isinstance(agg_scores, dict):
        for cid, score_raw in agg_scores.items():
            score = _as_float(score_raw)
            if score is None:
                continue
            key = str(cid).strip()
            if not key:
                continue
            scores[key] = float(score)

    if not scores:
        score_table = selection.get("score_table") if isinstance(selection.get("score_table"), list) else None
        if isinstance(score_table, list):
            for row in score_table:
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("id") or "").strip()
                score = _as_float(row.get("score"))
                if cid and score is not None:
                    scores[cid] = float(score)

    winner: str | None = None
    runner_up: str | None = None
    margin: float | None = None
    if scores:
        ranked = _stable_id_sort(list(scores.items()))
        winner = ranked[0][0]
        if len(ranked) >= 2:
            runner_up = ranked[1][0]
            margin = float(ranked[0][1] - ranked[1][1])
        else:
            margin = 0.0

    kind_by_id: dict[str, str] = {}
    candidates = raw.get("candidates") if isinstance(raw.get("candidates"), list) else None
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            cid = str(candidate.get("id") or "").strip()
            kind = str(candidate.get("kind") or "").strip().lower()
            if cid and kind:
                kind_by_id[cid] = kind

    score_a = scores.get("A")
    generated_scores = [
        scores[cid]
        for cid, kind in kind_by_id.items()
        if kind == "generated" and cid in scores
    ]
    best_generated_vs_a_delta: float | None
    if score_a is None or not generated_scores:
        best_generated_vs_a_delta = None
    else:
        best_generated_vs_a_delta = float(max(generated_scores) - float(score_a))

    return IterationRow(
        iteration=iteration,
        selection_mode=selection_mode,
        selected_ids=selected_ids,
        scores=scores,
        winner=winner,
        runner_up=runner_up,
        margin=margin,
        best_generated_vs_a_delta=best_generated_vs_a_delta,
        judge_scores=tuple(judge_scores),
    )


def _extract_run_scores(transcript_path: Path) -> RunScores:
    payload = _load_json(transcript_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Transcript must be an object: {transcript_path}")

    generation_id = payload.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id.strip():
        generation_id = transcript_path.name.replace("_transcript.json", "").strip()

    seed_raw = payload.get("seed")
    seed = int(seed_raw) if _as_float(seed_raw) is not None else None

    pr = None
    blackbox_scoring = payload.get("blackbox_scoring")
    if isinstance(blackbox_scoring, dict):
        pr = blackbox_scoring.get("prompt_refine")
    iterations_raw = pr.get("iterations") if isinstance(pr, dict) else None

    iterations: list[IterationRow] = []
    if isinstance(iterations_raw, list) and iterations_raw:
        for raw in iterations_raw:
            if not isinstance(raw, dict):
                continue
            iterations.append(_extract_iteration_row(raw))

        # Make sure the iteration index is always present and monotonically increasing.
        fixed: list[IterationRow] = []
        for idx, row in enumerate(iterations, start=1):
            it = row.iteration if row.iteration > 0 else idx
            fixed.append(
                IterationRow(
                    iteration=it,
                    selection_mode=row.selection_mode,
                    selected_ids=row.selected_ids,
                    scores=row.scores,
                    winner=row.winner,
                    runner_up=row.runner_up,
                    margin=row.margin,
                    best_generated_vs_a_delta=row.best_generated_vs_a_delta,
                    judge_scores=row.judge_scores,
                )
            )
        iterations = fixed

    return RunScores(
        generation_id=generation_id.strip(),
        seed=seed,
        transcript_path=transcript_path,
        iterations=tuple(iterations),
    )


def _count_ordered(counter: dict[str, int], key: str, *, inc: int = 1) -> None:
    if key not in counter:
        counter[key] = 0
    counter[key] += int(inc)


def render_scores_by_iteration_md(*, experiment_root: Path) -> str:
    logs_dir = experiment_root / "logs"
    transcript_paths = sorted(logs_dir.glob("*_transcript.json"))
    if not transcript_paths:
        raise FileNotFoundError(f"No *_transcript.json files found under {logs_dir}")

    runs = sorted((_extract_run_scores(path) for path in transcript_paths), key=lambda r: r.generation_id)

    refine_runs = [run for run in runs if run.has_prompt_refine]
    all_iterations = [row for run in refine_runs for row in run.iterations]

    judge_meta: dict[str, dict[str, Any]] = {}
    for row in all_iterations:
        for judge in row.judge_scores:
            if judge.judge_id not in judge_meta:
                judge_meta[judge.judge_id] = {
                    "rubric": judge.rubric,
                    "weight": judge.weight,
                }

    judge_ids = sorted(judge_meta.keys())
    is_multi_judge = len(judge_ids) > 1

    selection_modes: dict[str, int] = {}
    selected_ids: dict[str, int] = {}
    margins: list[float] = []

    judge_winner_counts: dict[str, dict[str, int]] = {jid: {} for jid in judge_ids}
    judged_iterations = 0
    agreed_iterations = 0

    for row in all_iterations:
        if row.selection_mode:
            _count_ordered(selection_modes, row.selection_mode)
        for cid in row.selected_ids:
            _count_ordered(selected_ids, cid)
        if row.margin is not None:
            margins.append(float(row.margin))
        if judge_ids and row.judge_scores:
            by_id = {j.judge_id: j for j in row.judge_scores}
            winners: list[str] = []
            for jid in judge_ids:
                winner = by_id.get(jid).winner if jid in by_id else None
                if winner:
                    _count_ordered(judge_winner_counts[jid], winner)
                    winners.append(winner)
            if winners and len(winners) == len(judge_ids):
                judged_iterations += 1
                if len(set(winners)) == 1:
                    agreed_iterations += 1

    unique_iteration_counts = sorted({len(run.iterations) for run in refine_runs})

    margin_min = min(margins) if margins else None
    margin_max = max(margins) if margins else None
    margin_median = statistics.median(margins) if margins else None
    margin_mean = statistics.mean(margins) if margins else None

    lines: list[str] = []
    lines.append(f"# Scores by Iteration - {experiment_root.name}")
    lines.append("")
    lines.append(f"Source: `{_resolve_relpath(logs_dir)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Transcripts: {len(runs)} total; {len(refine_runs)} with `blackbox_scoring.prompt_refine`."
    )
    lines.append(f"- Decisions: {len(all_iterations)} (iterations per run: {unique_iteration_counts}).")
    if judge_ids:
        judge_desc = ", ".join(
            f"{jid}({judge_meta[jid].get('rubric') or 'n/a'})" for jid in judge_ids
        )
        lines.append(f"- Judges: {judge_desc}")
        lines.append(f"- Judge winner counts: {judge_winner_counts}")
        if judged_iterations:
            pct = 100.0 * float(agreed_iterations) / float(judged_iterations)
            lines.append(
                f"- Judge agreement (winner): {agreed_iterations}/{judged_iterations} ({pct:.1f}%)"
            )
    lines.append(f"- Selection modes: {selection_modes}")
    lines.append(f"- Selected ids: {selected_ids}")
    if margins:
        lines.append(
            "- Winner margin (top1 - top2): "
            f"min={_format_number(_as_float(margin_min))} "
            f"median={_format_number(_as_float(margin_median))} "
            f"mean={_format_number(_as_float(margin_mean), decimals=3)} "
            f"max={_format_number(_as_float(margin_max))}"
        )
    else:
        lines.append("- Winner margin (top1 - top2): <none>")

    lines.append("")
    lines.append("## Best Generated vs A (Per Iteration)")
    lines.append("")
    lines.append(
        "Delta = `max(score(generated candidates)) - score(A)`. Negative means A beat the best generated candidate."
    )
    lines.append("")
    lines.append("| run | deltas (per iter) | mean |")
    lines.append("|---|---|---:|")
    for run in refine_runs:
        deltas = [row.best_generated_vs_a_delta for row in run.iterations]
        present = [float(v) for v in deltas if v is not None]
        mean_delta = _mean(present)
        deltas_text = ", ".join(_format_number(_as_float(v)) for v in deltas)
        lines.append(
            f"| {run.generation_id} | {deltas_text} | {_format_number(_as_float(mean_delta), decimals=3)} |"
        )

    for run in runs:
        lines.append("")
        lines.append(f"## {run.generation_id}")
        lines.append("")
        lines.append(f"- Transcript: `{_resolve_relpath(run.transcript_path)}`")
        if run.seed is not None:
            lines.append(f"- Seed: `{run.seed}`")
        else:
            lines.append("- Seed: <none>")

        if not run.iterations:
            lines.append("- Prompt refine iterations: <none>")
            continue

        candidate_ids: list[str] = sorted({cid for row in run.iterations for cid in row.scores.keys()})

        lines.append("")
        if judge_ids:
            judge_desc = ", ".join(
                f"{jid}({judge_meta[jid].get('rubric') or 'n/a'})" for jid in judge_ids
            )
            lines.append(f"- Judges (vote order): {judge_desc}")
            lines.append("")

        header = ["iter", "mode", "selected"]
        if is_multi_judge:
            header.append("votes")
        if judge_ids:
            header.extend(judge_ids)
        header.extend([*candidate_ids, "winner", "runner_up", "margin"])
        lines.append("| " + " | ".join(header) + " |")
        align = ["---:|", "---|", "---|"]
        if is_multi_judge:
            align.append("---|")
        if judge_ids:
            align.extend(["---|" for _ in judge_ids])
        align.extend([*["---:|" for _ in candidate_ids], "---|", "---|", "---:|"])
        lines.append("|" + "".join(align))

        for row in run.iterations:
            selected = ", ".join(row.selected_ids) if row.selected_ids else ""
            mode = row.selection_mode or ""
            votes = ""
            by_id = {j.judge_id: j for j in row.judge_scores} if row.judge_scores else {}
            if is_multi_judge and row.judge_scores:
                vote_items = [(by_id.get(jid).winner if jid in by_id else None) for jid in judge_ids]
                votes = "/".join((v or "") for v in vote_items)
            cells = [
                str(int(row.iteration)),
                mode,
                selected,
            ]
            if is_multi_judge:
                cells.append(votes)
            if judge_ids:
                for jid in judge_ids:
                    judge = by_id.get(jid)
                    if judge is None:
                        cells.append("")
                        continue
                    parts: list[str] = []
                    for cid in candidate_ids:
                        value_text = _format_number(_as_float(judge.scores.get(cid)))
                        parts.append(f"{cid}={value_text if value_text else '?'}")
                    cells.append(" ".join(parts))
            cells.extend(
                [
                    *[_format_number(_as_float(row.scores.get(cid))) for cid in candidate_ids],
                    row.winner or "",
                    row.runner_up or "",
                    _format_number(_as_float(row.margin)),
                ]
            )
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="write_scores_by_iteration",
        description="Generate a scores_by_iteration.md summary from experiment transcript logs.",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        required=True,
        help="Experiment root directory containing a logs/ folder with *_transcript.json files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output markdown path (defaults to <experiment-root>/scores_by_iteration.md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else experiment_root / "scores_by_iteration.md"
    )

    markdown = render_scores_by_iteration_md(experiment_root=experiment_root)

    os.makedirs(output_path.parent, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
