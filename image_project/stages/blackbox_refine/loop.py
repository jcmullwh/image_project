"""Blackbox refinement loop utilities.

Prompt policy (generator/judge templates + parsing helpers) lives under
`image_project/prompts/blackbox_refine.py`; this module implements stage logic.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from image_project.framework import scoring as blackbox_scoring
from image_project.framework.scoring import NoveltyConfig
from pipelinekit.config_namespace import ConfigNamespace
from pipelinekit.engine.pipeline import ActionStep, Block, ChatStep
from image_project.framework.prompt_pipeline import PlanInputs, make_action_stage_block
from image_project.framework.runtime import RunContext
from image_project.prompts.blackbox_refine import (
    profile_representation_from_guidance,
    prompt_variants_judge_prompt,
    variation_generate_prompt,
)
from pipelinekit.stage_types import StageIO, StageRef


CandidateKind = Literal["generated", "parent"]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_ordinal: int
    parent_beam_index: int
    kind: CandidateKind
    stage_id: str | None
    output_key: str | None


@dataclass(frozen=True)
class PromptBlackboxRefineVariationPromptConfig:
    template: str
    include_concepts: bool
    include_context_guidance: bool
    include_profile: bool
    profile_source: str
    include_novelty_summary: bool
    include_mutation_directive: bool
    include_scoring_rubric: bool
    score_feedback: str
    score_feedback_max_chars: int


@dataclass(frozen=True)
class PromptBlackboxRefineMutationDirectivesConfig:
    mode: str
    directives: tuple[str, ...]


@dataclass(frozen=True)
class PromptBlackboxRefineJudgeConfig:
    id: str
    rubric: str
    weight: float
    model: str | None
    temperature: float | None


@dataclass(frozen=True)
class PromptBlackboxRefineJudgingConfig:
    judges: tuple[PromptBlackboxRefineJudgeConfig, ...]
    aggregation: str
    trimmed_mean_drop: int


@dataclass(frozen=True)
class PromptBlackboxRefineSelectionConfig:
    exploration_rate_override: float | None
    group_by_beam: bool
    tie_breaker: str


@dataclass(frozen=True)
class PromptBlackboxRefineConfig:
    algorithm: str
    iterations: int
    beam_width: int
    branching_factor: int
    include_parents_as_candidates: bool

    generator_model: str | None
    generator_temperature: float
    max_prompt_chars: int | None

    variation_prompt: PromptBlackboxRefineVariationPromptConfig
    mutation_directives: PromptBlackboxRefineMutationDirectivesConfig
    judging: PromptBlackboxRefineJudgingConfig
    selection: PromptBlackboxRefineSelectionConfig

    # Stage-owned scoring defaults (formerly under prompt.scoring.*).
    judge_model: str | None
    judge_temperature: float
    judge_profile_source: str
    exploration_rate: float
    novelty: NoveltyConfig


def pick_mutation_directive(
    *,
    mode: str,
    directives: tuple[str, ...],
    seed: int,
    iteration: int,
    candidate_id: str,
    candidate_ordinal: int,
) -> str | None:
    key = (mode or "").strip().lower()
    if key == "none":
        return None
    if not directives:
        return None
    if key == "fixed":
        return directives[0]
    if key == "cycle":
        idx = (int(iteration) - 1 + int(candidate_ordinal)) % len(directives)
        return directives[idx]
    if key == "random":
        raw = f"{int(seed)}|{int(iteration)}|{str(candidate_id).strip()}".encode("utf-8")
        digest = hashlib.sha256(raw).digest()
        idx = int.from_bytes(digest[:8], "big") % len(directives)
        return directives[idx]
    raise ValueError(f"Unknown mutation directive mode: {mode!r}")


def _beams_count_for_iteration(cfg: PromptBlackboxRefineConfig, iteration: int) -> int:
    if cfg.algorithm == "beam" and iteration > 1:
        return int(cfg.beam_width)
    return 1


def _candidate_specs_for_iteration(
    cfg: PromptBlackboxRefineConfig,
    iteration: int,
    *,
    beams_count: int,
) -> list[CandidateSpec]:
    include_parent = bool(cfg.include_parents_as_candidates)
    total_candidates = beams_count * (int(cfg.branching_factor) + (1 if include_parent else 0))
    ids = blackbox_scoring.expected_idea_ids(total_candidates)

    specs: list[CandidateSpec] = []
    ordinal = 0
    idx = 0
    for beam_index in range(1, beams_count + 1):
        if include_parent:
            candidate_id = ids[idx]
            idx += 1
            specs.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    candidate_ordinal=ordinal,
                    parent_beam_index=beam_index,
                    kind="parent",
                    stage_id=None,
                    output_key=None,
                )
            )
            ordinal += 1

        for _ in range(int(cfg.branching_factor)):
            candidate_id = ids[idx]
            idx += 1
            stage_id = f"blackbox_refine.iter_{iteration:02d}.beam_{beam_index:02d}.cand_{candidate_id}"
            output_key = f"bbref.iter_{iteration:02d}.beam_{beam_index:02d}.cand_{candidate_id}.prompt"
            specs.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    candidate_ordinal=ordinal,
                    parent_beam_index=beam_index,
                    kind="generated",
                    stage_id=stage_id,
                    output_key=output_key,
                )
            )
            ordinal += 1

    return specs


def _iteration_stage_block(
    inputs: PlanInputs,
    cfg: PromptBlackboxRefineConfig,
    iteration: int,
    *,
    config_path_root: str,
) -> Block:
    beams_count = _beams_count_for_iteration(cfg, iteration)
    candidates = _candidate_specs_for_iteration(cfg, iteration, beams_count=beams_count)

    nodes: list[Any] = []

    for beam_index in range(1, beams_count + 1):
        beam_nodes: list[Any] = []
        for cand in candidates:
            if cand.kind != "generated":
                continue
            if int(cand.parent_beam_index) != int(beam_index):
                continue
            beam_nodes.append(
                _candidate_generation_step(cfg, iteration, cand, config_path_root=config_path_root)
            )
        nodes.append(
            Block(
                name=f"beam_{beam_index:02d}",
                merge="none",
                nodes=beam_nodes,
            )
        )

    judge_steps: list[Any] = []
    for judge in cfg.judging.judges:
        judge_steps.append(
            _judge_scores_step(
                inputs, cfg, iteration, judge, candidates, config_path_root=config_path_root
            )
        )
    nodes.append(Block(name="judge", merge="none", nodes=judge_steps))

    nodes.append(
        _select_action_step(inputs, cfg, iteration, candidates, config_path_root=config_path_root)
    )

    return Block(
        name=f"blackbox_refine.iter_{iteration:02d}",
        merge="none",
        nodes=nodes,
        meta={
            "doc": "Blackbox refine iteration (generate, judge, select).",
            "tags": ["blackbox_refine"],
            "iteration": int(iteration),
        },
    )


def _init_state_block(
    inputs: PlanInputs,
    cfg: PromptBlackboxRefineConfig,
    *,
    seed_output_key: str,
    seed_source: str | None,
) -> Block:
    def _action(ctx: RunContext) -> dict[str, Any]:
        seed_prompt = ctx.outputs.get(seed_output_key)
        if not isinstance(seed_prompt, str) or not seed_prompt.strip():
            raise ValueError(f"Missing required output: {seed_output_key}")
        seed_prompt = seed_prompt.strip()
        ctx.outputs[seed_output_key] = seed_prompt

        ctx.outputs["bbref.beams"] = [
            {
                "beam_index": 1,
                "prompt": seed_prompt,
                "source": seed_source or "seed",
            }
        ]

        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}

        prompt_refine = ctx.blackbox_scoring.get("prompt_refine")
        if not isinstance(prompt_refine, dict):
            prompt_refine = {}
            ctx.blackbox_scoring["prompt_refine"] = prompt_refine

        prompt_refine["config_snapshot"] = asdict(cfg)
        prompt_refine["seed_output_key"] = seed_output_key
        prompt_refine["seed_source"] = seed_source or "seed"
        prompt_refine["seed_prompt"] = seed_prompt
        prompt_refine["iterations"] = []

        novelty_enabled = bool(cfg.novelty.enabled and cfg.novelty.window > 0)

        ctx.logger.info(
            "Blackbox prompt refine init: algorithm=%s iterations=%d beam_width=%d branching_factor=%d judges=%d aggregation=%s novelty=%s seed_source=%s",
            cfg.algorithm,
            cfg.iterations,
            cfg.beam_width,
            cfg.branching_factor,
            len(cfg.judging.judges),
            cfg.judging.aggregation,
            novelty_enabled,
            seed_source or "seed",
        )

        return {
            "seed_source": seed_source or "seed",
            "seed_output_key": seed_output_key,
        }

    return make_action_stage_block(
        "blackbox_refine.init_state",
        fn=_action,
        merge="none",
        tags=("blackbox_refine",),
        doc="Initialize blackbox refine loop state.",
        source="stages.blackbox_refine.loop._init_state_block",
    )


def _candidate_generation_step(
    cfg: PromptBlackboxRefineConfig,
    iteration: int,
    candidate: CandidateSpec,
    *,
    config_path_root: str,
) -> ChatStep:
    assert candidate.kind == "generated"
    assert candidate.output_key is not None

    generator_params: dict[str, Any] = {}
    if cfg.generator_model:
        generator_params["model"] = cfg.generator_model

    def _prompt(ctx: RunContext, *, candidate=candidate, iteration=iteration) -> str:
        base_prompt = _resolve_beam_prompt(
            ctx,
            beam_index=candidate.parent_beam_index,
            stage_id=candidate.stage_id or "<unknown>",
        )

        score_feedback: Mapping[str, Any] | None = None
        if cfg.variation_prompt.score_feedback == "best_worst" and int(iteration) > 1:
            score_feedback = _resolve_best_worst_score_feedback(
                ctx,
                iteration=int(iteration) - 1,
                beam_index=int(candidate.parent_beam_index),
                stage_id=candidate.stage_id or "<unknown>",
            )

        profile_text: str | None = None
        if cfg.variation_prompt.include_profile:
            profile_text = _resolve_profile_representation(
                ctx,
                profile_source=cfg.variation_prompt.profile_source,
                stage_id=candidate.stage_id or "<unknown>",
                config_path=f"{config_path_root}.variation_prompt.profile_source",
            )

        novelty_summary: Mapping[str, Any] | None = None
        if cfg.variation_prompt.include_novelty_summary:
            novelty_summary = _resolve_novelty_summary(ctx)

        mutation_directive: str | None = None
        if cfg.variation_prompt.include_mutation_directive:
            mutation_directive = pick_mutation_directive(
                mode=cfg.mutation_directives.mode,
                directives=cfg.mutation_directives.directives,
                seed=int(ctx.seed),
                iteration=int(iteration),
                candidate_id=candidate.candidate_id,
                candidate_ordinal=candidate.candidate_ordinal,
            )

        return variation_generate_prompt(
            template=cfg.variation_prompt.template,
            base_prompt=base_prompt,
            concepts=list(ctx.selected_concepts),
            context_guidance=str(ctx.outputs.get("context_guidance") or ""),
            profile_text=profile_text,
            novelty_summary=novelty_summary,
            mutation_directive=mutation_directive,
            score_feedback=score_feedback,
            include_concepts=cfg.variation_prompt.include_concepts,
            include_context_guidance=cfg.variation_prompt.include_context_guidance,
            include_profile=cfg.variation_prompt.include_profile,
            include_novelty_summary=cfg.variation_prompt.include_novelty_summary,
            include_mutation_directive=cfg.variation_prompt.include_mutation_directive,
            include_scoring_rubric=cfg.variation_prompt.include_scoring_rubric,
            score_feedback_max_chars=int(cfg.variation_prompt.score_feedback_max_chars),
            max_prompt_chars=cfg.max_prompt_chars,
        )

    return ChatStep(
        name=f"cand_{candidate.candidate_id}",
        prompt=_prompt,
        temperature=float(cfg.generator_temperature),
        merge="none",
        capture_key=candidate.output_key,
        params=generator_params,
        meta={
            "doc": "Generate a prompt variant (isolated).",
            "source": "prompts.blackbox_refine.variation_generate_prompt",
            "candidate_id": candidate.candidate_id,
            "candidate_kind": candidate.kind,
            "parent_beam": int(candidate.parent_beam_index),
            "iteration": int(iteration),
        },
    )


def _judge_scores_step(
    inputs: PlanInputs,
    cfg: PromptBlackboxRefineConfig,
    iteration: int,
    judge: PromptBlackboxRefineJudgeConfig,
    candidates: list[CandidateSpec],
    *,
    config_path_root: str,
) -> ChatStep:
    stage_id = f"blackbox_refine.iter_{iteration:02d}.judge.{judge.id}"
    output_key = f"bbref.iter_{iteration:02d}.judge.{judge.id}.scores_json"

    judge_params: dict[str, Any] = {}
    judge_model = judge.model if judge.model else cfg.judge_model
    if judge_model:
        judge_params["model"] = judge_model

    temperature = (
        float(judge.temperature)
        if judge.temperature is not None
        else float(cfg.judge_temperature)
    )

    def _prompt(ctx: RunContext, *, candidates=candidates, judge=judge) -> str:
        candidate_rows: list[dict[str, Any]] = []
        for cand in candidates:
            text, _orig_chars, _truncated = _resolve_candidate_prompt_text(
                ctx,
                cand,
                stage_id=stage_id,
                max_prompt_chars=cfg.max_prompt_chars,
            )
            candidate_rows.append(
                {
                    "id": cand.candidate_id,
                    "prompt": text,
                }
            )

        # Mitigate positional/label bias by shuffling candidate order deterministically.
        shuffle_material = f"{int(ctx.seed)}|{int(iteration)}|{judge.id}".encode("utf-8")
        shuffle_seed = int.from_bytes(hashlib.sha256(shuffle_material).digest()[:8], "big")
        random.Random(shuffle_seed).shuffle(candidate_rows)

        raw_profile = _resolve_blackbox_profile_text(
            ctx,
            source=cfg.judge_profile_source,
            stage_id=stage_id,
            config_path=f"{config_path_root}.judge_profile_source",
        )

        novelty_summary = _resolve_novelty_summary(ctx)
        context_guidance = (str(ctx.outputs.get("context_guidance") or "").strip() or None)

        return prompt_variants_judge_prompt(
            judge_id=judge.id,
            rubric=judge.rubric,
            concepts=list(ctx.selected_concepts),
            context_guidance=context_guidance,
            raw_profile=raw_profile,
            candidates=candidate_rows,
            recent_motif_summary=novelty_summary,
        )

    return ChatStep(
        name=str(judge.id),
        prompt=_prompt,
        temperature=temperature,
        merge="none",
        capture_key=output_key,
        params=judge_params,
        meta={
            "doc": "Score prompt variants (strict JSON, isolated).",
            "source": "prompts.blackbox_refine.prompt_variants_judge_prompt",
            "judge_id": str(judge.id),
            "rubric": str(judge.rubric),
            "weight": float(judge.weight),
            "iteration": int(iteration),
        },
    )


def _select_action_step(
    inputs: PlanInputs,
    cfg: PromptBlackboxRefineConfig,
    iteration: int,
    candidates: list[CandidateSpec],
    *,
    config_path_root: str,
) -> ActionStep:
    stage_id = f"blackbox_refine.iter_{iteration:02d}.select"
    pipeline_step_path = f"pipeline/blackbox_refine.iter_{iteration:02d}/select"

    def _action(ctx: RunContext, *, iteration=iteration, candidates=candidates) -> dict[str, Any]:
        novelty_cfg = cfg.novelty
        novelty_enabled_cfg = bool(novelty_cfg.enabled and novelty_cfg.window > 0)
        novelty_summary = _resolve_novelty_summary(ctx)
        novelty_available = bool(isinstance(novelty_summary, dict) and novelty_summary.get("enabled"))

        warnings: list[str] = []
        if cfg.variation_prompt.include_novelty_summary and not novelty_available:
            warnings.append(
                f"{pipeline_step_path}: {config_path_root}.variation_prompt.include_novelty_summary=true but novelty summary is unavailable; using <none>"
            )
        if novelty_enabled_cfg and not novelty_available:
            warnings.append(
                f"{pipeline_step_path}: {config_path_root}.novelty.enabled=true but novelty summary is unavailable; novelty penalties disabled for this run"
            )

        candidate_by_id = {c.candidate_id: c for c in candidates}
        expected_ids = [c.candidate_id for c in candidates]

        candidate_prompts: dict[str, str] = {}
        candidate_rows: list[dict[str, Any]] = []
        for cand in candidates:
            text, original_chars, truncated = _resolve_candidate_prompt_text(
                ctx,
                cand,
                stage_id=stage_id,
                max_prompt_chars=cfg.max_prompt_chars,
            )
            candidate_prompts[cand.candidate_id] = text

            directive: str | None = None
            if cfg.variation_prompt.include_mutation_directive and cand.kind == "generated":
                directive = pick_mutation_directive(
                    mode=cfg.mutation_directives.mode,
                    directives=cfg.mutation_directives.directives,
                    seed=int(ctx.seed),
                    iteration=int(iteration),
                    candidate_id=cand.candidate_id,
                    candidate_ordinal=cand.candidate_ordinal,
                )

            candidate_rows.append(
                {
                    "id": cand.candidate_id,
                    "kind": cand.kind,
                    "parent_beam": cand.parent_beam_index,
                    "prompt_chars": len(text),
                    "prompt_chars_original": int(original_chars),
                    "prompt_truncated": bool(truncated),
                    "max_prompt_chars": int(cfg.max_prompt_chars)
                    if cfg.max_prompt_chars is not None
                    else None,
                    "mutation_directive": directive,
                }
            )

        truncated_ids = [row["id"] for row in candidate_rows if row.get("prompt_truncated") is True]
        if truncated_ids:
            warnings.append(
                f"Candidate prompt(s) exceeded {config_path_root}.max_prompt_chars and were truncated: max={cfg.max_prompt_chars} ids={truncated_ids}"
            )

        for warning in warnings:
            ctx.logger.warning("%s", warning)

        judge_outputs: list[dict[str, Any]] = []
        judge_scores_by_id: dict[str, dict[str, int]] = {}
        for judge in cfg.judging.judges:
            judge_stage_id = f"blackbox_refine.iter_{iteration:02d}.judge.{judge.id}"
            judge_output_key = f"bbref.iter_{iteration:02d}.judge.{judge.id}.scores_json"
            raw_json = ctx.outputs.get(judge_output_key)
            if not isinstance(raw_json, str) or not raw_json.strip():
                raise ValueError(f"Missing required output: {judge_output_key}")

            try:
                parsed = blackbox_scoring.parse_judge_scores_json(raw_json, expected_ids=expected_ids)
            except Exception as exc:
                setattr(exc, "pipeline_step", judge_stage_id)
                setattr(
                    exc,
                    "pipeline_path",
                    f"pipeline/blackbox_refine.iter_{iteration:02d}/judge/{judge.id}",
                )
                raise

            judge_scores_by_id[judge.id] = parsed
            judge_outputs.append(
                {
                    "judge_id": judge.id,
                    "model": judge.model or cfg.judge_model,
                    "temperature": judge.temperature
                    if judge.temperature is not None
                    else cfg.judge_temperature,
                    "weight": judge.weight,
                    "rubric": judge.rubric,
                    "raw_json": raw_json,
                    "parsed_scores": parsed,
                    "parse_status": "ok",
                }
            )

        aggregate = _aggregate_scores(
            candidates=expected_ids,
            judges=cfg.judging.judges,
            judge_scores_by_id=judge_scores_by_id,
            method=cfg.judging.aggregation,
            trimmed_mean_drop=cfg.judging.trimmed_mean_drop,
        )

        penalty_by_id: dict[str, int] = {cid: 0 for cid in expected_ids}
        novelty_breakdown_by_id: dict[str, dict[str, Any]] = {
            cid: {"penalty": 0, "reason": "novelty_disabled"} for cid in expected_ids
        }
        if novelty_enabled_cfg and novelty_available:
            candidate_cards = [
                {"id": cid, "prompt": candidate_prompts.get(cid, "")} for cid in expected_ids
            ]
            penalty_by_id, novelty_breakdown_by_id = blackbox_scoring.novelty_penalties(
                candidate_cards,
                novelty_cfg,
                novelty_summary,
                text_field="prompt",
            )
        elif novelty_enabled_cfg and not novelty_available:
            novelty_breakdown_by_id = {cid: {"penalty": 0, "reason": "novelty_missing"} for cid in expected_ids}

        scored: list[dict[str, Any]] = []
        for cid in expected_ids:
            raw_score = float(aggregate.get(cid, 0.0))
            novelty_penalty = int(penalty_by_id.get(cid, 0))
            effective = max(0.0, raw_score - float(novelty_penalty))
            cand = candidate_by_id.get(cid)
            scored.append(
                {
                    "id": cid,
                    "score": raw_score,
                    "novelty_penalty": novelty_penalty,
                    "novelty_detail": novelty_breakdown_by_id.get(cid),
                    "effective_score": effective,
                    "prompt_chars": len(candidate_prompts.get(cid, "")),
                    "parent_beam": cand.parent_beam_index if cand else None,
                    "kind": cand.kind if cand else None,
                }
            )

        scored.sort(key=lambda row: _candidate_sort_key(cfg, row))

        def _clip_for_feedback(text: str) -> tuple[str, bool]:
            max_chars = int(cfg.variation_prompt.score_feedback_max_chars)
            if max_chars <= 0:
                return (text or "").strip(), False
            raw = (text or "").strip()
            if not raw:
                return "", False
            if len(raw) <= max_chars:
                return raw, False
            clipped = raw[:max_chars].rstrip()
            return (clipped if clipped else ""), True

        def _row_to_feedback_example(row: Mapping[str, Any]) -> dict[str, Any]:
            cid = str(row.get("id") or "")
            prompt_text, prompt_truncated = _clip_for_feedback(candidate_prompts.get(cid, ""))
            return {
                "id": cid,
                "kind": row.get("kind"),
                "parent_beam": row.get("parent_beam"),
                "score": float(row.get("score", 0.0)),
                "novelty_penalty": int(row.get("novelty_penalty", 0) or 0),
                "effective_score": float(row.get("effective_score", 0.0)),
                "prompt": prompt_text,
                "prompt_truncated_for_feedback": bool(prompt_truncated),
            }

        score_feedback_by_beam: list[dict[str, Any]] = []
        beam_indices = sorted({int(c.parent_beam_index) for c in candidates if isinstance(c.parent_beam_index, int)})
        for beam_idx in beam_indices:
            rows_for_beam = [row for row in scored if row.get("parent_beam") == beam_idx]
            if not rows_for_beam:
                continue
            rows_for_beam.sort(key=lambda row: _candidate_sort_key(cfg, row))
            best_row = rows_for_beam[0]
            worst_row = rows_for_beam[-1]
            score_feedback_by_beam.append(
                {
                    "iteration": int(iteration),
                    "beam_index": int(beam_idx),
                    "best": _row_to_feedback_example(best_row),
                    "worst": _row_to_feedback_example(worst_row),
                }
            )

        score_feedback_overall: dict[str, Any] | None = None
        if scored:
            score_feedback_overall = {
                "iteration": int(iteration),
                "best": _row_to_feedback_example(scored[0]),
                "worst": _row_to_feedback_example(scored[-1]),
            }

        ctx.outputs[f"bbref.iter_{iteration:02d}.score_feedback_best_worst_by_beam"] = score_feedback_by_beam
        if score_feedback_overall is not None:
            ctx.outputs[f"bbref.iter_{iteration:02d}.score_feedback_best_worst_overall"] = score_feedback_overall

        exploration_rate = (
            float(cfg.selection.exploration_rate_override)
            if cfg.selection.exploration_rate_override is not None
            else float(cfg.exploration_rate)
        )
        scoring_seed = (int(ctx.seed) ^ 0xBB0F00D) + int(iteration)
        rng = random.Random(scoring_seed)
        exploration_roll = float(rng.random())
        explore = bool(exploration_rate > 0 and exploration_roll < exploration_rate)
        selection_mode = "explore" if explore else "exploit"

        k = 1 if cfg.algorithm == "hillclimb" else int(cfg.beam_width)
        selected = _select_candidates(cfg, scored=scored, k=k, exploration=explore, rng=rng)
        selected_ids = [row["id"] for row in selected]

        beams_out: list[dict[str, Any]] = []
        for beam_index, row in enumerate(selected, start=1):
            cid = str(row["id"])
            prompt = candidate_prompts.get(cid, "")
            if not prompt.strip():
                raise ValueError(f"Selected candidate produced empty prompt: {cid}")
            beams_out.append(
                {
                    "beam_index": beam_index,
                    "prompt": prompt,
                    "candidate_id": cid,
                    "score": float(row["score"]),
                    "novelty_penalty": int(row["novelty_penalty"]),
                    "effective_score": float(row["effective_score"]),
                    "parent_beam": row.get("parent_beam"),
                    "kind": row.get("kind"),
                }
            )

        ctx.outputs["bbref.beams"] = beams_out

        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        prompt_refine = ctx.blackbox_scoring.get("prompt_refine")
        if not isinstance(prompt_refine, dict):
            prompt_refine = {}
            ctx.blackbox_scoring["prompt_refine"] = prompt_refine
        iterations_log = prompt_refine.get("iterations")
        if not isinstance(iterations_log, list):
            iterations_log = []
            prompt_refine["iterations"] = iterations_log

        iterations_log.append(
            {
                "iteration": int(iteration),
                "algorithm": cfg.algorithm,
                "beam_width": int(cfg.beam_width),
                "branching_factor": int(cfg.branching_factor),
                "include_parents_as_candidates": bool(cfg.include_parents_as_candidates),
                "candidate_count": len(expected_ids),
                "variation_prompt": {
                    "template": cfg.variation_prompt.template,
                    "include_concepts": cfg.variation_prompt.include_concepts,
                    "include_context_guidance": cfg.variation_prompt.include_context_guidance,
                    "include_profile": cfg.variation_prompt.include_profile,
                    "profile_source": cfg.variation_prompt.profile_source,
                    "include_novelty_summary": cfg.variation_prompt.include_novelty_summary,
                    "include_mutation_directive": cfg.variation_prompt.include_mutation_directive,
                    "include_scoring_rubric": cfg.variation_prompt.include_scoring_rubric,
                    "score_feedback": cfg.variation_prompt.score_feedback,
                    "score_feedback_max_chars": int(cfg.variation_prompt.score_feedback_max_chars),
                },
                "candidates": candidate_rows,
                "judge_outputs": judge_outputs,
                "aggregate": {
                    "method": cfg.judging.aggregation,
                    "trimmed_mean_drop": int(cfg.judging.trimmed_mean_drop),
                    "scores": {cid: float(aggregate.get(cid, 0.0)) for cid in expected_ids},
                },
                "selection": {
                    "exploration_seed": scoring_seed,
                    "exploration_rate": exploration_rate,
                    "exploration_roll": exploration_roll,
                    "selection_mode": selection_mode,
                    "tie_breaker": cfg.selection.tie_breaker,
                    "selected_ids": selected_ids,
                    "novelty": {
                        "method": novelty_cfg.method,
                        "enabled_cfg": bool(novelty_enabled_cfg),
                        "summary_available": bool(novelty_available),
                        "missing_summary": bool(novelty_enabled_cfg and not novelty_available),
                    },
                    "score_table": scored,
                },
                "score_feedback": {
                    "mode": cfg.variation_prompt.score_feedback,
                    "by_beam": [
                        {
                            "beam_index": row.get("beam_index"),
                            "best_id": (row.get("best") or {}).get("id"),
                            "best_effective_score": (row.get("best") or {}).get("effective_score"),
                            "worst_id": (row.get("worst") or {}).get("id"),
                            "worst_effective_score": (row.get("worst") or {}).get("effective_score"),
                        }
                        for row in score_feedback_by_beam
                        if isinstance(row, Mapping)
                    ],
                    "output_keys": {
                        "by_beam": f"bbref.iter_{iteration:02d}.score_feedback_best_worst_by_beam",
                        "overall": f"bbref.iter_{iteration:02d}.score_feedback_best_worst_overall",
                    },
                },
                "beams_out": beams_out,
                "warnings": warnings,
            }
        )

        ctx.logger.info(
            "Blackbox prompt refine iter=%02d mode=%s selected=%s",
            int(iteration),
            selection_mode,
            selected_ids,
        )

        return {
            "iteration": int(iteration),
            "selection_mode": selection_mode,
            "selected_ids": selected_ids,
        }

    return ActionStep(
        name="select",
        fn=_action,
        merge="none",
        meta={
            "doc": "Select winning candidate(s) and update beams/state.",
            "source": "stages.blackbox_refine.loop._select_action_step",
            "iteration": int(iteration),
        },
    )


def _finalize_block(cfg: PromptBlackboxRefineConfig) -> Block:
    def _action(ctx: RunContext) -> str:
        beams = ctx.outputs.get("bbref.beams")
        if not isinstance(beams, list) or not beams:
            raise ValueError("blackbox_refine.finalize requires bbref.beams")
        first = beams[0] if isinstance(beams[0], dict) else None
        prompt = (first or {}).get("prompt") if isinstance(first, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("blackbox_refine.finalize produced empty prompt")
        final_prompt = prompt.strip()

        if ctx.blackbox_scoring is not None and isinstance(ctx.blackbox_scoring.get("prompt_refine"), dict):
            ctx.blackbox_scoring["prompt_refine"]["final_prompt"] = final_prompt

        ctx.logger.info("Blackbox prompt refine finalize: chars=%d", len(final_prompt))
        return final_prompt

    return make_action_stage_block(
        "blackbox_refine.finalize",
        fn=_action,
        merge="none",
        tags=("blackbox_refine",),
        doc="Finalize blackbox refinement loop output (beam[0]).",
        source="stages.blackbox_refine.loop._finalize_block",
    )


def _resolve_beam_prompt(ctx: RunContext, *, beam_index: int, stage_id: str) -> str:
    beams = ctx.outputs.get("bbref.beams")
    if not isinstance(beams, list) or not beams:
        raise ValueError(f"{stage_id} requires bbref.beams")
    if not (1 <= int(beam_index) <= len(beams)):
        raise ValueError(f"{stage_id} requires bbref.beams[{beam_index}] (beams={len(beams)})")
    beam = beams[int(beam_index) - 1]
    if not isinstance(beam, Mapping):
        raise ValueError(f"{stage_id} requires bbref.beams[{beam_index}] to be an object")
    prompt = beam.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{stage_id} requires a non-empty beam prompt (beam_index={beam_index})")
    return prompt.strip()


def _resolve_best_worst_score_feedback(
    ctx: RunContext,
    *,
    iteration: int,
    beam_index: int,
    stage_id: str,
) -> Mapping[str, Any] | None:
    """
    Resolve best/worst scored examples (with prompt text) from a previous iteration.

    For beam-search runs, the current beam may have originated from a different parent beam in the
    previous iteration; we follow `bbref.beams[*].parent_beam` to select the right feedback bucket.
    """
    if int(iteration) < 1:
        return None

    beams = ctx.outputs.get("bbref.beams")
    if not isinstance(beams, list) or not beams:
        return None
    if not (1 <= int(beam_index) <= len(beams)):
        raise ValueError(f"{stage_id} requires bbref.beams[{beam_index}] (beams={len(beams)})")
    beam = beams[int(beam_index) - 1]
    if not isinstance(beam, Mapping):
        return None

    origin = beam.get("parent_beam")
    origin_beam_index = int(origin) if isinstance(origin, int) and origin >= 1 else int(beam_index)

    by_beam_key = f"bbref.iter_{int(iteration):02d}.score_feedback_best_worst_by_beam"
    rows = ctx.outputs.get(by_beam_key)
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if int(row.get("beam_index") or 0) != origin_beam_index:
                continue
            best = row.get("best")
            worst = row.get("worst")
            if isinstance(best, Mapping) and isinstance(worst, Mapping):
                return {
                    "iteration": int(iteration),
                    "beam_index": origin_beam_index,
                    "best": dict(best),
                    "worst": dict(worst),
                }

    overall_key = f"bbref.iter_{int(iteration):02d}.score_feedback_best_worst_overall"
    overall = ctx.outputs.get(overall_key)
    if isinstance(overall, Mapping):
        best = overall.get("best")
        worst = overall.get("worst")
        if isinstance(best, Mapping) and isinstance(worst, Mapping):
            return {
                "iteration": int(iteration),
                "beam_index": origin_beam_index,
                "best": dict(best),
                "worst": dict(worst),
            }

    return None


def _resolve_candidate_prompt_text(
    ctx: RunContext,
    candidate: CandidateSpec,
    *,
    stage_id: str,
    max_prompt_chars: int | None,
) -> tuple[str, int, bool]:
    if candidate.kind == "parent":
        text = _resolve_beam_prompt(ctx, beam_index=candidate.parent_beam_index, stage_id=stage_id)
    else:
        assert candidate.output_key is not None
        raw = ctx.outputs.get(candidate.output_key)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Missing required output: {candidate.output_key}")
        text = raw.strip()

    original_chars = len(text)
    if max_prompt_chars is not None and original_chars > int(max_prompt_chars):
        truncated = text[: int(max_prompt_chars)].rstrip()
        return truncated, original_chars, True
    return text, original_chars, False


def _resolve_blackbox_profile_text(
    ctx: RunContext,
    *,
    source: str,
    stage_id: str,
    config_path: str,
) -> str:
    if source == "raw":
        raw_profile = str(ctx.outputs.get("preferences_guidance") or "").strip()
        return raw_profile
    if source == "generator_hints":
        hints = ctx.outputs.get("generator_profile_hints")
        if not isinstance(hints, str) or not hints.strip():
            raise ValueError(
                f"{stage_id} requires generator_profile_hints for {config_path}=generator_hints"
            )
        return hints
    if source == "generator_hints_plus_dislikes":
        hints = ctx.outputs.get("generator_profile_hints")
        if not isinstance(hints, str) or not hints.strip():
            raise ValueError(
                f"{stage_id} requires generator_profile_hints for {config_path}=generator_hints_plus_dislikes"
            )

        dislikes_raw = ctx.outputs.get("dislikes")
        if dislikes_raw is None:
            dislikes = []
        elif isinstance(dislikes_raw, list):
            dislikes = [str(v).strip() for v in dislikes_raw if str(v).strip()]
        else:
            raise ValueError(
                f"{stage_id} requires dislikes to be a list for {config_path}=generator_hints_plus_dislikes"
            )

        dislikes_block = "\n".join(f"- {item}" for item in dislikes) if dislikes else "- <none>"

        return (
            "Profile extraction (generator-safe hints):\n"
            + hints.strip()
            + "\n\nDislikes:\n"
            + dislikes_block
        ).strip()
    raise ValueError(f"Unknown profile source for {config_path}: {source}")


def _resolve_profile_representation(
    ctx: RunContext,
    *,
    profile_source: str,
    stage_id: str,
    config_path: str,
) -> str:
    raw_guidance = ctx.outputs.get("preferences_guidance")
    if raw_guidance is None:
        raw_guidance_text = ""
    elif isinstance(raw_guidance, str):
        raw_guidance_text = raw_guidance
    else:
        raise ValueError(f"{stage_id} requires preferences_guidance for {config_path}")

    generator_hints: str | None = None
    if profile_source in ("generator_hints", "combined"):
        hints = ctx.outputs.get("generator_profile_hints")
        if isinstance(hints, str) and hints.strip():
            generator_hints = hints
        elif profile_source == "generator_hints":
            raise ValueError(
                f"{stage_id} requires generator_profile_hints for {config_path}=generator_hints"
            )

    dislikes: list[str] | None = None
    if profile_source in ("dislikes_only", "combined"):
        dislikes_raw = ctx.outputs.get("dislikes")
        if dislikes_raw is None:
            dislikes = []
        elif isinstance(dislikes_raw, list):
            dislikes = [str(v).strip() for v in dislikes_raw if str(v).strip()]
        else:
            raise ValueError(f"{stage_id} requires dislikes to be a list for {config_path}")

    try:
        return profile_representation_from_guidance(
            profile_source=profile_source,
            preferences_guidance=raw_guidance_text,
            generator_profile_hints=generator_hints,
            dislikes=dislikes,
        )
    except Exception as exc:
        raise ValueError(f"{stage_id} failed to build profile representation for {config_path}: {exc}") from exc


def _resolve_novelty_summary(ctx: RunContext) -> Mapping[str, Any] | None:
    if ctx.blackbox_scoring is None:
        return None
    raw = ctx.blackbox_scoring.get("novelty_summary")
    return raw if isinstance(raw, dict) else None


def _aggregate_scores(
    *,
    candidates: list[str],
    judges: tuple[PromptBlackboxRefineJudgeConfig, ...],
    judge_scores_by_id: dict[str, dict[str, int]],
    method: str,
    trimmed_mean_drop: int,
) -> dict[str, float]:
    agg = (method or "").strip().lower()
    if agg not in ("mean", "median", "min", "max", "trimmed_mean"):
        raise ValueError(f"Unknown aggregation: {method!r}")

    out: dict[str, float] = {}
    for cid in candidates:
        items: list[tuple[int, float]] = []
        for judge in judges:
            scores = judge_scores_by_id.get(judge.id)
            if scores is None or cid not in scores:
                raise ValueError(f"Missing judge score: judge={judge.id} candidate={cid}")
            items.append((int(scores[cid]), float(judge.weight)))

        if agg == "min":
            out[cid] = float(min(score for score, _w in items))
            continue
        if agg == "max":
            out[cid] = float(max(score for score, _w in items))
            continue
        if agg == "median":
            scores_sorted = sorted(score for score, _w in items)
            mid = len(scores_sorted) // 2
            if len(scores_sorted) % 2 == 1:
                out[cid] = float(scores_sorted[mid])
            else:
                out[cid] = float((scores_sorted[mid - 1] + scores_sorted[mid]) / 2.0)
            continue

        if agg == "trimmed_mean":
            ordered = sorted(items, key=lambda sw: sw[0])
            if trimmed_mean_drop:
                ordered = ordered[trimmed_mean_drop : len(ordered) - trimmed_mean_drop]
            if not ordered:
                raise ValueError("trimmed_mean produced empty judge set")
            num = sum(float(score) * float(weight) for score, weight in ordered)
            denom = sum(float(weight) for _score, weight in ordered) or 1.0
            out[cid] = float(num / denom)
            continue

        # mean (weighted)
        num = sum(float(score) * float(weight) for score, weight in items)
        denom = sum(float(weight) for _score, weight in items) or 1.0
        out[cid] = float(num / denom)

    return out


def _candidate_sort_key(cfg: PromptBlackboxRefineConfig, row: Mapping[str, Any]) -> tuple:
    effective = float(row.get("effective_score", 0.0))
    raw = float(row.get("score", 0.0))
    candidate_id = str(row.get("id") or "")

    if cfg.selection.tie_breaker == "prefer_shorter":
        prompt_chars = int(row.get("prompt_chars", 0) or 0)
        return (-effective, -raw, prompt_chars, candidate_id)
    if cfg.selection.tie_breaker == "prefer_novel":
        penalty = int(row.get("novelty_penalty", 0) or 0)
        return (-effective, -raw, penalty, candidate_id)
    return (-effective, -raw, candidate_id)


def _select_candidates(
    cfg: PromptBlackboxRefineConfig,
    *,
    scored: list[dict[str, Any]],
    k: int,
    exploration: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > len(scored):
        raise ValueError(f"Selection requires k <= candidates (k={k} candidates={len(scored)})")

    if not exploration:
        if cfg.algorithm == "beam" and cfg.selection.group_by_beam:
            per_beam_best: dict[int, dict[str, Any]] = {}
            for row in scored:
                beam = row.get("parent_beam")
                if not isinstance(beam, int):
                    continue
                if beam not in per_beam_best:
                    per_beam_best[beam] = row
            chosen = list(per_beam_best.values())
            chosen.sort(key=lambda r: _candidate_sort_key(cfg, r))
            if len(chosen) >= k:
                return chosen[:k]
            remaining = [row for row in scored if row["id"] not in {c["id"] for c in chosen}]
            return [*chosen, *remaining[: (k - len(chosen))]]
        return scored[:k]

    pool_size = max(k, int(math.ceil(len(scored) / 4)))
    # Ensure exploration has a chance to pick something other than the top candidate
    # when k < len(scored).
    if len(scored) > k:
        pool_size = max(pool_size, k + 1)
    pool_size = min(pool_size, len(scored))
    pool = list(scored[:pool_size])

    if cfg.algorithm == "beam" and cfg.selection.group_by_beam:
        per_beam_best: dict[int, dict[str, Any]] = {}
        for row in pool:
            beam = row.get("parent_beam")
            if not isinstance(beam, int):
                continue
            existing = per_beam_best.get(beam)
            if existing is None or _candidate_sort_key(cfg, row) < _candidate_sort_key(cfg, existing):
                per_beam_best[beam] = row
        pool = list(per_beam_best.values())
        pool.sort(key=lambda r: _candidate_sort_key(cfg, r))

    selected: list[dict[str, Any]] = []
    remaining = list(pool)
    while remaining and len(selected) < k:
        weights = [max(1.0, float(row.get("effective_score", 0.0) or 0.0)) for row in remaining]
        total = float(sum(weights))
        pick = float(rng.uniform(0.0, total))
        running = 0.0
        chosen_idx = len(remaining) - 1
        for idx, (_row, weight) in enumerate(zip(remaining, weights, strict=False)):
            running += float(weight)
            if pick <= running:
                chosen_idx = idx
                break
        selected.append(remaining.pop(chosen_idx))

    selected.sort(key=lambda r: _candidate_sort_key(cfg, r))
    if len(selected) < k:
        remaining_all = [row for row in scored if row["id"] not in {s["id"] for s in selected}]
        selected.extend(remaining_all[: (k - len(selected))])
    return selected[:k]


KIND_ID = "blackbox_refine.loop"


def _parse_loop_config(
    cfg: ConfigNamespace, *, inputs: PlanInputs
) -> tuple[PromptBlackboxRefineConfig, str]:
    """Parse stage-owned configuration for `blackbox_refine.loop`."""

    seed_source = cfg.get_str("seed_source", default=None)
    if seed_source is None:
        inferred = "draft" if (inputs.draft_prompt or "").strip() else "blackbox"
        seed_source = inferred
        cfg.set_effective("seed_source", inferred)

    iterations = cfg.get_int("iterations", default=3, min_value=1)
    algorithm = cfg.get_str("algorithm", default="hillclimb", choices=("hillclimb", "beam"))
    if algorithm is None:
        raise ValueError(f"{cfg.path}.algorithm cannot be null")

    beam_width = cfg.get_int("beam_width", default=3, min_value=1)
    if algorithm == "beam" and beam_width < 2:
        raise ValueError(f"{cfg.path}.beam_width must be >= 2 when algorithm=beam (got {beam_width})")

    branching_factor = cfg.get_int("branching_factor", default=6, min_value=1)
    include_parents_as_candidates = cfg.get_bool("include_parents_as_candidates", default=True)

    generator_model = cfg.get_str("generator_model", default=None)
    generator_temperature = cfg.get_float(
        "generator_temperature", default=0.9, min_value=0.0, max_value=2.0
    )
    max_prompt_chars = cfg.get_optional_int("max_prompt_chars", default=3500, min_value=1)

    vp_ns = cfg.namespace("variation_prompt", default={})
    template = vp_ns.get_str("template", default="v1", choices=("v1", "v2"))
    if template is None:
        raise ValueError(f"{vp_ns.path}.template cannot be null")
    include_concepts = vp_ns.get_bool("include_concepts", default=True)
    include_context_guidance = vp_ns.get_bool("include_context_guidance", default=False)
    include_profile = vp_ns.get_bool("include_profile", default=True)
    profile_source = vp_ns.get_str(
        "profile_source",
        default="likes_dislikes",
        choices=("raw", "generator_hints", "likes_dislikes", "dislikes_only", "combined"),
    )
    if profile_source is None:
        raise ValueError(f"{vp_ns.path}.profile_source cannot be null")
    include_novelty_summary = vp_ns.get_bool("include_novelty_summary", default=True)
    include_mutation_directive = vp_ns.get_bool("include_mutation_directive", default=True)
    include_scoring_rubric = vp_ns.get_bool("include_scoring_rubric", default=True)
    score_feedback = vp_ns.get_str(
        "score_feedback",
        default="best_worst",
        choices=("none", "best_worst"),
    )
    if score_feedback is None:
        raise ValueError(f"{vp_ns.path}.score_feedback cannot be null")
    score_feedback_max_chars = vp_ns.get_int("score_feedback_max_chars", default=900, min_value=0)
    variation_prompt = PromptBlackboxRefineVariationPromptConfig(
        template=str(template),
        include_concepts=bool(include_concepts),
        include_context_guidance=bool(include_context_guidance),
        include_profile=bool(include_profile),
        profile_source=str(profile_source),
        include_novelty_summary=bool(include_novelty_summary),
        include_mutation_directive=bool(include_mutation_directive),
        include_scoring_rubric=bool(include_scoring_rubric),
        score_feedback=str(score_feedback),
        score_feedback_max_chars=int(score_feedback_max_chars),
    )

    md_ns = cfg.namespace("mutation_directives", default={})
    directive_mode = md_ns.get_str("mode", default="random", choices=("none", "random", "cycle", "fixed"))
    if directive_mode is None:
        raise ValueError(f"{md_ns.path}.mode cannot be null")
    directives = tuple(md_ns.get_list_str("directives", default=(), allow_empty=True))
    mutation_directives = PromptBlackboxRefineMutationDirectivesConfig(
        mode=str(directive_mode),
        directives=tuple(directives),
    )

    judging_ns = cfg.namespace("judging", default={})
    judges_raw = judging_ns.get_list_mapping("judges")
    judges: list[PromptBlackboxRefineJudgeConfig] = []
    for idx, judge_item in enumerate(judges_raw):
        judge_ns = ConfigNamespace(judge_item, path=f"{judging_ns.path}.judges[{idx}]")
        judge_id = judge_ns.get_str("id")
        if judge_id is None:
            raise ValueError(f"{judge_ns.path}.id cannot be null")
        rubric = judge_ns.get_str(
            "rubric",
            default="default",
            choices=("default", "strict", "novelty_heavy"),
        )
        if rubric is None:
            raise ValueError(f"{judge_ns.path}.rubric cannot be null")
        weight = judge_ns.get_float("weight", default=1.0, min_value=0.0)
        model = judge_ns.get_str("model", default=None)
        temperature = judge_ns.get_optional_float("temperature", default=None, min_value=0.0, max_value=2.0)
        judge_ns.assert_consumed()
        judges.append(
            PromptBlackboxRefineJudgeConfig(
                id=str(judge_id),
                rubric=str(rubric),
                weight=float(weight),
                model=model,
                temperature=temperature,
            )
        )

    if not judges:
        raise ValueError(f"{judging_ns.path}.judges must contain at least one judge")
    seen_judges: set[str] = set()
    dupes = [j.id for j in judges if (j.id in seen_judges or seen_judges.add(j.id)) and j.id]
    if dupes:
        raise ValueError(f"{judging_ns.path}.judges contains duplicate ids: {', '.join(sorted(set(dupes)))}")

    aggregation = judging_ns.get_str(
        "aggregation",
        default="mean",
        choices=("mean", "median", "min", "max", "trimmed_mean"),
    )
    if aggregation is None:
        raise ValueError(f"{judging_ns.path}.aggregation cannot be null")
    trimmed_mean_drop = judging_ns.get_int("trimmed_mean_drop", default=0, min_value=0)
    if aggregation == "trimmed_mean" and (2 * int(trimmed_mean_drop) >= len(judges)):
        raise ValueError(
            f"{judging_ns.path}.trimmed_mean_drop is too large for judge count (drop={trimmed_mean_drop} judges={len(judges)})"
        )
    judging = PromptBlackboxRefineJudgingConfig(
        judges=tuple(judges),
        aggregation=str(aggregation),
        trimmed_mean_drop=int(trimmed_mean_drop),
    )

    selection_ns = cfg.namespace("selection", default={})
    exploration_rate_override = selection_ns.get_optional_float(
        "exploration_rate_override", default=None, min_value=0.0, max_value=1.0
    )
    group_by_beam = selection_ns.get_bool("group_by_beam", default=False)
    tie_breaker = selection_ns.get_str(
        "tie_breaker",
        default="stable_id",
        choices=("stable_id", "prefer_shorter", "prefer_novel"),
    )
    if tie_breaker is None:
        raise ValueError(f"{selection_ns.path}.tie_breaker cannot be null")
    selection = PromptBlackboxRefineSelectionConfig(
        exploration_rate_override=exploration_rate_override,
        group_by_beam=bool(group_by_beam),
        tie_breaker=str(tie_breaker),
    )

    judge_model = cfg.get_str("judge_model", default=None)
    judge_temperature = cfg.get_float("judge_temperature", default=0.0, min_value=0.0, max_value=2.0)
    judge_profile_source = cfg.get_str(
        "judge_profile_source",
        default="raw",
        choices=("raw", "generator_hints", "generator_hints_plus_dislikes"),
    )
    if judge_profile_source is None:
        raise ValueError(f"{cfg.path}.judge_profile_source cannot be null")
    exploration_rate = cfg.get_float("exploration_rate", default=0.15, min_value=0.0, max_value=1.0)

    novelty_ns = cfg.namespace("novelty", default={})
    novelty_enabled = novelty_ns.get_bool("enabled", default=False)
    novelty_window = novelty_ns.get_int("window", default=0, min_value=0)
    novelty_method = novelty_ns.get_str(
        "method",
        default="df_overlap_v1",
        choices=("df_overlap_v1",),
    )
    if novelty_method is None:
        raise ValueError(f"{novelty_ns.path}.method cannot be null")
    novelty_df_min = novelty_ns.get_int("df_min", default=3, min_value=1)
    novelty_max_motifs = novelty_ns.get_int("max_motifs", default=200, min_value=1)
    novelty_min_token_len = novelty_ns.get_int("min_token_len", default=3, min_value=1)
    novelty_stopwords_extra = tuple(novelty_ns.get_list_str("stopwords_extra", default=(), allow_empty=True))
    novelty_max_penalty = novelty_ns.get_int("max_penalty", default=20, min_value=0)
    novelty_df_cap = novelty_ns.get_int("df_cap", default=10, min_value=1)
    novelty_alpha_only = novelty_ns.get_bool("alpha_only", default=True)
    novelty_scaling = novelty_ns.get_str(
        "scaling",
        default="linear",
        choices=("linear", "sqrt", "quadratic"),
    )
    if novelty_scaling is None:
        raise ValueError(f"{novelty_ns.path}.scaling cannot be null")

    novelty_cfg = NoveltyConfig(
        enabled=bool(novelty_enabled),
        window=int(novelty_window),
        method=str(novelty_method),
        df_min=int(novelty_df_min),
        max_motifs=int(novelty_max_motifs),
        min_token_len=int(novelty_min_token_len),
        stopwords_extra=tuple(novelty_stopwords_extra),
        max_penalty=int(novelty_max_penalty),
        df_cap=int(novelty_df_cap),
        alpha_only=bool(novelty_alpha_only),
        scaling=str(novelty_scaling),
    )

    loop_cfg = PromptBlackboxRefineConfig(
        algorithm=str(algorithm),
        iterations=int(iterations),
        beam_width=int(beam_width),
        branching_factor=int(branching_factor),
        include_parents_as_candidates=bool(include_parents_as_candidates),
        generator_model=generator_model,
        generator_temperature=float(generator_temperature),
        max_prompt_chars=max_prompt_chars,
        variation_prompt=variation_prompt,
        mutation_directives=mutation_directives,
        judging=judging,
        selection=selection,
        judge_model=judge_model,
        judge_temperature=float(judge_temperature),
        judge_profile_source=str(judge_profile_source),
        exploration_rate=float(exploration_rate),
        novelty=novelty_cfg,
    )

    return loop_cfg, str(seed_source)


def _build_blackbox_refine_loop(
    inputs: PlanInputs,
    *,
    instance_id: str,
    cfg: ConfigNamespace,
) -> Block:
    """Build the composite stage that runs init + N iterations + finalize."""

    if instance_id != KIND_ID:
        raise ValueError(f"{KIND_ID} does not support custom instance ids: {instance_id}")

    loop_cfg, seed_source = _parse_loop_config(cfg, inputs=inputs)

    nodes: list[Any] = [
        _init_state_block(
            inputs,
            loop_cfg,
            seed_output_key="bbref.seed_prompt",
            seed_source=seed_source,
        ),
    ]
    for iteration in range(1, int(loop_cfg.iterations) + 1):
        nodes.append(_iteration_stage_block(inputs, loop_cfg, iteration, config_path_root=cfg.path))
    nodes.append(_finalize_block(loop_cfg))

    cfg.assert_consumed()
    return Block(
        name=instance_id,
        merge="none",
        nodes=nodes,
        meta={
            "doc": "Blackbox refinement loop (init + iterations + finalize).",
            "tags": ["blackbox_refine"],
        },
    )


STAGE = StageRef(
    id=KIND_ID,
    builder=_build_blackbox_refine_loop,
    doc="Run the blackbox refinement loop (init + N iterations + finalize).",
    source="stages.blackbox_refine.loop._build_blackbox_refine_loop",
    tags=("blackbox_refine",),
    kind="composite",
    io=StageIO(
        requires=("bbref.seed_prompt",),
        provides=("bbref.beams",),
    ),
)
