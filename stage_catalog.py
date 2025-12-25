from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import prompts

from pipeline import RunContext

from prompt_plans import ActionStageSpec, PlanInputs, StageNodeSpec, StageSpec

StageBuilder = Callable[[PlanInputs], StageNodeSpec]


@dataclass(frozen=True)
class StageEntry:
    stage_id: str
    builder: StageBuilder
    doc: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = ()


class StageCatalog:
    _REGISTRY: dict[str, StageEntry] = {}

    @classmethod
    def register(
        cls,
        stage_id: str,
        *,
        doc: str | None = None,
        source: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> Callable[[StageBuilder], StageBuilder]:
        if not isinstance(stage_id, str) or not stage_id.strip():
            raise TypeError("stage_id must be a non-empty string")
        key = stage_id.strip()

        def decorator(fn: StageBuilder) -> StageBuilder:
            if key in cls._REGISTRY:
                raise ValueError(f"Duplicate stage id: {key}")
            cls._REGISTRY[key] = StageEntry(
                stage_id=key,
                builder=fn,
                doc=doc,
                source=source,
                tags=tuple(tags),
            )
            return fn

        return decorator

    @classmethod
    def build(cls, stage_id: str, inputs: PlanInputs) -> StageNodeSpec:
        if not isinstance(stage_id, str) or not stage_id.strip():
            raise ValueError("stage_id must be a non-empty string")
        key = stage_id.strip()

        entry = cls._REGISTRY.get(key)
        if entry is None and "." not in key:
            matches = sorted(s for s in cls._REGISTRY.keys() if s.endswith("." + key))
            if len(matches) == 1:
                key = matches[0]
                entry = cls._REGISTRY[key]
            elif len(matches) > 1:
                raise ValueError(
                    f"Ambiguous stage id: {stage_id} (matches: {', '.join(matches)})"
                )
        if entry is None:
            available = ", ".join(sorted(cls._REGISTRY.keys())) or "<none>"
            raise ValueError(f"Unknown stage id: {stage_id} (available: {available})")

        spec = entry.builder(inputs)
        if spec.stage_id != key:
            raise ValueError(
                f"Stage builder returned mismatched stage_id: expected={key} got={spec.stage_id}"
            )

        next_doc = spec.doc if spec.doc is not None else entry.doc
        next_source = spec.source if spec.source is not None else entry.source
        next_tags = spec.tags if spec.tags else entry.tags

        if next_doc != spec.doc or next_source != spec.source or next_tags != spec.tags:
            if isinstance(spec, ActionStageSpec):
                return ActionStageSpec(
                    stage_id=spec.stage_id,
                    fn=spec.fn,
                    merge=spec.merge,
                    tags=tuple(next_tags),
                    output_key=spec.output_key,
                    doc=next_doc,
                    source=next_source,
                    is_default_capture=spec.is_default_capture,
                )
            return StageSpec(
                stage_id=spec.stage_id,
                prompt=spec.prompt,
                temperature=spec.temperature,
                params=dict(spec.params),
                allow_empty_prompt=spec.allow_empty_prompt,
                allow_empty_response=spec.allow_empty_response,
                tags=tuple(next_tags),
                refinement_policy=spec.refinement_policy,
                is_default_capture=spec.is_default_capture,
                merge=spec.merge,
                output_key=spec.output_key,
                doc=next_doc,
                source=next_source,
            )

        return spec

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._REGISTRY.keys()))

    @classmethod
    def describe(cls) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "stage_id": entry.stage_id,
                "doc": entry.doc,
                "source": entry.source,
                "tags": list(entry.tags),
            }
            for entry in sorted(cls._REGISTRY.values(), key=lambda e: e.stage_id)
        )


@StageCatalog.register(
    "preprompt.select_concepts",
    doc="Select concepts (random/fixed/file) and store them on the run context.",
    source="prompts.select_random_concepts",
    tags=("preprompt",),
)
def preprompt_select_concepts(inputs: PlanInputs) -> ActionStageSpec:
    selection_cfg = inputs.cfg.prompt_concepts.selection

    def _action(ctx: RunContext) -> dict[str, Any]:
        strategy = selection_cfg.strategy
        if strategy == "random":
            selected = prompts.select_random_concepts(inputs.prompt_data, inputs.rng)
            file_path: str | None = None
        elif strategy in ("fixed", "file"):
            selected = list(selection_cfg.fixed)
            file_path = selection_cfg.file_path if strategy == "file" else None
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown prompt.concepts.selection.strategy: {strategy!r}")

        if not selected:
            raise ValueError(
                f"Concept selection produced no concepts (strategy={strategy!r})"
            )

        ctx.selected_concepts = list(selected)
        return {
            "strategy": strategy,
            "file_path": file_path,
            "selected_concepts": list(ctx.selected_concepts),
        }

    return ActionStageSpec(
        stage_id="preprompt.select_concepts",
        fn=_action,
        merge="none",
    )


@StageCatalog.register(
    "preprompt.filter_concepts",
    doc="Apply configured concept filters (in order) and record outcomes.",
    source="concept_filters.apply_concept_filters",
    tags=("preprompt",),
)
def preprompt_filter_concepts(inputs: PlanInputs) -> ActionStageSpec:
    filters_cfg = inputs.cfg.prompt_concepts.filters

    def _action(ctx: RunContext) -> dict[str, Any]:
        from concept_filters import apply_concept_filters, make_dislike_rewrite_filter

        concepts_in = list(ctx.selected_concepts)
        filters = []
        applied: list[str] = []
        skipped: list[dict[str, str]] = []

        if not filters_cfg.enabled:
            skipped.append({"name": "<all>", "reason": "disabled"})
        else:
            for name in filters_cfg.order:
                if name == "dislike_rewrite":
                    if not filters_cfg.dislike_rewrite.enabled:
                        skipped.append({"name": name, "reason": "disabled"})
                        continue

                    dislikes_raw = ctx.outputs.get("dislikes")
                    dislikes = (
                        [str(value).strip() for value in dislikes_raw if str(value).strip()]
                        if isinstance(dislikes_raw, list)
                        else []
                    )
                    filters.append(
                        make_dislike_rewrite_filter(
                            dislikes=dislikes,
                            ai_text=inputs.ai_text,
                            temperature=filters_cfg.dislike_rewrite.temperature,
                        )
                    )
                    applied.append(name)
                else:  # pragma: no cover - guarded by config validation
                    skipped.append({"name": name, "reason": "unknown"})

        filtered, outcomes = apply_concept_filters(concepts_in, filters, logger=ctx.logger)

        ctx.outputs["concept_filter_log"] = {
            "enabled": bool(filters_cfg.enabled),
            "order": list(filters_cfg.order),
            "applied": applied,
            "skipped": skipped,
            "input": concepts_in,
            "output": list(filtered),
            "filters": outcomes,
        }
        ctx.selected_concepts = list(filtered)

        return {
            "input": concepts_in,
            "output": list(filtered),
            "applied": applied,
            "skipped": skipped,
        }

    return ActionStageSpec(
        stage_id="preprompt.filter_concepts",
        fn=_action,
        merge="none",
    )


@StageCatalog.register(
    "standard.initial_prompt",
    doc="Generate candidate themes/stories.",
    source="prompts.generate_first_prompt",
    tags=("standard",),
)
def standard_initial_prompt(inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        if not ctx.selected_concepts:
            raise ValueError(
                "standard.initial_prompt requires selected concepts; "
                "run preprompt.select_concepts first (or include it in the plan)."
            )

        prompt_1, _selected = prompts.generate_first_prompt(
            inputs.prompt_data,
            inputs.user_profile,
            inputs.rng,
            context_guidance=(inputs.context_guidance or None),
            selected_concepts=list(ctx.selected_concepts),
        )
        return prompt_1

    return StageSpec(
        stage_id="standard.initial_prompt",
        prompt=_prompt,
        temperature=0.8,
    )


def _standard_simple_prompt(
    inputs: PlanInputs,
    *,
    stage_id: str,
    prompt_fn: Callable[[], str],
) -> StageSpec:
    return StageSpec(
        stage_id=stage_id,
        prompt=lambda _ctx: prompt_fn(),
        temperature=0.8,
    )


@StageCatalog.register(
    "standard.section_2_choice",
    doc="Pick the most compelling choice.",
    source="prompts.generate_second_prompt",
    tags=("standard",),
)
def standard_section_2_choice(inputs: PlanInputs) -> StageSpec:
    return _standard_simple_prompt(
        inputs,
        stage_id="standard.section_2_choice",
        prompt_fn=prompts.generate_second_prompt,
    )


@StageCatalog.register(
    "standard.section_2b_title_and_story",
    doc="Generate title and story details.",
    source="prompts.generate_secondB_prompt",
    tags=("standard",),
)
def standard_section_2b_title_and_story(inputs: PlanInputs) -> StageSpec:
    return _standard_simple_prompt(
        inputs,
        stage_id="standard.section_2b_title_and_story",
        prompt_fn=prompts.generate_secondB_prompt,
    )


@StageCatalog.register(
    "standard.section_3_message_focus",
    doc="Clarify the message to convey.",
    source="prompts.generate_third_prompt",
    tags=("standard",),
)
def standard_section_3_message_focus(inputs: PlanInputs) -> StageSpec:
    return _standard_simple_prompt(
        inputs,
        stage_id="standard.section_3_message_focus",
        prompt_fn=prompts.generate_third_prompt,
    )


@StageCatalog.register(
    "standard.section_4_concise_description",
    doc="Write the concise detailed description.",
    source="prompts.generate_fourth_prompt",
    tags=("standard",),
)
def standard_section_4_concise_description(inputs: PlanInputs) -> StageSpec:
    return _standard_simple_prompt(
        inputs,
        stage_id="standard.section_4_concise_description",
        prompt_fn=prompts.generate_fourth_prompt,
    )


@StageCatalog.register(
    "standard.image_prompt_creation",
    doc="Create the final image prompt.",
    source="prompts.generate_image_prompt",
    tags=("standard",),
)
def standard_image_prompt_creation(inputs: PlanInputs) -> StageSpec:
    return StageSpec(
        stage_id="standard.image_prompt_creation",
        prompt=lambda _ctx: prompts.generate_image_prompt(),
        temperature=0.8,
        is_default_capture=True,
    )


def _resolve_blackbox_profile_text(
    ctx: RunContext,
    *,
    source: str,
    stage_id: str,
    config_path: str,
) -> str:
    if source == "raw":
        return str(ctx.outputs.get("preferences_guidance") or "")
    if source == "generator_hints":
        hints = ctx.outputs.get("generator_profile_hints")
        if not isinstance(hints, str) or not hints.strip():
            raise ValueError(
                f"{stage_id} requires generator_profile_hints for {config_path}=generator_hints"
            )
        return hints

    raise ValueError(f"Unknown profile source for {config_path}: {source}")


@StageCatalog.register(
    "blackbox.prepare",
    doc="Prepare blackbox scoring (novelty summary + default generator hints).",
    source="blackbox_scoring.extract_recent_motif_summary",
    tags=("blackbox",),
)
def blackbox_prepare(inputs: PlanInputs) -> ActionStageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring

    def _action(ctx: RunContext) -> dict[str, Any]:
        import blackbox_scoring

        if not scoring_cfg.enabled:
            raise ValueError("blackbox.prepare requires prompt.scoring.enabled=true")

        novelty_enabled = bool(scoring_cfg.novelty.enabled and scoring_cfg.novelty.window > 0)
        ctx.logger.info(
            "Blackbox scoring enabled: num_ideas=%d, exploration_rate=%.3g, novelty=%s",
            scoring_cfg.num_ideas,
            scoring_cfg.exploration_rate,
            novelty_enabled,
        )

        novelty_summary: dict[str, Any]
        if novelty_enabled:
            try:
                novelty_summary = blackbox_scoring.extract_recent_motif_summary(
                    generations_csv_path=ctx.cfg.generations_csv_path,
                    window=scoring_cfg.novelty.window,
                )
            except Exception as exc:
                ctx.logger.warning(
                    "Novelty enabled but history unavailable; disabling novelty for this run: %s %s",
                    ctx.cfg.generations_csv_path,
                    exc,
                )
                novelty_enabled = False
                novelty_summary = {
                    "enabled": False,
                    "window": scoring_cfg.novelty.window,
                    "rows_considered": 0,
                    "top_tokens": [],
                }
        else:
            novelty_summary = {
                "enabled": False,
                "window": scoring_cfg.novelty.window,
                "rows_considered": 0,
                "top_tokens": [],
            }

        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        ctx.blackbox_scoring["novelty_summary"] = novelty_summary

        # Default generator hints (may be overwritten by profile_abstraction stage).
        ctx.outputs["generator_profile_hints"] = str(ctx.outputs.get("preferences_guidance") or "")

        return {
            "novelty_enabled": novelty_enabled,
            "novelty_window": scoring_cfg.novelty.window,
        }

    return ActionStageSpec(
        stage_id="blackbox.prepare",
        fn=_action,
        merge="none",
    )


@StageCatalog.register(
    "blackbox.profile_abstraction",
    doc="Create generator-safe profile hints.",
    source="prompts.profile_abstraction_prompt",
    tags=("blackbox",),
)
def blackbox_profile_abstraction(inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        return prompts.profile_abstraction_prompt(
            preferences_guidance=str(ctx.outputs.get("preferences_guidance") or "")
        )

    return StageSpec(
        stage_id="blackbox.profile_abstraction",
        prompt=_prompt,
        temperature=0.0,
        merge="none",
        output_key="generator_profile_hints",
        refinement_policy="none",
    )


@StageCatalog.register(
    "blackbox.idea_cards_generate",
    doc="Generate idea cards (strict JSON).",
    source="prompts.idea_cards_generate_prompt",
    tags=("blackbox",),
)
def blackbox_idea_cards_generate(inputs: PlanInputs) -> StageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring

    def _prompt(ctx: RunContext) -> str:
        hints = ctx.outputs.get("generator_profile_hints")
        if not isinstance(hints, str) or not hints.strip():
            hints = str(ctx.outputs.get("preferences_guidance") or "")
        return prompts.idea_cards_generate_prompt(
            concepts=list(ctx.selected_concepts),
            generator_profile_hints=str(hints or ""),
            num_ideas=scoring_cfg.num_ideas,
        )

    return StageSpec(
        stage_id="blackbox.idea_cards_generate",
        prompt=_prompt,
        temperature=0.8,
        merge="none",
        output_key="idea_cards_json",
        refinement_policy="none",
    )


@StageCatalog.register(
    "blackbox.idea_cards_judge_score",
    doc="Judge idea cards and emit scores (strict JSON).",
    source="prompts.idea_cards_judge_prompt",
    tags=("blackbox",),
)
def blackbox_idea_cards_judge_score(inputs: PlanInputs) -> StageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring
    judge_params: dict[str, Any] = {}
    if scoring_cfg.judge_model:
        judge_params["model"] = scoring_cfg.judge_model

    judge_profile_source = scoring_cfg.judge_profile_source

    def _prompt(ctx: RunContext) -> str:
        import json

        idea_cards_json = ctx.outputs.get("idea_cards_json")
        if not isinstance(idea_cards_json, str) or not idea_cards_json.strip():
            raise ValueError("Missing required output: idea_cards_json")

        novelty_summary = (ctx.blackbox_scoring or {}).get("novelty_summary")
        recent_motif_summary: str | None = None
        if isinstance(novelty_summary, dict) and novelty_summary.get("enabled"):
            recent_motif_summary = json.dumps(novelty_summary, ensure_ascii=False, indent=2)

        return prompts.idea_cards_judge_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=_resolve_blackbox_profile_text(
                ctx,
                source=judge_profile_source,
                stage_id="blackbox.idea_cards_judge_score",
                config_path="prompt.scoring.judge_profile_source",
            ),
            idea_cards_json=idea_cards_json,
            recent_motif_summary=recent_motif_summary,
        )

    return StageSpec(
        stage_id="blackbox.idea_cards_judge_score",
        prompt=_prompt,
        temperature=scoring_cfg.judge_temperature,
        merge="none",
        params=judge_params,
        output_key="idea_scores_json",
        refinement_policy="none",
    )


@StageCatalog.register(
    "blackbox.select_idea_card",
    doc="Select an idea card using judge scores (and novelty penalties when enabled).",
    source="blackbox_scoring.select_candidate",
    tags=("blackbox",),
)
def blackbox_select_idea_card(inputs: PlanInputs) -> ActionStageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring

    def _action(ctx: RunContext) -> dict[str, Any]:
        import random

        import blackbox_scoring

        if not scoring_cfg.enabled:
            raise ValueError("blackbox.select_idea_card requires prompt.scoring.enabled=true")

        idea_cards_json = ctx.outputs.get("idea_cards_json")
        if not isinstance(idea_cards_json, str) or not idea_cards_json.strip():
            raise ValueError("Missing required output: idea_cards_json")
        idea_scores_json = ctx.outputs.get("idea_scores_json")
        if not isinstance(idea_scores_json, str) or not idea_scores_json.strip():
            raise ValueError("Missing required output: idea_scores_json")

        novelty_summary: dict[str, Any] | None = None
        if ctx.blackbox_scoring is not None:
            raw = ctx.blackbox_scoring.get("novelty_summary")
            if isinstance(raw, dict) and raw.get("enabled"):
                novelty_summary = raw

        try:
            idea_cards = blackbox_scoring.parse_idea_cards_json(
                idea_cards_json, expected_num_ideas=scoring_cfg.num_ideas
            )
        except Exception as exc:
            setattr(exc, "pipeline_step", "blackbox.idea_cards_generate")
            setattr(exc, "pipeline_path", "pipeline/blackbox.idea_cards_generate/draft")
            raise

        expected_ids = [card.get("id") for card in idea_cards if isinstance(card, dict)]

        try:
            scores = blackbox_scoring.parse_judge_scores_json(
                idea_scores_json, expected_ids=expected_ids
            )
        except Exception as exc:
            setattr(exc, "pipeline_step", "blackbox.idea_cards_judge_score")
            setattr(exc, "pipeline_path", "pipeline/blackbox.idea_cards_judge_score/draft")
            raise

        scoring_seed = int(ctx.seed) ^ 0xB10C5C0F
        selection = blackbox_scoring.select_candidate(
            scores=scores,
            idea_cards=idea_cards,
            exploration_rate=scoring_cfg.exploration_rate,
            rng=random.Random(scoring_seed),
            novelty_summary=novelty_summary,
        )

        selected_card = next(
            (card for card in idea_cards if card.get("id") == selection.selected_id),
            None,
        )
        if not isinstance(selected_card, dict):
            raise ValueError(
                f"selection inconsistency: selected id not found: {selection.selected_id}"
            )

        ctx.outputs["selected_idea_card"] = selected_card
        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        ctx.blackbox_scoring.update(
            {
                "scoring_seed": scoring_seed,
                "exploration_rate": scoring_cfg.exploration_rate,
                "exploration_roll": selection.exploration_roll,
                "selection_mode": selection.selection_mode,
                "selected_id": selection.selected_id,
                "selected_score": selection.selected_score,
                "selected_effective_score": selection.selected_effective_score,
                "score_table": selection.score_table,
            }
        )

        ctx.logger.info(
            "Selected candidate: id=%s, score=%d, selection_mode=%s",
            selection.selected_id,
            selection.selected_score,
            selection.selection_mode,
        )

        return {
            "selected_id": selection.selected_id,
            "selection_mode": selection.selection_mode,
        }

    return ActionStageSpec(
        stage_id="blackbox.select_idea_card",
        fn=_action,
        merge="none",
    )


@StageCatalog.register(
    "blackbox.image_prompt_creation",
    doc="Create final prompt from selected idea card.",
    source="prompts.final_prompt_from_selected_idea_prompt",
    tags=("blackbox",),
)
def blackbox_image_prompt_creation(inputs: PlanInputs) -> StageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring
    final_profile_source = scoring_cfg.final_profile_source

    def _prompt(ctx: RunContext) -> str:
        selected_card = ctx.outputs.get("selected_idea_card")
        if not isinstance(selected_card, dict):
            raise ValueError("Missing required output: selected_idea_card")
        return prompts.final_prompt_from_selected_idea_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=_resolve_blackbox_profile_text(
                ctx,
                source=final_profile_source,
                stage_id="blackbox.image_prompt_creation",
                config_path="prompt.scoring.final_profile_source",
            ),
            selected_idea_card=selected_card,
        )

    return StageSpec(
        stage_id="blackbox.image_prompt_creation",
        prompt=_prompt,
        temperature=0.8,
        is_default_capture=True,
    )


@StageCatalog.register(
    "refine.image_prompt_refine",
    doc="Refine a provided draft into the final image prompt.",
    source="prompts.refine_image_prompt_prompt",
    tags=("refine",),
)
def refine_image_prompt_refine(inputs: PlanInputs) -> StageSpec:
    draft_text = (inputs.draft_prompt or "").strip()
    if not draft_text:
        raise ValueError(
            "stage refine.image_prompt_refine requires inputs.draft_prompt (prompt.plan=refine_only)"
        )

    prompt = prompts.refine_image_prompt_prompt(draft_text)
    return StageSpec(
        stage_id="refine.image_prompt_refine",
        prompt=prompt,
        temperature=0.8,
        is_default_capture=True,
    )
