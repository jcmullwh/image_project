from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Callable

from image_project.framework.prompt_pipeline.pipeline_overrides import PromptPipelineConfig
from image_project.prompts import concept_filters as concept_filter_prompts


@dataclass(frozen=True)
class ResolvedPromptInputs:
    draft_prompt: str | None = None


def resolve_prompt_inputs(
    prompt_cfg: PromptPipelineConfig, *, required: tuple[str, ...] = ()
) -> ResolvedPromptInputs:
    """
    Resolve plan inputs that may involve I/O, separate from orchestration.

    Plans should consume these resolved inputs rather than reading files directly.
    """

    required_set = set(required)
    unknown_required = sorted(required_set - {"draft_prompt"})
    if unknown_required:
        raise ValueError(f"Unknown required inputs: {unknown_required}")

    draft_prompt: str | None = None
    refine_only = prompt_cfg.refine_only
    if refine_only.draft:
        draft_prompt = refine_only.draft
    elif refine_only.draft_path:
        path = str(refine_only.draft_path)
        if not os.path.exists(path):
            raise ValueError(f"prompt.refine_only.draft_path not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            draft_prompt = handle.read()

    draft_prompt = (draft_prompt or "").strip() or None
    if "draft_prompt" in required_set and not draft_prompt:
        raise ValueError("prompt.plan=refine_only requires prompt.refine_only.draft or draft_path")

    return ResolvedPromptInputs(draft_prompt=draft_prompt)


ConceptFilter = Callable[[list[str]], "ConceptFilterOutcome"]


@dataclass
class ConceptFilterOutcome:
    name: str
    input_concepts: list[str]
    output_concepts: list[str]
    raw_response: str | None = None
    error: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_dislikes(user_profile: Any) -> list[str]:
    """
    Pull a clean list of dislikes from the user profile DataFrame (if present).
    """
    if user_profile is None:
        return []

    def _get_values(column: str) -> list[str]:
        try:
            series = user_profile.get(column)
        except Exception:
            series = None
        if series is None:
            return []
        return [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]

    combined: list[str] = [*_get_values("Dislikes"), *_get_values("Hates")]

    # De-dupe while preserving order for stable concept filtering.
    seen: set[str] = set()
    out: list[str] = []
    for value in combined:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def apply_concept_filters(
    concepts: list[str],
    filters: Iterable[ConceptFilter],
    *,
    logger: Any = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Run the selected concepts through a pipeline of filters.
    Returns the final concepts and a serializable log of each filter outcome.
    """
    current = [str(concept).strip() for concept in concepts if str(concept).strip()]
    outcomes: list[dict[str, Any]] = []

    for filter_fn in filters:
        try:
            outcome = filter_fn(current)
        except Exception as exc:  # pragma: no cover - defensive
            outcome = ConceptFilterOutcome(
                name=getattr(filter_fn, "__name__", "unknown_filter"),
                input_concepts=current,
                output_concepts=current,
                error=str(exc),
            )

        current = list(outcome.output_concepts)
        outcomes.append(outcome.as_dict())

        if logger:
            logger.info(
                "Concept filter %s: input=%s output=%s",
                outcome.name,
                outcome.input_concepts,
                outcome.output_concepts,
            )
            if outcome.raw_response:
                logger.debug(
                    "Concept filter %s raw response: %s", outcome.name, outcome.raw_response
                )
            if outcome.error:
                logger.warning("Concept filter %s error: %s", outcome.name, outcome.error)
            elif outcome.note:
                logger.info("Concept filter %s note: %s", outcome.name, outcome.note)

    return current, outcomes


def make_dislike_rewrite_filter(
    *,
    dislikes: list[str],
    ai_text: Any,
    temperature: float = 0.25,
) -> ConceptFilter:
    """
    Build a filter that asks the LLM to reinterpret concepts that conflict with user dislikes.
    """

    clean_dislikes = [str(value).strip() for value in dislikes if str(value).strip()]

    def _filter(concepts: list[str]) -> ConceptFilterOutcome:
        input_concepts = [str(concept).strip() for concept in concepts if str(concept).strip()]

        if not clean_dislikes:
            return ConceptFilterOutcome(
                name="dislike_rewrite",
                input_concepts=input_concepts,
                output_concepts=input_concepts,
                note="skipped: no dislikes provided",
            )

        if not input_concepts:
            return ConceptFilterOutcome(
                name="dislike_rewrite",
                input_concepts=input_concepts,
                output_concepts=input_concepts,
                note="skipped: no concepts provided",
            )

        if ai_text is None or not hasattr(ai_text, "text_chat"):
            return ConceptFilterOutcome(
                name="dislike_rewrite",
                input_concepts=input_concepts,
                output_concepts=input_concepts,
                note="skipped: ai_text unavailable",
            )

        messages = concept_filter_prompts.build_dislike_rewrite_messages(
            selected_concepts=input_concepts,
            dislikes=clean_dislikes,
        )

        try:
            response = ai_text.text_chat(messages, temperature=temperature)
        except Exception as exc:  # pragma: no cover - defensive
            return ConceptFilterOutcome(
                name="dislike_rewrite",
                input_concepts=input_concepts,
                output_concepts=input_concepts,
                error=str(exc),
            )

        parsed = concept_filter_prompts.parse_concept_list_response(response)
        if not parsed:
            return ConceptFilterOutcome(
                name="dislike_rewrite",
                input_concepts=input_concepts,
                output_concepts=input_concepts,
                raw_response=response if isinstance(response, str) else None,
                error="Failed to parse JSON array; kept original concepts",
            )

        return ConceptFilterOutcome(
            name="dislike_rewrite",
            input_concepts=input_concepts,
            output_concepts=parsed,
            raw_response=response if isinstance(response, str) else None,
        )

    return _filter
