from __future__ import annotations

import json
import os
import re
import textwrap
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Callable

from image_project.framework.config import RunConfig


@dataclass(frozen=True)
class ResolvedPromptInputs:
    draft_prompt: str | None = None


def resolve_prompt_inputs(cfg: RunConfig, *, required: tuple[str, ...] = ()) -> ResolvedPromptInputs:
    """
    Resolve plan inputs that may involve I/O, separate from orchestration.

    Plans should consume these resolved inputs rather than reading files directly.
    """

    required_set = set(required)
    unknown_required = sorted(required_set - {"draft_prompt"})
    if unknown_required:
        raise ValueError(f"Unknown required inputs: {unknown_required}")

    draft_prompt: str | None = None
    if cfg.prompt_refine_only_draft:
        draft_prompt = cfg.prompt_refine_only_draft
    elif cfg.prompt_refine_only_draft_path:
        path = str(cfg.prompt_refine_only_draft_path)
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

        prompt = textwrap.dedent(
            f"""\
            Selected concepts (keep length and order): 
            {json.dumps(input_concepts, ensure_ascii=False)}

            User dislikes (avoid conflicts): 
            {json.dumps(clean_dislikes, ensure_ascii=False)}

            If any selected concept conflicts with a dislike, rewrite just that concept so it no longer conflicts but still fits the original creative intent and variety. If there is no conflict, keep the concept unchanged.

            Return ONLY a JSON array of the revised concepts (strings), same length and order as provided. Do not add commentary or keys.
            """
        ).strip()

        try:
            response = ai_text.text_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You rewrite selected creative concepts so none of them conflict with the user's dislikes. "
                            "Keep variety, keep count, and avoid over-censoring."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ConceptFilterOutcome(
                name="dislike_rewrite",
                input_concepts=input_concepts,
                output_concepts=input_concepts,
                error=str(exc),
            )

        parsed = _parse_concept_list(response)
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


def _parse_concept_list(response: Any) -> list[str]:
    """
    Try to coerce the model response into a clean list of concept strings.
    """
    if not isinstance(response, str):
        return []

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()

    candidates = [cleaned]

    bracket_match = re.search(r"\\[.*\\]", cleaned, flags=re.DOTALL)
    if bracket_match:
        candidates.insert(0, bracket_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if isinstance(parsed, list):
            coerced = [str(item).strip() for item in parsed if str(item).strip()]
            if coerced:
                return coerced

    return []
