from __future__ import annotations

import json
import textwrap
from typing import Any, Mapping


def _parse_profile_sections(preferences_guidance: str) -> dict[str, str]:
    text = (preferences_guidance or "").strip()
    if not text:
        return {}

    sections: dict[str, str] = {}
    for block in text.split("\n\n"):
        raw = block.strip()
        if not raw:
            continue
        lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0].strip()
        if not header.endswith(":"):
            continue
        key = header[:-1].strip().lower()
        body = "\n".join(lines[1:]).strip()
        sections[key] = body
    return sections


def profile_representation_from_guidance(
    *,
    profile_source: str,
    preferences_guidance: str,
    generator_profile_hints: str | None,
    dislikes: list[str] | None,
) -> str:
    """
    Build a profile representation string for prompt templates.

    This function is intentionally pure and does not access the run context.
    Validation (missing required artifacts) is handled by the caller.
    """

    source = (profile_source or "").strip().lower()
    raw = (preferences_guidance or "").strip()
    hints = (generator_profile_hints or "").strip()
    dislikes_list = [str(v).strip() for v in (dislikes or []) if str(v).strip()]

    if source == "raw":
        return raw or "<none>"
    if source == "generator_hints":
        return hints or "<none>"

    sections = _parse_profile_sections(raw)
    likes = sections.get("likes", "").strip()
    dislikes_section = sections.get("dislikes", "").strip()
    hates_section = sections.get("hates", "").strip()

    if source == "likes_dislikes":
        blocks: list[str] = []
        if likes:
            blocks.append("Likes:\n" + likes)
        if hates_section:
            blocks.append("Hates:\n" + hates_section)
        if dislikes_section:
            blocks.append("Dislikes:\n" + dislikes_section)
        return "\n\n".join(blocks).strip() or "<none>"

    if source == "dislikes_only":
        blocks = []
        if hates_section:
            blocks.append("Hates:\n" + hates_section)
        if dislikes_section:
            blocks.append("Dislikes:\n" + dislikes_section)
        if blocks:
            return "\n\n".join(blocks).strip()
        if dislikes_list:
            return (
                "Dislikes/Hates:\n" + "\n".join(f"- {item}" for item in dislikes_list)
            ).strip()
        return "<none>"

    if source == "combined":
        blocks = []
        if hints:
            blocks.append("Generator-safe profile hints:\n" + hints)
        blocks.append(
            "Likes/Dislikes/Hates (raw, minimal):\n"
            + profile_representation_from_guidance(
                profile_source="likes_dislikes",
                preferences_guidance=raw,
                generator_profile_hints=hints,
                dislikes=dislikes_list,
            )
        )
        return "\n\n".join(blocks).strip() or "<none>"

    raise ValueError(f"Unknown profile_source: {profile_source!r}")


def scoring_rubric_text(*, rubric: str) -> str:
    key = (rubric or "").strip().lower() or "default"
    if key not in ("default", "strict", "novelty_heavy"):
        raise ValueError(f"Unknown rubric: {rubric!r}")

    if key == "strict":
        return textwrap.dedent(
            """\
            Score 0-100. Be strict and literal.

            Criteria (in priority order):
            1) Alignment with selected concepts and explicit constraints.
            2) Clear, renderable, non-contradictory visual description (subject, setting, style, lighting, composition).
            3) Strong adherence to user likes/dislikes and avoid-list intent.
            4) No fluff: avoid vague adjectives, empty hype, and redundant synonyms.
            """
        ).strip()

    if key == "novelty_heavy":
        return textwrap.dedent(
            """\
            Score 0-100 with a heavy novelty bias.

            Criteria (in priority order):
            1) Alignment with selected concepts and user preferences.
            2) Novelty: avoid recently repeated motifs and clichéd combinations.
            3) Visual specificity and coherence (renderable, concrete, consistent).
            4) Prompt craftsmanship: concise, high-signal, minimal redundancy.
            """
        ).strip()

    return textwrap.dedent(
        """\
        Score 0-100.

        Criteria (in priority order):
        1) Alignment with selected concepts.
        2) Alignment with user likes/dislikes.
        3) Visual specificity and coherence (renderable, concrete, consistent).
        4) Prompt craftsmanship: concise, high-signal, concrete nouns, minimal redundancy.
        5) Originality: avoid clichéd combinations and avoid repeating recent motifs when applicable.
        """
    ).strip()


def variation_generate_prompt(
    *,
    template: str,
    base_prompt: str,
    concepts: list[str],
    context_guidance: str | None,
    profile_text: str | None,
    novelty_summary: Mapping[str, Any] | None,
    mutation_directive: str | None,
    include_concepts: bool,
    include_context_guidance: bool,
    include_profile: bool,
    include_novelty_summary: bool,
    include_mutation_directive: bool,
    include_scoring_rubric: bool,
    max_prompt_chars: int | None,
) -> str:
    tmpl = (template or "").strip().lower()
    if tmpl not in ("v1", "v2"):
        raise ValueError(f"Unknown variation prompt template: {template!r}")

    base = (base_prompt or "").strip()
    if not base:
        raise ValueError("base_prompt is empty")

    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    profile_block = (profile_text or "").strip() or "<none>"
    context_block = (context_guidance or "").strip() or "<none>"
    novelty_block = (
        json.dumps(novelty_summary, ensure_ascii=False, indent=2)
        if isinstance(novelty_summary, dict) and novelty_summary
        else "<none>"
    )
    directive_block = (mutation_directive or "").strip() or "<none>"
    scoring_block = scoring_rubric_text(rubric="default")

    output_constraints: list[str] = [
        "Output ONLY the new final image prompt. No title, no quotes, no markdown.",
        "Do not include explanations or analysis.",
    ]
    if max_prompt_chars is not None:
        output_constraints.append(f"Output must be <= {int(max_prompt_chars)} characters.")

    mutation_bar = textwrap.dedent(
        """\
        Mutation bar (must satisfy):
        - The variant must be meaningfully different from the base prompt (not a paraphrase).
        - Prefer concrete, renderable specifics (materials, textures, camera/framing, lighting design); avoid vague filler.
        - If NOVELTY SUMMARY is present, actively avoid the most repeated motifs/phrases.
        - Follow MUTATION DIRECTIVE when present.
        """
    ).strip()

    if tmpl == "v2":
        return textwrap.dedent(
            f"""\
            You rewrite image prompts.

            Task: produce ONE image prompt variant based on the base prompt and optional reference blocks.

            BASE PROMPT (to improve):
            {base}

            {"SELECTED CONCEPTS:\n" + concepts_block if include_concepts else "SELECTED CONCEPTS:\n<omitted>"}

            {"CONTEXT GUIDANCE:\n" + context_block if include_context_guidance else "CONTEXT GUIDANCE:\n<omitted>"}

            {"PROFILE (authoritative preferences):\n" + profile_block if include_profile else "PROFILE:\n<omitted>"}

            {"NOVELTY SUMMARY:\n" + novelty_block if include_novelty_summary else "NOVELTY SUMMARY:\n<omitted>"}

            {"MUTATION DIRECTIVE:\n" + directive_block if include_mutation_directive else "MUTATION DIRECTIVE:\n<omitted>"}

            {"SCORING RUBRIC (what judges reward):\n" + scoring_block if include_scoring_rubric else "SCORING RUBRIC:\n<omitted>"}

            {mutation_bar}

            HARD RULES:
            - {output_constraints[0]}
            - {output_constraints[1]}
            {("- " + output_constraints[2]) if len(output_constraints) >= 3 else ""}

            Output the new image prompt now:
            """
        ).strip()

    # v1 (minimal, labeled sections)
    parts: list[str] = [
        "Generate one improved image prompt variant. Use the base prompt as the starting point.",
        "It must be meaningfully different from the base (not a paraphrase) and change at least TWO major axes (composition/perspective, setting/time/weather, lighting/palette, medium/style, subject/action/props).",
        "Prefer concrete, renderable specifics; avoid vague filler and clichéd combinations.",
        "",
        "Base prompt:",
        base,
        "",
    ]
    if include_concepts:
        parts.extend(["Selected concepts:", concepts_block, ""])
    if include_context_guidance:
        parts.extend(["Context guidance:", context_block, ""])
    if include_profile:
        parts.extend(["Profile (authoritative):", profile_block, ""])
    if include_novelty_summary:
        parts.extend(["Novelty summary:", novelty_block, ""])
    if include_mutation_directive:
        parts.extend(["Mutation directive:", directive_block, ""])
    if include_scoring_rubric:
        parts.extend(["Scoring rubric:", scoring_block, ""])

    parts.extend(["Mutation bar:", mutation_bar, ""])
    parts.extend(["Output constraints:", *[f"- {rule}" for rule in output_constraints]])
    return "\n".join(parts).strip()


def prompt_variants_judge_prompt(
    *,
    judge_id: str,
    rubric: str,
    concepts: list[str],
    context_guidance: str | None = None,
    raw_profile: str,
    candidates: list[dict[str, Any]],
    recent_motif_summary: Mapping[str, Any] | None,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    context_block = (context_guidance or "").strip() or "<none>"
    profile_block = (raw_profile or "").strip() or "<none>"
    motif_block = (
        json.dumps(recent_motif_summary, ensure_ascii=False, indent=2)
        if isinstance(recent_motif_summary, dict) and recent_motif_summary
        else "<none>"
    )
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    rubric_text = scoring_rubric_text(rubric=rubric)

    return textwrap.dedent(
        f"""\
        You are a strict numeric judge ({judge_id}). Score each candidate image prompt from 0 to 100.

        Selected concepts (must align strongly):
        {concepts_block}

        Context guidance (optional; reward tasteful use when present):
        {context_block}

        Raw user profile (authoritative preferences; use for judging only):
        {profile_block}

        Recent motifs summary (penalize repetition when present):
        {motif_block}

        Candidates (JSON array of objects with id+prompt):
        {candidates_json}

        Rubric:
        {rubric_text}

        Output MUST be strict JSON ONLY with this exact schema and nothing else:
        {{
          "scores": [
            {{"id": "A", "score": 0}}
          ]
        }}

        Rules:
        - "score" must be an integer in [0, 100].
        - Include exactly one score entry per candidate id (no missing, no extra ids).
        - No additional keys, no explanations, no prose.
        """
    ).strip()
