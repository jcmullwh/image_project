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
        rendered = "\n\n".join(blocks).strip()
        return rendered or "<none>"

    if source == "dislikes_only":
        blocks = []
        if hates_section:
            blocks.append("Hates:\n" + hates_section)
        if dislikes_section:
            blocks.append("Dislikes:\n" + dislikes_section)
        if blocks:
            rendered = "\n\n".join(blocks).strip()
            return rendered
        if dislikes_list:
            rendered = (
                "Dislikes/Hates:\n" + "\n".join(f"- {item}" for item in dislikes_list)
            ).strip()
            return rendered
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

            Profile handling (format-agnostic):
            - The profile may be bullets, free text, "generator-safe hints", likes/dislikes lists, or other formats.
            - Treat DISLIKE/HATE/AVOID/NEVER/DO-NOT signals as high-priority constraints.
            - Treat LIKE/LOVE/PREFER signals as moderate-positive guidance (helpful, not absolute).
            - Use judgment: reward adjacent/compatible interpretations of strong likes when plausible (do not keyword-match).
            - Do NOT give credit for "fixing" a dislike by adding the disliked thing with a positive modifier (e.g., dislike "wrong/incorrect X" does NOT mean adding "correct X" is good).

            Scoring method (apply in order; clamp final to [0,100]):
            - Start at 70.
            - Selected concepts + explicit constraints: +0 to +25
              - Strong: +18 to +25 | Moderate: +10 to +17 | Weak: +0 to +9
              - If a concept/constraint is contradicted: -15 to -30 instead of adding points
            - Renderability + internal consistency: +0 to +20
              - Strong: +14 to +20 | Moderate: +7 to +13 | Weak: +0 to +6
              - Contradictions / impossible mashups: -10 to -25
            - Dislikes/Hates (hard penalties; apply once at highest severity):
              - DISLIKE/AVOID minor brush-by: -15
              - DISLIKE/AVOID meaningful presence: -35
              - HATE/NEVER or central/defining violation: -70 and cap total score at 20
            - Likes/Loves (moderate impact; adjacent matches allowed): +0 to +8
              - Strong: +6 to +8 | Moderate: +3 to +5 | Weak: +0 to +2
            - No fluff / high-signal writing:
              - Dense, concrete, minimal redundancy: +0 to +7
              - Vague hype / empty adjectives / redundant synonyms: -5 to -15

            """
        ).strip()

    if key == "novelty_heavy":
        return textwrap.dedent(
            """\
            Score 0-100 with a heavy novelty bias.

            Profile handling (format-agnostic):
            - The profile may be bullets, free text, "generator-safe hints", likes/dislikes lists, or other formats.
            - Treat DISLIKE/HATE/AVOID/NEVER/DO-NOT signals as high-priority constraints.
            - Treat LIKE/LOVE/PREFER signals as moderate-positive guidance (helpful, not absolute).
            - Use judgment: reward adjacent/compatible interpretations of strong likes when plausible (do not keyword-match).
            - Do NOT give credit for "fixing" a dislike by adding the disliked thing with a positive modifier (e.g., dislike "wrong/incorrect X" does NOT mean adding "correct X" is good).

            Scoring method (apply in order; clamp final to [0,100]):
            - Start at 70.
            - Selected concepts alignment: +0 to +20
              - Strong: +14 to +20 | Moderate: +7 to +13 | Weak: +0 to +6
              - If a concept is contradicted: -15 to -30 instead of adding points
            - Novelty + anti-repetition (use "recent motifs" when present):
              - Fresh, non-cliche, non-repetitive: +10 to +20
              - Somewhat fresh: +5 to +9
              - Generic/cliche: +0 to +4
              - Repeats recent motifs meaningfully: -10 to -25
            - Dislikes/Hates (hard penalties; apply once at highest severity):
              - DISLIKE/AVOID minor brush-by: -15
              - DISLIKE/AVOID meaningful presence: -35
              - HATE/NEVER or central/defining violation: -70 and cap total score at 20
            - Likes/Loves (moderate impact; adjacent matches allowed): +0 to +8
              - Strong: +5 to +8 | Moderate: +2 to +4 | Weak: +0 to +1
            - Visual specificity + coherence: +0 to +12; contradictions: -10 to -25
            - Prompt craftsmanship (concise, high-signal): +0 to +5; vagueness/redundancy: -5 to -12

            """
        ).strip()

    return textwrap.dedent(
        """\
        Score 0-100.

        Profile handling (format-agnostic):
        - The profile may be bullets, free text, "generator-safe hints", likes/dislikes lists, or other formats.
        - Treat DISLIKE/HATE/AVOID/NEVER/DO-NOT signals as high-priority constraints.
        - Treat LIKE/LOVE/PREFER signals as moderate-positive guidance (helpful, not absolute).
        - Use judgment: reward adjacent/compatible interpretations of strong likes when plausible (do not keyword-match).
        - Do NOT give credit for "fixing" a dislike by adding the disliked thing with a positive modifier (e.g., dislike "wrong/incorrect X" does NOT mean adding "correct X" is good).

        Scoring method (apply in order; clamp final to [0,100]):
        - Start at 70.
        - Selected concepts alignment: +0 to +20
          - Strong: +14 to +20 | Moderate: +7 to +13 | Weak: +0 to +6
          - If a concept is contradicted: -15 to -30 instead of adding points
        - Dislikes/Hates (hard penalties; apply once at highest severity):
          - DISLIKE/AVOID minor brush-by: -15
          - DISLIKE/AVOID meaningful presence: -35
          - HATE/NEVER or central/defining violation: -70 and cap total score at 20
        - Likes/Loves (moderate impact; adjacent matches allowed): +0 to +10
          - Strong: +5 to +10 | Moderate: +2 to +4 | Weak: +0 to +1
        - Visual specificity + coherence: +0 to +15; contradictions: -10 to -25
        - Prompt craftsmanship (concise, high-signal, concrete nouns): +0 to +8; vagueness/redundancy: -5 to -12
        - Originality / non-cliche (use "recent motifs" when applicable): +0 to +7; repetition/cliche: -5 to -20

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
    score_feedback: Mapping[str, Any] | None,
    include_concepts: bool,
    include_context_guidance: bool,
    include_profile: bool,
    include_novelty_summary: bool,
    include_mutation_directive: bool,
    include_scoring_rubric: bool,
    score_feedback_max_chars: int | None,
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

    def _format_score_feedback() -> str | None:
        if not isinstance(score_feedback, Mapping) or not score_feedback:
            return None

        max_chars = int(score_feedback_max_chars) if score_feedback_max_chars is not None else None
        if max_chars is not None and max_chars <= 0:
            max_chars = None

        def _fmt_num(v: Any) -> str:
            if isinstance(v, bool) or v is None:
                return "<none>"
            try:
                return f"{float(v):.1f}"
            except Exception:
                return str(v)

        def _clip(text: str) -> tuple[str, bool]:
            t = (text or "").strip()
            if not t:
                return "<none>", False
            if max_chars is None or len(t) <= max_chars:
                return t, False
            clipped = t[:max_chars].rstrip()
            return (clipped if clipped else "<truncated>"), True

        iteration = score_feedback.get("iteration")
        beam_index = score_feedback.get("beam_index")
        header_bits: list[str] = []
        if iteration is not None:
            header_bits.append(f"iter={iteration}")
        if beam_index is not None:
            header_bits.append(f"beam={beam_index}")
        header = (" (" + ", ".join(header_bits) + ")") if header_bits else ""

        best = score_feedback.get("best")
        worst = score_feedback.get("worst")
        if not isinstance(best, Mapping) or not isinstance(worst, Mapping):
            return None

        best_prompt, best_trunc = _clip(str(best.get("prompt") or ""))
        worst_prompt, worst_trunc = _clip(str(worst.get("prompt") or ""))

        best_id = str(best.get("id") or "").strip() or "<unknown>"
        worst_id = str(worst.get("id") or "").strip() or "<unknown>"

        best_raw = _fmt_num(best.get("score"))
        best_penalty = str(best.get("novelty_penalty") if best.get("novelty_penalty") is not None else "<none>")
        best_eff = _fmt_num(best.get("effective_score"))

        worst_raw = _fmt_num(worst.get("score"))
        worst_penalty = str(
            worst.get("novelty_penalty") if worst.get("novelty_penalty") is not None else "<none>"
        )
        worst_eff = _fmt_num(worst.get("effective_score"))

        trunc_note = ""
        if best_trunc or worst_trunc:
            trunc_note = f"\n\n(Note: prompt text truncated to {max_chars} chars for context.)"

        return textwrap.dedent(
            f"""\
            SCORE FEEDBACK EXAMPLES{header} (higher is better; use as a gradient):

            BEST (id={best_id} raw={best_raw} novelty_penalty={best_penalty} effective={best_eff}):
            {best_prompt}

            WORST (id={worst_id} raw={worst_raw} novelty_penalty={worst_penalty} effective={worst_eff}):
            {worst_prompt}
            """
        ).strip() + trunc_note

    score_feedback_block = _format_score_feedback()

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
        feedback_section = f"\n\n{score_feedback_block}\n" if score_feedback_block else ""
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
            {feedback_section}

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
    if score_feedback_block:
        parts.extend([score_feedback_block, ""])

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
        - Candidate id/order carries no meaning; judge solely on the prompt text.
        """
    ).strip()
