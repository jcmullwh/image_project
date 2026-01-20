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
              - HATE/NEVER or central/defining presence: -60
            """
        ).strip()

    if key == "novelty_heavy":
        return textwrap.dedent(
            """\
            Score 0-100. Strongly reward novelty and penalize repeated motifs.

            - Start at 65.
            - Concepts alignment: +0 to +20
            - Profile alignment: +0 to +20 (dislikes/hates can still hard-penalize)
            - Novelty: +0 to +25 (prefer unusual but coherent combinations)
            - Renderability: +0 to +15
            - Repetition penalty: -0 to -30 when repeating recent motifs/phrases.
            """
        ).strip()

    return textwrap.dedent(
        """\
        Score 0-100.

        - Start at 70.
        - Concepts alignment: +0 to +20
        - Profile alignment: +0 to +20 (dislikes/hates can penalize heavily)
        - Renderability + coherence: +0 to +20
        - Novelty (optional): +0 to +10
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
    include_concepts: bool = True,
    include_context_guidance: bool = True,
    include_profile: bool = True,
    include_novelty_summary: bool = True,
    include_mutation_directive: bool = True,
    include_scoring_rubric: bool = True,
    score_feedback_max_chars: int = 900,
    max_prompt_chars: int | None = None,
) -> str:
    tmpl = (template or "").strip().lower() or "v1"
    if tmpl not in ("v1", "v2"):
        raise ValueError(f"Unknown variation template: {template!r}")

    base = (base_prompt or "").strip()
    if not base:
        raise ValueError("base_prompt cannot be empty")

    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    context_block = (context_guidance or "").strip() or "<none>"
    profile_block = (profile_text or "").strip() or "<none>"
    novelty_block = (
        json.dumps(novelty_summary, ensure_ascii=False, indent=2)
        if isinstance(novelty_summary, Mapping) and novelty_summary
        else "<none>"
    )
    directive_block = (mutation_directive or "").strip() or "<none>"
    scoring_block = scoring_rubric_text(rubric="default") if include_scoring_rubric else "<omitted>"

    def _clip_text(text: str) -> tuple[str, bool]:
        raw = (text or "").strip()
        if not raw:
            return "", False
        max_chars = int(score_feedback_max_chars)
        if max_chars <= 0 or len(raw) <= max_chars:
            return raw, False
        clipped = raw[:max_chars].rstrip()
        return (clipped if clipped else ""), True

    def _format_score_example(label: str, example: Mapping[str, Any] | None) -> list[str]:
        if not isinstance(example, Mapping) or not example:
            return [f"{label}:", "- <missing>"]

        cid = str(example.get("id") or "").strip() or "<unknown>"
        raw_score = example.get("score")
        effective = example.get("effective_score")
        novelty_penalty = example.get("novelty_penalty")
        kind = example.get("kind")
        parent_beam = example.get("parent_beam")

        parts: list[str] = [f"{label}:"]

        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
            raw_str = str(float(raw_score))
        else:
            raw_str = "<missing>"

        if isinstance(effective, (int, float)) and not isinstance(effective, bool):
            effective_str = str(float(effective))
        else:
            effective_str = "<missing>"

        if isinstance(novelty_penalty, (int, float)) and not isinstance(novelty_penalty, bool):
            novelty_str = str(int(novelty_penalty))
        else:
            novelty_str = "<missing>"

        extra = []
        if kind is not None:
            extra.append(f"kind={kind}")
        if parent_beam is not None:
            extra.append(f"parent_beam={parent_beam}")
        extra_suffix = f" {' '.join(extra)}" if extra else ""

        parts.append(
            f"- id={cid} raw={raw_str} effective={effective_str} novelty_penalty={novelty_str}{extra_suffix}"
        )

        prompt_raw = example.get("prompt")
        prompt = str(prompt_raw or "").strip() if prompt_raw is not None else ""
        prompt, clipped = _clip_text(prompt)
        if clipped:
            parts.append(f"- prompt (clipped to {int(score_feedback_max_chars)} chars):")
        else:
            parts.append("- prompt:")
        parts.append(prompt or "<missing>")

        return parts

    score_feedback_block = ""
    if isinstance(score_feedback, Mapping) and score_feedback:
        best = score_feedback.get("best") if isinstance(score_feedback, Mapping) else None
        worst = score_feedback.get("worst") if isinstance(score_feedback, Mapping) else None
        iteration = score_feedback.get("iteration") if isinstance(score_feedback, Mapping) else None
        beam_index = score_feedback.get("beam_index") if isinstance(score_feedback, Mapping) else None

        header_bits = []
        if isinstance(iteration, int):
            header_bits.append(f"iteration={iteration}")
        if isinstance(beam_index, int):
            header_bits.append(f"beam={beam_index}")
        header_suffix = f" ({', '.join(header_bits)})" if header_bits else ""

        lines: list[str] = (
            [f"SCORE FEEDBACK EXAMPLES{header_suffix}"]
            + _format_score_example("BEST", best if isinstance(best, Mapping) else None)
            + [""]
            + _format_score_example("WORST", worst if isinstance(worst, Mapping) else None)
        )
        score_feedback_block = "\n".join(lines).strip()

    output_constraints = (
        "Output ONLY the new image prompt (no analysis, no commentary).",
        "Do not include markdown or code fences.",
        "Keep it concrete and renderable.",
    )
    if max_prompt_chars is not None:
        output_constraints = (
            *output_constraints,
            f"Keep the output under {int(max_prompt_chars)} characters.",
        )

    mutation_bar = "MUTATION DIRECTIVE: " + (directive_block if directive_block else "<none>")

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
        "Prefer concrete, renderable specifics; avoid vague filler and clich\u00e9 combinations.",
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
