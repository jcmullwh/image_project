from __future__ import annotations

import json
import textwrap
from typing import Any, Mapping


def profile_abstraction_prompt(*, preferences_guidance: str) -> str:
    profile_text = (preferences_guidance or "").strip()
    if not profile_text:
        raise ValueError("preferences_guidance cannot be empty")

    return textwrap.dedent(
        f"""\
        You rewrite a raw user profile into generator-safe hints.

        Goals:
        - Capture tastes, motifs, and constraints.
        - Make it safe and usable in an image generation prompt.
        - Preserve dislikes/hates as hard avoid constraints.

        Hard rules:
        - Do not copy profile text verbatim.
        - Do not include sensitive personal info or proper nouns.
        - Do not add new preferences not supported by the profile.

        Output:
        - Output ONLY plain text (no JSON, no markdown).
        - Use short bullets and short phrases.

        Raw profile:
        {profile_text}
        """
    ).strip()


def idea_cards_generate_prompt(
    *,
    concepts: list[str],
    generator_profile_hints: str,
    num_ideas: int,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    hints_block = (generator_profile_hints or "").strip() or "<none>"

    return textwrap.dedent(
        f"""\
        You generate {int(num_ideas)} distinct "idea cards" for an image prompt.

        Input:
        - Selected concepts (must be integrated thoughtfully)
        - Optional generator-safe profile hints

        Output MUST be strict JSON ONLY with this schema (no markdown, no comments):
        {{
          "ideas": [
            {{
              "id": "idea_01",
              "hook": "One sentence hook.",
              "narrative": "One paragraph narrative (2-6 sentences).",
              "options": {{
                "composition": ["..."],
                "palette": ["..."],
                "medium": ["..."],
                "mood": ["..."]
              }}
            }}
          ]
        }}

        Rules:
        - Output exactly {int(num_ideas)} ideas.
        - Each idea must be meaningfully different from the others.
        - No proper nouns; no brands; no named people/places.
        - Keep options lists short and concrete (1-4 items each).

        Selected concepts:
        {concepts_block}

        Generator-safe profile hints (optional):
        {hints_block}
        """
    ).strip()


def idea_card_generate_prompt(
    *,
    concepts: list[str],
    generator_profile_hints: str,
    idea_id: str,
    idea_ordinal: int,
    num_ideas: int,
    context_guidance: str | None,
    diversity_directive: str,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    hints_block = (generator_profile_hints or "").strip() or "<none>"
    directive = (diversity_directive or "").strip()
    context_block = (context_guidance or "").strip() or "<none>"

    return textwrap.dedent(
        f"""\
        You generate ONE strict-JSON "idea card" ({idea_id}) for an image prompt.

        Constraints:
        - You MUST integrate all selected concepts into a coherent visual idea.
        - You MUST avoid generic/cliche combinations; make it feel specific and intentional.
        - No proper nouns; no brands; no named people/places.
        - Be concrete and renderable.

        Diversity directive (to keep ideas distinct across the set):
        {directive if directive else "<none>"}

        Context guidance (optional):
        {context_block}

        Selected concepts:
        {concepts_block}

        Generator-safe profile hints (optional):
        {hints_block}

        Output MUST be strict JSON ONLY with this exact schema and nothing else:
        {{
          "id": "{idea_id}",
          "hook": "One sentence hook.",
          "narrative": "One paragraph narrative (2-6 sentences).",
          "options": {{
            "composition": ["..."],
            "palette": ["..."],
            "medium": ["..."],
            "mood": ["..."]
          }}
        }}

        Note:
        - This is idea {int(idea_ordinal)} of {int(num_ideas)}.
        """
    ).strip()


def idea_cards_judge_prompt(
    *,
    concepts: list[str],
    raw_profile: str,
    idea_cards_json: str,
    recent_motif_summary: str | None,
    context_guidance: str | None = None,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    profile_block = (raw_profile or "").strip() or "<none>"
    context_block = (context_guidance or "").strip() or "<none>"
    recent_block = (recent_motif_summary or "").strip()

    return textwrap.dedent(
        f"""\
        You are a strict numeric judge. Score each idea card from 0 to 100.

        Selected concepts (must align strongly):
        {concepts_block}

        Context guidance (optional; reward tasteful use when present):
        {context_block}

        Raw user profile (authoritative preferences; use for judging only):
        {profile_block}

        Ideas JSON (must be strict JSON; do not modify it):
        {idea_cards_json.strip()}

        - Recent motifs summary (penalize repetition when present):
        {recent_block if recent_block else "<none>"}

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
        """.strip()
    )


def final_prompt_from_selected_idea_prompt(
    *,
    concepts: list[str],
    raw_profile: str,
    selected_idea_card: dict[str, object],
    context_guidance: str | None = None,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    idea_json = json.dumps(selected_idea_card, ensure_ascii=False, indent=2)
    context_block = (context_guidance or "").strip()
    return textwrap.dedent(
        f"""
        Create a single final image prompt, based on the selected idea card.

        Output format:
        - Output ONLY the final image prompt. No title, no explanation, no quotes, no markdown.
        - Prefer a high-signal prompt with short labeled lines (omit labels that don't apply).
        - Be concrete: subject/action, setting, lighting, composition/framing, medium/style, palette, textures/materials.
        - Avoid generic adjectives and avoid clich\u00e9 combinations; make it feel like a specific artwork, not a generic scene.

        Preference handling:
        - Dislikes/Hates are avoid constraints.
        - Do NOT "fix" a dislike by adding the disliked thing with positive modifiers (e.g., dislike "wrong/incorrect X" does NOT mean adding "correct X" improves the prompt).

        Selected concepts (must be integrated thoughtfully):
        {concepts_block}

        Context guidance (optional; incorporate subtle seasonal/holiday cues when present):
        {context_block if context_block else "<none>"}

        Raw user profile (authoritative preferences and avoid constraints):
        {raw_profile.strip()}

        Selected idea card (JSON):
        {idea_json}
        """.strip()
    )


def openai_image_prompt_from_selected_idea_prompt(
    *,
    concepts: list[str],
    raw_profile: str,
    selected_idea_card: dict[str, object],
    context_guidance: str | None = None,
    max_chars: int | None = None,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    idea_json = json.dumps(selected_idea_card, ensure_ascii=False, indent=2)
    context_block = (context_guidance or "").strip()
    target = int(max_chars) if max_chars is not None else 3500
    return textwrap.dedent(
        f"""
        You are writing the image prompt text for OpenAI GPT Image 1.5.

        Output rules:
        - Output ONLY the final image prompt (no analysis, no commentary, no JSON/YAML).
        - Use short labeled sections with line breaks; omit any section that doesn't apply.
        - Prefer concrete nouns + renderable constraints; avoid vague hype and redundant synonyms.
        - Your output MUST be fewer than {target} characters.

        Preference handling:
        - Dislikes/Hates are avoid constraints.
        - Do NOT "fix" a dislike by adding the disliked thing with positive modifiers (e.g., dislike "wrong/incorrect X" does NOT mean adding "correct X" improves the prompt).

        Use this rough order (rename freely if it reads better):
        
        1. DELIVERABLE / INTENT

        * What kind of image this is (e.g., "editorial photo", "abstract painting", "UI mockup", "infographic", "logo") and what it should feel like (1 sentence).

        2. CONTENT (works for representational or abstract)

        * If representational: the main entities + actions/poses + key attributes.
        * If abstract/non-representational: the primary forms/motifs (geometry, strokes, textures), relationships (layering, symmetry, repetition, flow), and whether there is *no* recognizable subject matter.

        3. CONTEXT / WORLD (optional)

        * Setting, time, atmosphere, environment rules; or for abstract work: canvas/material, spatial depth, background treatment.

        4. STYLE / MEDIUM

        * Specify the medium (photo, watercolor, vector, 3D render, ink, collage, generative pattern).
        * Add 2-5 concrete style cues tied to visuals (materials, texture, line quality, grain).

        5. COMPOSITION / GEOMETRY

        * Framing/viewpoint (close-up/wide/top-down), perspective/angle, and lighting/mood when relevant.
        * If layout matters, specify placement explicitly ("centered", "negative space left", "text top-right", "balanced margins", "grid with 3 columns").

        6. CONSTRAINTS (be explicit and minimal)

        * MUST INCLUDE: short bullets for non-negotiables.
        * MUST PRESERVE: identity/geometry/layout/brand elements that cannot change (if relevant).
        * MUST NOT INCLUDE: short bullets for exclusions (e.g., "no watermark", "no extra text", "no logos/trademarks").

        7. TEXT IN IMAGE (only if required)

        * Put exact copy in quotes or ALL CAPS.
        * Specify typography constraints (font style, weight, color, size, placement) and demand verbatim rendering with no extra characters.
        * For tricky spellings/brand names: optionally spell the word letter-by-letter.

        Selected concepts (must be integrated thoughtfully):
        {concepts_block}

        Context guidance (optional; incorporate subtle seasonal/holiday cues when present):
        {context_block if context_block else "<none>"}

        Raw user profile (authoritative preferences and avoid constraints):
        {raw_profile.strip()}

        Selected idea card (JSON):
        {idea_json}
        """.strip()
    )


def draft_prompt_from_selected_idea_prompt(
    *,
    concepts: list[str],
    raw_profile: str,
    selected_idea_card: dict[str, object],
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    idea_json = json.dumps(selected_idea_card, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        Create a draft image prompt (Midjourney-style) based on the selected idea card.

        Selected concepts (must be integrated thoughtfully):
        {concepts_block}

        Raw user profile (authoritative preferences and avoid constraints):
        {raw_profile.strip()}

        Selected idea card (JSON):
        {idea_json}

        Output ONLY the draft image prompt. No title, no explanation, no quotes, no markdown.
        """
    ).strip()


def refine_draft_prompt_from_selected_idea_prompt(
    *,
    concepts: list[str],
    raw_profile: str,
    selected_idea_card: dict[str, object],
    draft_prompt: str,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    idea_json = json.dumps(selected_idea_card, ensure_ascii=False, indent=2)
    draft_text = (draft_prompt or "").strip()
    if not draft_text:
        raise ValueError("Draft prompt cannot be empty")
    return textwrap.dedent(
        f"""
        Refine the following draft image prompt into a stronger final Midjourney-style prompt.

        Selected concepts (must be integrated thoughtfully):
        {concepts_block}

        Raw user profile (authoritative preferences and avoid constraints):
        {raw_profile.strip()}

        Selected idea card (JSON):
        {idea_json}

        Refinement block (apply silently; do not output this block):
        - Respect preference strength: Loves are near-mandatory positives; Hates are hard excludes.
        - Do NOT "fix" a dislike by adding the disliked thing with positive modifiers (e.g., dislike "wrong/incorrect X" does NOT mean adding "correct X" improves the prompt).
        - Keep the core intent; improve specificity, coherence, and visual grounding.
        - Remove contradictions and vague phrasing.
        - Keep it concise and generator-friendly.

        Draft prompt:
        {draft_text}

        Output ONLY the final refined image prompt. No title, no explanation, no quotes, no markdown.
        """
    ).strip()


def final_prompt_from_concepts_and_profile_prompt(
    *,
    concepts: list[str],
    raw_profile: str,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    return textwrap.dedent(
        f"""
        Create a single final image prompt (Midjourney-style) using ONLY the provided concepts and user profile.

        Selected concepts (must be integrated thoughtfully):
        {concepts_block}

        Raw user profile (authoritative preferences and avoid constraints):
        {raw_profile.strip()}

        Output ONLY the final image prompt. No title, no explanation, no quotes, no markdown.
        """.strip()
    )

