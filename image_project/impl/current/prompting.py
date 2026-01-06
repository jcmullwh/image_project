"""Prompt strings and prompt-construction helpers.

Main "what prompts run?" source of truth for the current implementation.
"""

from __future__ import annotations

import json
import random
import textwrap

import pandas as pd

DEFAULT_SYSTEM_PROMPT = (
    "You are a highly skilled enclave of Artists trained to generate meaningful, edgy, artistic images on par "
    "with the greatest artists of any time, anywhere, past or future, Earth or any other planet. The enclave "
    "invents unique images that weave together seemingly disparate elements into cohesive wholes that push "
    "boundaries and elicit deep emotions in a human viewer. Keep your responses concise and focused on the task at hand."
)

CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Subject Matter", "Narrative"),
    ("Mood", "Composition", "Perspective"),
    ("Style", "Time Period_Context", "Color Scheme"),
)

def build_preferences_guidance(user_profile: pd.DataFrame) -> str:
    if user_profile is None or user_profile.empty:
        return ""

    sections: list[str] = []
    columns = [str(column).strip() for column in user_profile.columns]
    if "Loves" in columns or "Hates" in columns:
        sections.append(
            textwrap.dedent(
                """\
                Preference strength legend:
                - Loves: will, by itself, make the user like the image.
                - Likes: shows a preference for but not strongly (feedback may be slightly conflicting or not absolute).
                - Dislikes: shows a dislike for but not strongly (feedback may be slightly conflicting or not absolute).
                - Hates: will, by itself, make the user dislike the image.
                """
            ).strip()
        )
    for column in user_profile.columns:
        values = [str(value).strip() for value in user_profile[column].dropna().tolist()]
        values = [value for value in values if value]
        if values:
            sections.append(f"{column}:\n" + "\n".join(f"- {value}" for value in values))

    return "\n\n".join(sections).strip()


def load_prompt_data(file_path):
    # load the prompt data from the csv file
    
    data = pd.read_csv(file_path)
    
    return data


def select_random_concepts(prompt_data: pd.DataFrame, rng: random.Random) -> list[str]:
    selections: list[str] = []
    for group in CONCEPT_GROUPS:
        value = get_random_value_from_group(group, prompt_data, rng)
        if value:
            selections.append(value)
    return selections


def generate_first_prompt(
    prompt_data,
    user_profile,
    rng: random.Random,
    *,
    context_guidance: str | None = None,
    selected_concepts: list[str] | None = None,
):
    preferences_guidance = build_preferences_guidance(user_profile)
    preferences_block = ""
    if preferences_guidance:
        preferences_block = f"\n\nPreferences guidance (authoritative):\n{preferences_guidance}\n"

    context_block = ""
    if context_guidance and str(context_guidance).strip():
        rendered = str(context_guidance).strip()
        if rendered.lower().startswith("context guidance"):
            context_block = f"\n\n{rendered}\n"
        else:
            context_block = f"\n\nContext guidance (optional):\n{rendered}\n"

    preamble = (
        "The enclave's job is to describe an art piece (some form of image, painting, photography, still-frame, etc. displayed in 1792x1024 resolution) "
        "for a specific human, 'Lana'."
        + preferences_block
        + context_block
        + "\nCreate an art piece for Lana that incorporates and thoughtfully blends the below elements. "
    )

    final_lines =  "What are four possible central themes or stories of the art piece and what important messages are each trying to tell the viewer? \
Ensure that your choices are highly sophisticated and nuanced, well integrated with the elements and deeply meaningful to the viewer. \
Ensure that that it is what an AI Artist would find meaningful and important to convey to a human viewer.\
Ensure that the themes and stories are not similar to each other. Ensure that they are not too abstract or conceptual. \
Finally, ensure that they are not boring, cliche, trite, overdone, obvious, or most importantly: milquetoast. Say something and say it with conviction."      

    
    prompt = preamble
    random_values = [str(value).strip() for value in (selected_concepts or []) if str(value).strip()]
    if not random_values:
        random_values = select_random_concepts(prompt_data, rng)

    for value in random_values:
        prompt += f"{value} "

    prompt += final_lines

    return prompt, random_values


def get_random_value_from_group(group, data, rng: random.Random):
    combined_values = []
    for column in group:
        column_values = data[column].dropna().tolist()
        combined_values.extend(column_values)
    return rng.choice(combined_values) if combined_values else None


def generate_second_prompt():
                    
                    
    second_prompt = "Considering each of the four possible choices, what is the consensus on which is the one that is the most compelling, resonant, impactful and cohesive?"
    return second_prompt


def generate_secondB_prompt():        
    second_prompt = "What is the title of the art piece? What is the story of the art piece? What is the role of each of the elements in supporting that theme in a visually cohesive way? \
Try to integrate all elements but if an element is not critical to the theme, do not include it.\
Be very explicit about your description. Do not refer to the elements by name or in abstract/concetual terms. Describe what, in detail, about the art piece evokes or represents the elements. \
What is the story of the piece? What are the layers and the details that cannot be seen in the image? What is the mood? What is the perspective? What is the style? What is the time period? What is the color scheme? What is the subject matter? What is the narrative?\
Somewhere in the image include a loving couple in their 30s. The woman is Asian and the man is \
white with a man-bun and a beard. The couple MUST NOT be the focus of the image. They should be in the background or a corner, ideally barely discernable."
    return second_prompt


def generate_third_prompt():
    
    third_prompt = "What is the most important message you want to convey to the viewer? \
Why does an AI artist find it important to convey this message? \
How could it be more provocative and daring? \
How could it be more radical and trailblazing? \
What is the story that you want to tell and why does that story have depth and dimension? \
Considering your message, the story, the cohesiveness and the visual impact of the art piece you described: \
What are the most important elements of the art piece? \
What detracts from the cohesiveness and impact of your chosen focus for the piece? \
How could you make it stronger and what should be taken away? If an aspect of the art piece is not critical to the message, do not include it (even if it was one of the original elements). \
If something could be added to improve the message, cohesiveness or impact, what would it be?"

    return third_prompt


def generate_thirdB_prompt():
    
    thirdB_prompt = "Is your message clear and unambiguous? \
Is it provocative and daring? \
Is it acheivable in an image? \
How could it be more provocative and daring?\
Do you need to modify it to ensure that it's actually possible to convey in an image?"

    return thirdB_prompt


def generate_fourth_prompt():
    
    fourth_prompt = "considering the whole discussion, provide a concise description of the piece, in detail, for submission an image generation AI.\
Integrate Narrative and Visuals: When crafting a prompt, intertwine subtle narrative themes with concrete visual elements that can serve as symbolic representations of the narrative.\
Use Implicit Narratives: Incorporate rich and specific visual details that suggest the narrative. This allows the AI to construct a visual story without needing a detailed narrative explanation.\
Prioritize Detail Placement: Position the most significant visual details at the beginning of the prompt to ensure prominence. Utilize additional details to enrich the scene as the prompt progresses.\
Employ Thematic Symbolism: Include symbols and motifs that are universally associated with the narrative theme to provide clear guidance to the AI, while still leaving room for creative interpretation.\
Incorporate Action and Emotion: Utilize verbs that convey action and emotion relevant to the narrative to infuse the images with energy and affective depth.\
Layer Information: Construct prompts with multiple layers of information, blending abstract concepts with detailed visuals to provide the AI with a rich foundation for image creation.\
Emphasize Style and Color: When style and color are important, mention them explicitly and weave them into the description of the key elements to ensure they are reflected in the image.\
Reiterate Important Concepts: If certain concepts or themes are crucial to the prompt's intent, find ways to subtly reiterate them without being redundant. This can help ensure their presence is captured in the generated image.\
Use Action and Emotion Words: When describing scenes or elements, use verbs and adjectives that evoke emotion or action, as these can help the AI generate more dynamic and engaging images."
    return fourth_prompt


def generate_image_prompt():
    image_prompt = textwrap.dedent(
        """\
        You are writing the *image prompt text* for GPT Image 1.5. Output ONLY the prompt (no analysis, no YAML/JSON unless asked). Use short labeled sections with line breaks; omit any section that doesnâ€™t apply (do not force a â€œsubjectâ€ if the request is abstract or pattern-based). Follow the guidance but do not over-fit if it clashes with your specific image.

        Your output MUST be fewer than 3500 characters.

        Use this order (rename freely if it reads better for the task):

        1. DELIVERABLE / INTENT

        * What kind of image this is (e.g., â€œeditorial photoâ€, â€œabstract paintingâ€, â€œUI mockupâ€, â€œinfographicâ€, â€œlogoâ€) and what it should feel like (1 sentence).

        2. CONTENT (works for representational or abstract)

        * If representational: the main entities + actions/poses + key attributes.
        * If abstract/non-representational: the primary forms/motifs (geometry, strokes, textures), relationships (layering, symmetry, repetition, flow), and whether there is *no* recognizable subject matter.

        3. CONTEXT / WORLD (optional)

        * Setting, time, atmosphere, environment rules; or for abstract work: canvas/material, spatial depth, background treatment.

        4. STYLE / MEDIUM

        * Specify the medium (photo, watercolor, vector, 3D render, ink, collage, generative pattern).
        * Add 2â€“5 concrete style cues tied to visuals (materials, texture, line quality, grain).

        5. COMPOSITION / GEOMETRY

        * Framing/viewpoint (close-up/wide/top-down), perspective/angle, and lighting/mood when relevant.
        * If layout matters, specify placement explicitly (â€œcenteredâ€, â€œnegative space leftâ€, â€œtext top-rightâ€, â€œbalanced marginsâ€, â€œgrid with 3 columnsâ€).

        6. CONSTRAINTS (be explicit and minimal)

        * MUST INCLUDE: short bullets for non-negotiables.
        * MUST PRESERVE: identity/geometry/layout/brand elements that cannot change (if relevant).
        * MUST NOT INCLUDE: short bullets for exclusions (e.g., â€œno watermarkâ€, â€œno extra textâ€, â€œno logos/trademarksâ€).

        7. TEXT IN IMAGE (only if required)

        * Put exact copy in quotes or ALL CAPS.
        * Specify typography constraints (font style, weight, color, size, placement) and demand verbatim rendering with no extra characters.
        * For tricky spellings/brand names: optionally spell the word letter-by-letter.

        8. MULTI-IMAGE REFERENCES (only if applicable)

        * â€œImage 1: â€¦â€, â€œImage 2: â€¦â€ describing what each input is.
        * State precisely how they interact (â€œapply Image 2â€™s style to Image 1â€; â€œplace the object from Image 1 into Image 2 at â€¦â€; â€œmatch lighting/perspective/scaleâ€).

        General rules:

        * Prefer concrete nouns + measurable adjectives (â€œmatte ceramicâ€, â€œsoft diffuse lightâ€, â€œthin ink lineâ€) over vague hype (â€œstunningâ€, â€œmasterpieceâ€).
        * Avoid long grab-bags of synonyms. One requirement per line; no contradictions.
        * If you need â€œclean/minimal,â€ specify what that means visually (few elements, large negative space, limited palette, simple shapes).
        """
    ).strip()
    return image_prompt

def generate_fifth_prompt():
        
    fifth_prompt = "Review and Refine: Review the concise description to ensure that it flows logically, with the most critical elements front and center. Refine any parts that may lead to ambiguity or that do not directly serve the prompt's core intent.\
        we are creating a prompt for midjourney but be sure not to compromise on the vision. There are several important things that must be emphasized when prompting midjourney.\
	-  Midjourney Bot works best with simple, short sentences that describe what you want to see. Avoid long lists of requests. Instead of: Show me a picture of lots of blooming California poppies, make them bright, vibrant orange, and draw them in an illustrated style with colored pencils Try: Bright orange California poppies drawn with colored pencils- Your prompt must be very direct, simple, and succinct.\
	- The Midjourney Bot does not understand grammar, sentence structure, or words like humans. Word choice also matters. More specific synonyms work better in many circumstances. Instead of big, try gigantic, enormous, or immense. Remove words when possible. Fewer words mean each word has a more powerful influence. Use commas, brackets, and hyphens to help organize your thoughts, but know the Midjourney Bot will not reliably interpret them. The Midjourney Bot does not consider capitalization.\
	- Try to be clear about any context or details that are important to you. Think about:\
Subject: person, animal, character, location, object, etc.\
Medium: photo, painting, illustration, sculpture, doodle, tapestry, etc.\
Environment: indoors, outdoors, on the moon, in Narnia, underwater, the Emerald City, etc.\
Lighting: soft, ambient, overcast, neon, studio lights, etc\
Color: vibrant, muted, bright, monochromatic, colorful, black and white, pastel, etc.\
Mood: Sedate, calm, raucous, energetic, etc.\
Composition: Portrait, headshot, closeup, birds-eye view, etc.\
	- words and concepts that need emphasis may be repeated.\
	- you may add a double colon :: to a prompt indicates to the Midjourney Bot that it should consider each part of the prompt individually. For the prompt space ship both words are considered together, and the Midjourney Bot produces images of sci-fi spaceships. If the prompt is separated into two parts, space:: ship, both concepts are considered separately, then blended together creating a sailing ship traveling through space."
    return fifth_prompt


def generate_sixth_prompt():
        
    seventh_prompt = "Review and Refine Again: Review the concise description again to ensure that it flows logically, with the most critical elements front and center. Refine any parts that may lead to ambiguity or that do not directly serve the prompt's core intent. \
 remember we are creating a prompt for midjourney but be sure not to compromise on the vision. There are several important things that must be emphasized when prompting midjourney.\
	-  Midjourney Bot works best with simple, short sentences that describe what you want to see. Avoid long lists of requests. Instead of: Show me a picture of lots of blooming California poppies, make them bright, vibrant orange, and draw them in an illustrated style with colored pencils Try: Bright orange California poppies drawn with colored pencils- Your prompt must be very direct, simple, and succinct.\
	- The Midjourney Bot does not understand grammar, sentence structure, or words like humans. Word choice also matters. More specific synonyms work better in many circumstances. Instead of big, try gigantic, enormous, or immense. Remove words when possible. Fewer words mean each word has a more powerful influence. Use commas, brackets, and hyphens to help organize your thoughts, but know the Midjourney Bot will not reliably interpret them. The Midjourney Bot does not consider capitalization.\
	- Try to be clear about any context or details that are important to you. Think about:\
Subject: person, animal, character, location, object, etc.\
Medium: photo, painting, illustration, sculpture, doodle, tapestry, etc.\
Environment: indoors, outdoors, on the moon, in Narnia, underwater, the Emerald City, etc.\
Lighting: soft, ambient, overcast, neon, studio lights, etc\
Color: vibrant, muted, bright, monochromatic, colorful, black and white, pastel, etc.\
Mood: Sedate, calm, raucous, energetic, etc.\
Composition: Portrait, headshot, closeup, birds-eye view, etc.\
	- words and concepts that need emphasis may be repeated.\
	- you may add a double colon :: to a prompt indicates to the Midjourney Bot that it should consider each part of the prompt individually. For the prompt space ship both words are considered together, and the Midjourney Bot produces images of sci-fi spaceships. If the prompt is separated into two parts, space:: ship, both concepts are considered separately, then blended together creating a sailing ship traveling through space.\
This time, provide only the final prompt to the AI. Do not include anything except the final prompt in your response."
    return seventh_prompt


def refine_image_prompt_prompt(draft: str) -> str:
    draft_text = (draft or "").strip()
    if not draft_text:
        raise ValueError("Draft image prompt cannot be empty")

    return textwrap.dedent(
        f"""\
        Refine the following draft into a high-quality GPT Image 1.5 prompt.

        Goals:
        - Preserve the original intent and key details.
        - Remove contradictions, redundancies, and vague phrasing.
        - Make the prompt concrete and visually grounded.
        - Keep it under 3500 characters.

        Output rules:
        - Output ONLY the final image prompt (no analysis, no commentary).
        - Use short labeled sections with line breaks (omit sections that don't apply).

        Draft:
        {draft_text}
        """
    ).strip()


def profile_abstraction_prompt(*, preferences_guidance: str) -> str:
    guidance = (preferences_guidance or "").strip()
    return textwrap.dedent(
        f"""
        Create a generator-safe profile summary that captures broad tastes without overfitting.

        Rules:
        - Allowed: broad adjectives, high-level style constraints, general "avoid" constraints.
        - Disallowed: explicit colors, named motifs, named artists, and repeated n-grams copied from the raw likes list.
        - Keep it short (3-8 bullet points max). No prose beyond the bullets.

        Raw profile (for you only; do not copy phrases verbatim):
        {guidance}
        """.strip()
    )


def idea_cards_generate_prompt(
    *,
    concepts: list[str],
    generator_profile_hints: str,
    num_ideas: int,
) -> str:
    from image_project.framework.scoring import expected_idea_ids  # local import to keep prompting.py lightweight

    ids = expected_idea_ids(num_ideas)
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    hints = (generator_profile_hints or "").strip()
    return textwrap.dedent(
        f"""
        Generate {num_ideas} distinct image "idea cards" using ONLY the selected concepts and the generator-safe profile hints.

        Selected concepts:
        {concepts_block}

        Generator-safe profile hints:
        {hints if hints else "<none>"}

        Output must be STRICT JSON only (no markdown, no commentary), with this exact schema:
        {{
          "ideas": [
            {{
              "id": "A",
              "hook": "One sentence pitch.",
              "narrative": "Short paragraph.",
              "options": {{
                "composition": ["...", "..."],
                "palette": ["...", "..."],
                "medium": ["..."],
                "mood": ["..."]
              }},
              "avoid": ["..."]
            }}
          ]
        }}

        Hard constraints:
        - You MUST produce exactly {num_ideas} ideas with ids exactly: {json.dumps(ids)}.
        - Each options list must contain meaningful, non-synonym variation.
        - composition and palette lists must have at least 2 items.
        - medium and mood lists must have at least 1 item.
        - "avoid" may be omitted or an empty list.
        """.strip()
    )


def idea_card_generate_prompt(
    *,
    concepts: list[str],
    generator_profile_hints: str,
    idea_id: str,
    idea_ordinal: int | None = None,
    num_ideas: int,
    context_guidance: str | None = None,
    diversity_directive: str | None = None,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    hints = (generator_profile_hints or "").strip()
    context_block = (context_guidance or "").strip()
    directive_block = (diversity_directive or "").strip()
    ordinal = int(idea_ordinal) if idea_ordinal is not None else None
    return textwrap.dedent(
        f"""
        Generate ONE image "idea card" as strict JSON.

        This is idea id {idea_id!r} ({ordinal if ordinal is not None else "?"} of {num_ideas}). Your JSON MUST include "id": "{idea_id}".

        Goal:
        - Produce a bold, specific, non-generic idea with clear visual anchors.
        - Avoid vague filler ("beautiful", "stunning", "mystical", "epic") and avoid cliché combinations.
        - Make this idea feel meaningfully different in approach (composition/setting/medium/narrative) from other ids in the same batch.

        Diversity directive (apply it without contradicting selected concepts):
        {directive_block if directive_block else "<none>"}

        Selected concepts:
        {concepts_block}

        Context guidance (optional):
        {context_block if context_block else "<none>"}

        Generator-safe profile hints:
        {hints if hints else "<none>"}

        Output must be STRICT JSON only (no markdown, no commentary), with this exact schema:
        {{
          "id": "{idea_id}",
          "hook": "One sentence pitch.",
          "narrative": "Short paragraph.",
          "options": {{
            "composition": ["...", "..."],
            "palette": ["...", "..."],
            "medium": ["..."],
            "mood": ["..."]
          }},
          "avoid": ["..."]
        }}

        Hard constraints:
        - id MUST be exactly "{idea_id}".
        - Each options list must contain meaningful, non-synonym variation (materially different choices).
        - Composition options should be concrete and camera/composition-aware (e.g., framing, angle, lens, negative space).
        - Palette options should be concrete (named colors + lighting temperature, not just "vibrant").
        - composition and palette lists must have at least 2 items.
        - medium and mood lists must have at least 1 item.
        - "avoid" may be omitted or an empty list, but if present it should be specific (e.g., overused motifs, compositional pitfalls, disliked vibes).
        """.strip()
    )


def idea_cards_judge_prompt(
    *,
    concepts: list[str],
    raw_profile: str,
    idea_cards_json: str,
    recent_motif_summary: str | None,
    context_guidance: str | None = None,
) -> str:
    concepts_block = "\n".join(f"- {c}" for c in concepts) if concepts else "- <none>"
    recent_block = (recent_motif_summary or "").strip()
    context_block = (context_guidance or "").strip()
    return textwrap.dedent(
        f"""
        You are a strict numeric judge. Score each candidate idea card from 0 to 100.

        Inputs:
        - Selected concepts (must align strongly):
        {concepts_block}

        - Context guidance (optional; reward tasteful use when present):
        {context_block if context_block else "<none>"}

        - Raw user profile (use for judging only; do not rewrite it):
        {raw_profile.strip()}

        - Candidate idea cards JSON:
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
        - Avoid generic adjectives and avoid cliché combinations; make it feel like a specific artwork, not a generic scene.

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

        Use this rough order (rename freely if it reads better):
        1) DELIVERABLE / INTENT
        2) CONTENT
        3) CONTEXT / WORLD (optional)
        4) STYLE / MEDIUM
        5) COMPOSITION / GEOMETRY
        6) CONSTRAINTS
           - MUST INCLUDE:
           - MUST NOT INCLUDE:
        7) TEXT IN IMAGE (only if required)

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

import sys
from dataclasses import dataclass
from typing import Any, Callable

from image_project.framework.runtime import RunContext
from image_project.framework.prompting import (
    ActionStageSpec,
    PlanInputs,
    StageNodeSpec,
    StageSpec,
)

prompts = sys.modules[__name__]

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


def list_stages() -> None:
    for entry in StageCatalog.describe():
        doc = entry.get("doc") or ""
        print(f"{entry['stage_id']}\t{doc}")


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
        from image_project.framework.inputs import apply_concept_filters, make_dislike_rewrite_filter

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


@StageCatalog.register(
    "blackbox.prepare",
    doc="Prepare blackbox scoring (novelty summary + default generator hints).",
    source="blackbox_scoring.extract_recent_motif_summary",
    tags=("blackbox",),
)
def blackbox_prepare(inputs: PlanInputs) -> ActionStageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring

    def _action(ctx: RunContext) -> dict[str, Any]:
        from image_project.framework import scoring as blackbox_scoring

        if not scoring_cfg.enabled:
            raise ValueError("blackbox.prepare requires prompt.scoring.enabled=true")

        novelty_enabled_cfg = bool(scoring_cfg.novelty.enabled and scoring_cfg.novelty.window > 0)
        novelty_enabled_effective = bool(novelty_enabled_cfg)
        ctx.logger.info(
            "Blackbox scoring enabled: num_ideas=%d, exploration_rate=%.3g, novelty=%s",
            scoring_cfg.num_ideas,
            scoring_cfg.exploration_rate,
            novelty_enabled_cfg,
        )

        novelty_summary: dict[str, Any]
        if novelty_enabled_cfg:
            try:
                novelty_summary = blackbox_scoring.extract_recent_motif_summary(
                    generations_csv_path=ctx.cfg.generations_csv_path,
                    novelty_cfg=scoring_cfg.novelty,
                )
            except Exception as exc:
                ctx.logger.warning(
                    "Novelty enabled but history unavailable; disabling novelty for this run: %s %s",
                    ctx.cfg.generations_csv_path,
                    exc,
                )
                novelty_enabled_effective = False
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
        ctx.blackbox_scoring["novelty"] = {
            "method": scoring_cfg.novelty.method,
            "window": int(scoring_cfg.novelty.window),
            "enabled_cfg": bool(novelty_enabled_cfg),
            "enabled_effective": bool(novelty_enabled_effective),
        }

        # Default generator hints (may be overwritten by profile_abstraction stage).
        ctx.outputs["generator_profile_hints"] = str(ctx.outputs.get("preferences_guidance") or "")

        return {
            "novelty_enabled": bool(novelty_enabled_effective),
            "novelty_window": int(scoring_cfg.novelty.window),
            "novelty_method": scoring_cfg.novelty.method,
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
    "blackbox.profile_hints_load",
    doc="Load generator-safe profile hints from a file.",
    source="framework.profile_io.load_generator_profile_hints",
    tags=("blackbox",),
)
def blackbox_profile_hints_load(_inputs: PlanInputs) -> ActionStageSpec:
    def _action(ctx: RunContext) -> str:
        hints_path = ctx.cfg.prompt_scoring.generator_profile_hints_path
        if not hints_path:
            raise ValueError(
                "blackbox.profile_hints_load requires prompt.scoring.generator_profile_hints_path"
            )

        from image_project.framework.profile_io import load_generator_profile_hints

        hints = load_generator_profile_hints(hints_path)
        if not isinstance(hints, str) or not hints.strip():
            raise ValueError(f"Generator profile hints file was empty: {hints_path}")

        ctx.logger.info(
            "Loaded generator profile hints from %s (chars=%d)", hints_path, len(hints)
        )
        return hints

    return ActionStageSpec(
        stage_id="blackbox.profile_hints_load",
        fn=_action,
        merge="none",
        output_key="generator_profile_hints",
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
        idea_profile_source = scoring_cfg.idea_profile_source
        if idea_profile_source == "none":
            hints = ""
        elif idea_profile_source in ("raw", "generator_hints", "generator_hints_plus_dislikes"):
            hints = _resolve_blackbox_profile_text(
                ctx,
                source=idea_profile_source,
                stage_id="blackbox.idea_cards_generate",
                config_path="prompt.scoring.idea_profile_source",
            )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(
                "Unknown prompt.scoring.idea_profile_source: "
                f"{idea_profile_source!r} (expected: raw|generator_hints|generator_hints_plus_dislikes|none)"
            )
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


def build_blackbox_isolated_idea_card_specs(inputs: PlanInputs) -> list[StageNodeSpec]:
    from image_project.framework.scoring import expected_idea_ids

    scoring_cfg = inputs.cfg.prompt_scoring
    idea_ids = expected_idea_ids(scoring_cfg.num_ideas)
    context_guidance = inputs.context_guidance or None

    diversity_directives: tuple[str, ...] = (
        # Explicit wildcards / escape hatches
        "Develop your strongest interpretation of how these components fit together.",
        "Follow the most compelling idea that emerges from these components, even if it doesn’t match a clear pattern.",
        # Structured but non-prescriptive lenses
        "Synthesize the components into a single coherent idea, prioritizing internal logic over novelty.",
        "Reinterpret the role or meaning of one component while keeping all components recognizable.",
        "Let one component strongly shape how the others are understood or used.",
        "Treat all components as equally fundamental, without an obvious focal element.",
        # Semi-open exploration
        "Explore an interpretation that feels unexpected but still defensible given the components.",
        "Look for a non-obvious relationship or alignment between the components.",

    )

    def _resolve_profile_hints(ctx: RunContext, *, stage_id: str) -> str:
        idea_profile_source = scoring_cfg.idea_profile_source
        if idea_profile_source == "none":
            return ""
        if idea_profile_source == "raw":
            return str(ctx.outputs.get("preferences_guidance") or "")
        if idea_profile_source == "generator_hints":
            hints = ctx.outputs.get("generator_profile_hints")
            if not isinstance(hints, str) or not hints.strip():
                return str(ctx.outputs.get("preferences_guidance") or "")
            return hints
        if idea_profile_source == "generator_hints_plus_dislikes":
            return _resolve_blackbox_profile_text(
                ctx,
                source=idea_profile_source,
                stage_id=stage_id,
                config_path="prompt.scoring.idea_profile_source",
            )
        raise ValueError(
            f"Unknown prompt.scoring.idea_profile_source for {stage_id}: "
            f"{idea_profile_source!r} (expected: raw|generator_hints|generator_hints_plus_dislikes|none)"
        )

    specs: list[StageNodeSpec] = []
    for idea_ordinal, idea_id in enumerate(idea_ids, start=1):
        stage_id = f"blackbox.idea_card_generate.{idea_id}"
        output_key = f"blackbox.idea_card.{idea_id}.json"
        directive = diversity_directives[(idea_ordinal - 1) % len(diversity_directives)]

        def _prompt(
            ctx: RunContext,
            *,
            idea_id=idea_id,
            idea_ordinal=idea_ordinal,
            directive=directive,
            stage_id=stage_id,
            context_guidance=context_guidance,
        ) -> str:
            hints = _resolve_profile_hints(ctx, stage_id=stage_id)
            return prompts.idea_card_generate_prompt(
                concepts=list(ctx.selected_concepts),
                generator_profile_hints=str(hints or ""),
                idea_id=idea_id,
                idea_ordinal=idea_ordinal,
                num_ideas=scoring_cfg.num_ideas,
                context_guidance=context_guidance,
                diversity_directive=directive,
            )

        specs.append(
            StageSpec(
                stage_id=stage_id,
                prompt=_prompt,
                temperature=0.8,
                merge="none",
                output_key=output_key,
                refinement_policy="none",
                tags=("blackbox",),
                doc="Generate one idea card (strict JSON).",
                source="prompts.idea_card_generate_prompt",
            )
        )

    def _assemble(ctx: RunContext, *, idea_ids=idea_ids) -> str:
        import json

        from image_project.framework import scoring as blackbox_scoring

        ideas: list[dict[str, Any]] = []
        for idea_id in idea_ids:
            key = f"blackbox.idea_card.{idea_id}.json"
            raw = ctx.outputs.get(key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Missing required output: {key}")
            try:
                ideas.append(blackbox_scoring.parse_idea_card_json(raw, expected_id=idea_id))
            except Exception as exc:
                setattr(exc, "pipeline_step", f"blackbox.idea_card_generate.{idea_id}")
                setattr(exc, "pipeline_path", f"pipeline/blackbox.idea_card_generate.{idea_id}/draft")
                raise

        return json.dumps({"ideas": ideas}, ensure_ascii=False, indent=2)

    specs.append(
        ActionStageSpec(
            stage_id="blackbox.idea_cards_assemble",
            fn=_assemble,
            merge="none",
            output_key="idea_cards_json",
            tags=("blackbox",),
            doc="Assemble per-idea JSON outputs into a combined idea_cards_json payload.",
            source="blackbox_scoring.parse_idea_card_json",
        )
    )

    return specs


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
    context_guidance = inputs.context_guidance or None

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
            context_guidance=context_guidance,
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

        from image_project.framework import scoring as blackbox_scoring

        if not scoring_cfg.enabled:
            raise ValueError("blackbox.select_idea_card requires prompt.scoring.enabled=true")

        idea_cards_json = ctx.outputs.get("idea_cards_json")
        if not isinstance(idea_cards_json, str) or not idea_cards_json.strip():
            raise ValueError("Missing required output: idea_cards_json")
        idea_scores_json = ctx.outputs.get("idea_scores_json")
        if not isinstance(idea_scores_json, str) or not idea_scores_json.strip():
            raise ValueError("Missing required output: idea_scores_json")

        novelty_cfg = scoring_cfg.novelty
        novelty_enabled_cfg = bool(novelty_cfg.enabled and novelty_cfg.window > 0)
        novelty_summary: dict[str, Any] | None = None
        novelty_available = False
        if ctx.blackbox_scoring is not None:
            raw = ctx.blackbox_scoring.get("novelty_summary")
            if isinstance(raw, dict):
                novelty_summary = raw
                novelty_available = bool(raw.get("enabled"))

        novelty_missing = bool(novelty_enabled_cfg and not novelty_available)
        if ctx.blackbox_scoring is None:
            ctx.blackbox_scoring = {}
        novelty_meta = ctx.blackbox_scoring.get("novelty")
        if not isinstance(novelty_meta, dict):
            novelty_meta = {}
            ctx.blackbox_scoring["novelty"] = novelty_meta
        novelty_meta.update(
            {
                "method": novelty_cfg.method,
                "window": int(novelty_cfg.window),
                "enabled_cfg": bool(novelty_enabled_cfg),
                "summary_available": bool(novelty_available),
                "missing_summary": bool(novelty_missing),
            }
        )

        if novelty_missing:
            warn = (
                "pipeline/blackbox.select_idea_card/action: prompt.scoring.novelty.enabled=true but novelty summary is "
                "unavailable; novelty penalties disabled for this run"
            )
            ctx.logger.warning(warn)
            warnings_log = ctx.blackbox_scoring.get("warnings")
            if not isinstance(warnings_log, list):
                warnings_log = []
                ctx.blackbox_scoring["warnings"] = warnings_log
            warnings_log.append(warn)

        try:
            idea_cards = blackbox_scoring.parse_idea_cards_json(
                idea_cards_json, expected_num_ideas=scoring_cfg.num_ideas
            )
        except Exception as exc:
            pipeline = ctx.outputs.get("prompt_pipeline")
            resolved = pipeline.get("resolved_stages") if isinstance(pipeline, dict) else None
            if isinstance(resolved, list) and "blackbox.idea_cards_assemble" in resolved:
                setattr(exc, "pipeline_step", "blackbox.idea_cards_assemble")
                setattr(exc, "pipeline_path", "pipeline/blackbox.idea_cards_assemble/action")
            else:
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
            novelty_cfg=novelty_cfg,
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
    context_guidance = inputs.context_guidance or None

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
            context_guidance=context_guidance,
        )

    return StageSpec(
        stage_id="blackbox.image_prompt_creation",
        prompt=_prompt,
        temperature=0.8,
        is_default_capture=True,
    )


@StageCatalog.register(
    "blackbox.image_prompt_openai",
    doc="Create an OpenAI (GPT Image 1.5) formatted prompt from the selected idea card.",
    source="prompts.openai_image_prompt_from_selected_idea_prompt",
    tags=("blackbox",),
)
def blackbox_image_prompt_openai(inputs: PlanInputs) -> StageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring
    final_profile_source = scoring_cfg.final_profile_source
    context_guidance = inputs.context_guidance or None
    max_chars: int | None = None
    if inputs.cfg.prompt_blackbox_refine is not None:
        max_chars = inputs.cfg.prompt_blackbox_refine.max_prompt_chars

    def _prompt(ctx: RunContext) -> str:
        selected_card = ctx.outputs.get("selected_idea_card")
        if not isinstance(selected_card, dict):
            raise ValueError("Missing required output: selected_idea_card")
        return prompts.openai_image_prompt_from_selected_idea_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=_resolve_blackbox_profile_text(
                ctx,
                source=final_profile_source,
                stage_id="blackbox.image_prompt_openai",
                config_path="prompt.scoring.final_profile_source",
            ),
            selected_idea_card=selected_card,
            context_guidance=context_guidance,
            max_chars=max_chars,
        )

    return StageSpec(
        stage_id="blackbox.image_prompt_openai",
        prompt=_prompt,
        temperature=0.8,
        is_default_capture=True,
    )


@StageCatalog.register(
    "blackbox.image_prompt_draft",
    doc="Create a draft prompt from selected idea card (for downstream refinement).",
    source="prompts.draft_prompt_from_selected_idea_prompt",
    tags=("blackbox",),
)
def blackbox_image_prompt_draft(inputs: PlanInputs) -> StageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring
    final_profile_source = scoring_cfg.final_profile_source

    def _prompt(ctx: RunContext) -> str:
        selected_card = ctx.outputs.get("selected_idea_card")
        if not isinstance(selected_card, dict):
            raise ValueError("Missing required output: selected_idea_card")
        return prompts.draft_prompt_from_selected_idea_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=_resolve_blackbox_profile_text(
                ctx,
                source=final_profile_source,
                stage_id="blackbox.image_prompt_draft",
                config_path="prompt.scoring.final_profile_source",
            ),
            selected_idea_card=selected_card,
        )

    return StageSpec(
        stage_id="blackbox.image_prompt_draft",
        prompt=_prompt,
        temperature=0.8,
        merge="none",
        output_key="blackbox_draft_image_prompt",
        refinement_policy="none",
    )


@StageCatalog.register(
    "blackbox.image_prompt_refine",
    doc="Refine the draft prompt into a final prompt (no ToT).",
    source="prompts.refine_draft_prompt_from_selected_idea_prompt",
    tags=("blackbox",),
)
def blackbox_image_prompt_refine(inputs: PlanInputs) -> StageSpec:
    scoring_cfg = inputs.cfg.prompt_scoring
    final_profile_source = scoring_cfg.final_profile_source

    def _prompt(ctx: RunContext) -> str:
        selected_card = ctx.outputs.get("selected_idea_card")
        if not isinstance(selected_card, dict):
            raise ValueError("Missing required output: selected_idea_card")

        draft = ctx.outputs.get("blackbox_draft_image_prompt")
        if not isinstance(draft, str) or not draft.strip():
            raise ValueError("Missing required output: blackbox_draft_image_prompt")

        return prompts.refine_draft_prompt_from_selected_idea_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=_resolve_blackbox_profile_text(
                ctx,
                source=final_profile_source,
                stage_id="blackbox.image_prompt_refine",
                config_path="prompt.scoring.final_profile_source",
            ),
            selected_idea_card=selected_card,
            draft_prompt=draft,
        )

    return StageSpec(
        stage_id="blackbox.image_prompt_refine",
        prompt=_prompt,
        temperature=0.4,
        refinement_policy="none",
        is_default_capture=True,
    )


@StageCatalog.register(
    "direct.image_prompt_creation",
    doc="Create final prompt directly from concepts + profile.",
    source="prompts.final_prompt_from_concepts_and_profile_prompt",
    tags=("direct",),
)
def direct_image_prompt_creation(_inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        if not ctx.selected_concepts:
            raise ValueError(
                "direct.image_prompt_creation requires selected concepts; "
                "run preprompt.select_concepts first (or include it in the plan)."
            )

        return prompts.final_prompt_from_concepts_and_profile_prompt(
            concepts=list(ctx.selected_concepts),
            raw_profile=str(ctx.outputs.get("preferences_guidance") or ""),
        )

    return StageSpec(
        stage_id="direct.image_prompt_creation",
        prompt=_prompt,
        temperature=0.8,
        is_default_capture=True,
        refinement_policy="none",
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


@StageCatalog.register(
    "ab.random_token",
    doc="Generate a deterministic per-run random token.",
    source="inline",
    tags=("ab",),
)
def ab_random_token(_inputs: PlanInputs) -> ActionStageSpec:
    def _action(ctx: RunContext) -> str:
        roll = ctx.rng.randint(100000, 999999)
        return f"RV-{ctx.seed}-{roll}"

    return ActionStageSpec(
        stage_id="ab.random_token",
        fn=_action,
        merge="none",
        output_key="ab_random_token",
    )


@StageCatalog.register(
    "ab.scene_draft",
    doc="Create a scene draft from a random token.",
    source="prompts.ab_scene_draft",
    tags=("ab",),
)
def ab_scene_draft(_inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        token = ctx.outputs.get("ab_random_token")
        token_text = str(token).strip() if token is not None else ""
        if not token_text:
            raise ValueError("Missing required output: ab_random_token")

        return textwrap.dedent(
            f"""\
            You are drafting a cinematic scene description that will later be converted into an image generation prompt.

            Required token: {token_text}

            Requirements:
            - Include the token verbatim as visible text in the scene (e.g., on a sign, label, screen, tattoo, receipt).
            - Describe a single coherent moment (no montages).
            - Be concrete and visual: subject, setting, action, lighting, mood, camera/framing.
            - Avoid cliches and generic phrasing.

            Output:
            - Return ONLY the scene description as 4-6 sentences. No headings, no bullets.
            """
        ).strip()

    return StageSpec(
        stage_id="ab.scene_draft",
        prompt=_prompt,
        temperature=0.85,
        output_key="ab_scene_draft",
    )


def _ab_require_text_output(ctx: RunContext, key: str) -> str:
    value = ctx.outputs.get(key)
    if not isinstance(value, str):
        value = str(value or "")
    text = value.strip()
    if not text:
        raise ValueError(f"Missing required output: {key}")
    return text


@StageCatalog.register(
    "ab.scene_refine_no_block",
    doc="Refine the draft scene with a minimal instruction set.",
    source="prompts.ab_scene_refine_no_block",
    tags=("ab",),
)
def ab_scene_refine_no_block(_inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        token = _ab_require_text_output(ctx, "ab_random_token")
        draft = _ab_require_text_output(ctx, "ab_scene_draft")
        return textwrap.dedent(
            f"""\
            Refine the following scene draft for an image prompt.

            Constraints:
            - Keep the required token verbatim somewhere as visible text in the scene: {token}
            - Make the scene more specific, vivid, and visually grounded.

            Draft:
            {draft}

            Output:
            - Return ONLY the revised scene description (4-6 sentences). No headings, no bullets.
            """
        ).strip()

    return StageSpec(
        stage_id="ab.scene_refine_no_block",
        prompt=_prompt,
        temperature=0.75,
        output_key="ab_scene_refined",
    )


@StageCatalog.register(
    "ab.scene_refine_with_block",
    doc="Refine the draft scene with an explicit refinement block.",
    source="prompts.ab_scene_refine_with_block",
    tags=("ab",),
)
def ab_scene_refine_with_block(_inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        token = _ab_require_text_output(ctx, "ab_random_token")
        draft = _ab_require_text_output(ctx, "ab_scene_draft")
        return textwrap.dedent(
            f"""\
            Refine the following scene draft for an image prompt.

            Required token (must remain verbatim as visible text in the scene): {token}

            Refinement block (apply silently; do not output this block):
            - Subject: make the main subject unmistakable and unique (no generic "a person").
            - Setting: make time/place concrete (materials, era, weather, geography, props).
            - Composition: specify framing, lens feel, depth of field, foreground/background.
            - Lighting: specify the dominant light source(s) and the mood they create.
            - Color: specify a palette and a couple accent colors.
            - Specificity: add 3+ grounded details (textures, signage, wear, reflections, particles).
            - Token: integrate the token diegetically (a label, screen, tag), not as metadata.
            - Cliche filter: remove stock phrases and overused tropes.

            Draft:
            {draft}

            Output:
            - Return ONLY the revised scene description (4-6 sentences). No headings, no bullets.
            """
        ).strip()

    return StageSpec(
        stage_id="ab.scene_refine_with_block",
        prompt=_prompt,
        temperature=0.75,
        output_key="ab_scene_refined",
    )


@StageCatalog.register(
    "ab.scene_spec_json",
    doc="Convert the scene draft into a strict SceneSpec JSON intermediary.",
    source="inline",
    tags=("ab",),
)
def ab_scene_spec_json(_inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        token = _ab_require_text_output(ctx, "ab_random_token")
        draft = _ab_require_text_output(ctx, "ab_scene_draft")

        return textwrap.dedent(
            f"""\
            Convert the scene draft into a strict SceneSpec JSON object.

            Required token (must appear as visible text in the scene): {token}

            Scene draft:
            {draft}

            SceneSpec schema (required keys):
            {{
              "subject": "...",
              "setting": "...",
              "action": "...",
              "composition": "...",
              "camera": "...",
              "lighting": "...",
              "color": "...",
              "style": "...",
              "text_in_scene": "{token}",
              "must_keep": ["...", "..."],
              "avoid": ["...", "..."]
            }}

            Hard requirements:
            - Output ONLY valid JSON (no markdown, no code fences, no comments).
            - No empty strings. No empty arrays.
            - "subject" must be specific and unique (avoid generic subjects like "a person", "someone", "a figure").
            - "text_in_scene" must exactly equal the required token.
            - "must_keep" and "avoid" must each contain at least 3 concrete, visual items.
            - Keep it a single coherent moment (no montages, no scene cuts).

            Self-check before output (do not include this check in the output):
            - Every required key exists and is non-empty.
            - text_in_scene matches the token exactly.
            """
        ).strip()

    return StageSpec(
        stage_id="ab.scene_spec_json",
        prompt=_prompt,
        temperature=0.55,
        output_key="ab_scene_spec_json",
    )


@StageCatalog.register(
    "ab.final_prompt_format",
    doc="Format the refined scene into a strict single-line prompt template.",
    source="prompts.ab_final_prompt_format",
    tags=("ab",),
)
def ab_final_prompt_format(_inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        token = _ab_require_text_output(ctx, "ab_random_token")
        refined = _ab_require_text_output(ctx, "ab_scene_refined")

        return textwrap.dedent(
            f"""\
            Convert the refined scene description into a final image generation prompt using the exact one-line format below.

            Refined scene:
            {refined}

            Output format (exactly one line; keep labels and separators):
            SUBJECT=<...> | SETTING=<...> | ACTION=<...> | COMPOSITION=<...> | CAMERA=<...> | LIGHTING=<...> | COLOR=<...> | STYLE=<...> | TEXT_IN_SCENE="{token}" | AR=16:9

            Rules:
            - Ensure TEXT_IN_SCENE uses the token exactly as provided.
            - Do not add extra lines before/after.
            """
        ).strip()

    return StageSpec(
        stage_id="ab.final_prompt_format",
        prompt=_prompt,
        temperature=0.6,
        is_default_capture=True,
    )


@StageCatalog.register(
    "ab.final_prompt_format_from_scenespec",
    doc="Format a SceneSpec JSON intermediary into the strict single-line prompt template.",
    source="inline",
    tags=("ab",),
)
def ab_final_prompt_format_from_scenespec(_inputs: PlanInputs) -> StageSpec:
    def _prompt(ctx: RunContext) -> str:
        token = _ab_require_text_output(ctx, "ab_random_token")
        spec_json = _ab_require_text_output(ctx, "ab_scene_spec_json")

        return textwrap.dedent(
            f"""\
            Convert the SceneSpec JSON into a final image generation prompt using the exact one-line format below.

            SceneSpec JSON:
            {spec_json}

            Output format (exactly one line; keep labels and separators):
            SUBJECT=<...> | SETTING=<...> | ACTION=<...> | COMPOSITION=<...> | CAMERA=<...> | LIGHTING=<...> | COLOR=<...> | STYLE=<...> | TEXT_IN_SCENE="{token}" | AR=16:9

            Rules:
            - Use only information present in the JSON (no inventions).
            - Do not output placeholders like "<...>"; fill every field concretely.
            - Ensure TEXT_IN_SCENE uses the token exactly as provided.
            - Do not add extra lines before/after.
            """
        ).strip()

    return StageSpec(
        stage_id="ab.final_prompt_format_from_scenespec",
        prompt=_prompt,
        temperature=0.55,
    )
