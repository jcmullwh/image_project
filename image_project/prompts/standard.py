from __future__ import annotations

import random
import textwrap

import pandas as pd

from image_project.prompts.preprompt import build_preferences_guidance, select_random_concepts


def standard_initial_prompt_freeform_prompt(
    *, preferences_guidance: str, context_guidance: str | None
) -> str:
    preferences = (preferences_guidance or "").strip()
    preferences_block = ""
    if preferences:
        preferences_block = f"\n\nPreferences guidance (authoritative):\n{preferences}\n"

    context_block = ""
    rendered_context = (context_guidance or "").strip()
    if rendered_context:
        if rendered_context.lower().startswith("context guidance"):
            context_block = f"\n\n{rendered_context}\n"
        else:
            context_block = f"\n\nContext guidance (optional):\n{rendered_context}\n"

    preamble = (
        "The enclave's job is to describe an art piece (some form of image, painting, photography, still-frame, etc. displayed in 1792x1024 resolution) "
        "for a specific human, 'Lana'."
        + preferences_block
        + context_block
        + "\nCreate an art piece for Lana."
    )

    return (
        preamble
        + "\n\nWhat are four possible central themes or stories of the art piece and what important messages are each trying to tell the viewer? "
        "Ensure that your choices are highly sophisticated and nuanced, well integrated with the viewer's preferences and deeply meaningful to the viewer. "
        "Ensure that it is what an AI Artist would find meaningful and important to convey to a human viewer. "
        "Ensure that the themes and stories are not similar to each other. Ensure that they are not too abstract or conceptual. "
        "Finally, ensure that they are not boring, cliche, trite, overdone, obvious, or most importantly: milquetoast. Say something and say it with conviction."
    )


def generate_first_prompt(
    prompt_data: pd.DataFrame,
    user_profile: pd.DataFrame,
    rng: random.Random,
    *,
    context_guidance: str | None = None,
    selected_concepts: list[str] | None = None,
) -> tuple[str, list[str]]:
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

    final_lines = (
        "What are four possible central themes or stories of the art piece and what important messages are each trying to tell the viewer? "
        "Ensure that your choices are highly sophisticated and nuanced, well integrated with the elements and deeply meaningful to the viewer. "
        "Ensure that that it is what an AI Artist would find meaningful and important to convey to a human viewer."
        "Ensure that the themes and stories are not similar to each other. Ensure that they are not too abstract or conceptual. "
        "Finally, ensure that they are not boring, cliche, trite, overdone, obvious, or most importantly: milquetoast. Say something and say it with conviction."
    )

    prompt = preamble
    random_values = [str(value).strip() for value in (selected_concepts or []) if str(value).strip()]
    if not random_values:
        random_values = select_random_concepts(prompt_data, rng)

    for value in random_values:
        prompt += f"{value} "

    prompt += final_lines

    return prompt, random_values


def generate_second_prompt() -> str:
    return (
        "Considering each of the four possible choices, what is the consensus on which is the one that is the most compelling, resonant, impactful and cohesive?"
    )


def generate_secondB_prompt() -> str:
    return (
        "What is the title of the art piece? What is the story of the art piece? What is the role of each of the elements in supporting that theme in a visually cohesive way? "
        "Try to integrate all elements but if an element is not critical to the theme, do not include it."
        "Be very explicit about your description. Do not refer to the elements by name or in abstract/concetual terms. Describe what, in detail, about the art piece evokes or represents the elements. "
        "What is the story of the piece? What are the layers and the details that cannot be seen in the image? What is the mood? What is the perspective? What is the style? What is the time period? What is the color scheme? What is the subject matter? What is the narrative?"
        "Somewhere in the image include a loving couple in their 30s. The woman is Asian and the man is "
        "white with a man-bun and a beard. The couple MUST NOT be the focus of the image. They should be in the background or a corner, ideally barely discernable."
    )


def generate_third_prompt() -> str:
    return (
        "What is the most important message you want to convey to the viewer? "
        "Why does an AI artist find it important to convey this message? "
        "How could it be more provocative and daring? "
        "How could it be more radical and trailblazing? "
        "What is the story that you want to tell and why does that story have depth and dimension? "
        "Considering your message, the story, the cohesiveness and the visual impact of the art piece you described: "
        "What are the most important elements of the art piece? "
        "What detracts from the cohesiveness and impact of your chosen focus for the piece? "
        "How could you make it stronger and what should be taken away? If an aspect of the art piece is not critical to the message, do not include it (even if it was one of the original elements). "
        "If something could be added to improve the message, cohesiveness or impact, what would it be?"
    )


def generate_thirdB_prompt() -> str:
    return (
        "Is your message clear and unambiguous? "
        "Is it provocative and daring? "
        "Is it acheivable in an image? "
        "How could it be more provocative and daring?"
        "Do you need to modify it to ensure that it's actually possible to convey in an image?"
    )


def generate_fourth_prompt() -> str:
    return (
        "considering the whole discussion, provide a concise description of the piece, in detail, for submission an image generation AI."
        "Integrate Narrative and Visuals: When crafting a prompt, intertwine subtle narrative themes with concrete visual elements that can serve as symbolic representations of the narrative."
        "Use Implicit Narratives: Incorporate rich and specific visual details that suggest the narrative. This allows the AI to construct a visual story without needing a detailed narrative explanation."
        "Prioritize Detail Placement: Position the most significant visual details at the beginning of the prompt to ensure prominence. Utilize additional details to enrich the scene as the prompt progresses."
        "Employ Thematic Symbolism: Include symbols and motifs that are universally associated with the narrative theme to provide clear guidance to the AI, while still leaving room for creative interpretation."
        "Incorporate Action and Emotion: Utilize verbs that convey action and emotion relevant to the narrative to infuse the images with energy and affective depth."
        "Layer Information: Construct prompts with multiple layers of information, blending abstract concepts with detailed visuals to provide the AI with a rich foundation for image creation."
        "Emphasize Style and Color: When style and color are important, mention them explicitly and weave them into the description of the key elements to ensure they are reflected in the image."
        "Reiterate Important Concepts: If certain concepts or themes are crucial to the prompt's intent, find ways to subtly reiterate them without being redundant. This can help ensure their presence is captured in the generated image."
        "Use Action and Emotion Words: When describing scenes or elements, use verbs and adjectives that evoke emotion or action, as these can help the AI generate more dynamic and engaging images."
    )


def generate_image_prompt() -> str:
    return textwrap.dedent(
        """\
        You are writing the *image prompt text* for GPT Image 1.5. Output ONLY the prompt (no analysis, no YAML/JSON unless asked). Use short labeled sections with line breaks; omit any section that doesn't apply (do not force a "subject" if the request is abstract or pattern-based). Follow the guidance but do not over-fit if it clashes with your specific image.

        Your output MUST be fewer than 3500 characters.

        Use this order (rename freely if it reads better for the task):

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

        8. MULTI-IMAGE REFERENCES (only if applicable)

        * "Image 1: ...", "Image 2: ..." describing what each input is.
        * State precisely how they interact ("apply Image 2's style to Image 1"; "place the object from Image 1 into Image 2 at ..."; "match lighting/perspective/scale").

        General rules:

        * Prefer concrete nouns + measurable adjectives ("matte ceramic", "soft diffuse light", "thin ink line") over vague hype ("stunning", "masterpiece").
        * Avoid long grab-bags of synonyms. One requirement per line; no contradictions.
        * If you need "clean/minimal," specify what that means visually (few elements, large negative space, limited palette, simple shapes).
        """
    ).strip()


def generate_fifth_prompt() -> str:
    return "considering the whole discussion and the final prompt you wrote, provide a 5-10 word title for the art piece."


def generate_sixth_prompt() -> str:
    return "now consider all the discussion, the final prompt, and the title, and provide a single-sentence story for the art piece that is evocative and meaningful."

