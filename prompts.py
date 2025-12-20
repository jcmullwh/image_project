"""Prompt strings and prompt-construction helpers.

This module keeps prompt text and ToT/enclave prompt factories out of main orchestration.
"""

from __future__ import annotations

import random
import textwrap
from typing import Any

import pandas as pd

from pipeline import Block, ChatStep, RunContext


DEFAULT_SYSTEM_PROMPT = (
    "You are a highly skilled enclave of Artists trained to generate meaningful, edgy, artistic images on par "
    "with the greatest artists of any time, anywhere, past or future, Earth or any other planet. The enclave "
    "invents unique images that weave together seemingly disparate elements into cohesive wholes that push "
    "boundaries and elicit deep emotions in a human viewer. Keep your responses concise and focused on the task at hand."
)

def build_preferences_guidance(user_profile: pd.DataFrame) -> str:
    if user_profile is None or user_profile.empty:
        return ""

    sections: list[str] = []
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


def generate_first_prompt(
    prompt_data,
    user_profile,
    rng: random.Random,
    *,
    context_guidance: str | None = None,
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

    
    # Define the groups
    group1 = ['Subject Matter', 'Narrative']
    group2 = ['Mood', 'Composition', 'Perspective']
    group3 = ['Style', 'Time Period_Context', 'Color Scheme']
    
    # Generate the prompt
    prompt = preamble
    random_values = []

    random_value1 = get_random_value_from_group(group1, prompt_data, rng)
    if random_value1:
        prompt += f"{random_value1} "
        random_values.append(random_value1)

    random_value2 = get_random_value_from_group(group2, prompt_data, rng)
    if random_value2:
        prompt += f"{random_value2} "
        random_values.append(random_value2)

    random_value3 = get_random_value_from_group(group3, prompt_data, rng)
    if random_value3:
        prompt += f"{random_value3} "
        random_values.append(random_value3)
    

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
        You are writing the *image prompt text* for GPT Image 1.5. Output ONLY the prompt (no analysis, no YAML/JSON unless asked). Use short labeled sections with line breaks; omit any section that doesn’t apply (do not force a “subject” if the request is abstract or pattern-based). Follow the guidance but do not over-fit if it clashes with your specific image.

        Your output MUST be fewer than 3500 characters.

        Use this order (rename freely if it reads better for the task):

        1. DELIVERABLE / INTENT

        * What kind of image this is (e.g., “editorial photo”, “abstract painting”, “UI mockup”, “infographic”, “logo”) and what it should feel like (1 sentence).

        2. CONTENT (works for representational or abstract)

        * If representational: the main entities + actions/poses + key attributes.
        * If abstract/non-representational: the primary forms/motifs (geometry, strokes, textures), relationships (layering, symmetry, repetition, flow), and whether there is *no* recognizable subject matter.

        3. CONTEXT / WORLD (optional)

        * Setting, time, atmosphere, environment rules; or for abstract work: canvas/material, spatial depth, background treatment.

        4. STYLE / MEDIUM

        * Specify the medium (photo, watercolor, vector, 3D render, ink, collage, generative pattern).
        * Add 2–5 concrete style cues tied to visuals (materials, texture, line quality, grain).

        5. COMPOSITION / GEOMETRY

        * Framing/viewpoint (close-up/wide/top-down), perspective/angle, and lighting/mood when relevant.
        * If layout matters, specify placement explicitly (“centered”, “negative space left”, “text top-right”, “balanced margins”, “grid with 3 columns”).

        6. CONSTRAINTS (be explicit and minimal)

        * MUST INCLUDE: short bullets for non-negotiables.
        * MUST PRESERVE: identity/geometry/layout/brand elements that cannot change (if relevant).
        * MUST NOT INCLUDE: short bullets for exclusions (e.g., “no watermark”, “no extra text”, “no logos/trademarks”).

        7. TEXT IN IMAGE (only if required)

        * Put exact copy in quotes or ALL CAPS.
        * Specify typography constraints (font style, weight, color, size, placement) and demand verbatim rendering with no extra characters.
        * For tricky spellings/brand names: optionally spell the word letter-by-letter.

        8. MULTI-IMAGE REFERENCES (only if applicable)

        * “Image 1: …”, “Image 2: …” describing what each input is.
        * State precisely how they interact (“apply Image 2’s style to Image 1”; “place the object from Image 1 into Image 2 at …”; “match lighting/perspective/scale”).

        General rules:

        * Prefer concrete nouns + measurable adjectives (“matte ceramic”, “soft diffuse light”, “thin ink line”) over vague hype (“stunning”, “masterpiece”).
        * Avoid long grab-bags of synonyms. One requirement per line; no contradictions.
        * If you need “clean/minimal,” specify what that means visually (few elements, large negative space, limited palette, simple shapes).
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


ENCLAVE_ARTISTS: list[tuple[str, str, str]] = [
    (
        "hemingway",
        "Hemingway",
        "Terse, concrete, and opinionated. Iceberg theory: focus on what matters and cut fluff.",
    ),
    (
        "munch",
        "Munch",
        "Emotion-first expressionist. Prioritize mood, tension, symbolism, and visceral resonance.",
    ),
    (
        "da_vinci",
        "da Vinci",
        "Systems thinker. Tie disparate elements into a cohesive whole with strong composition and purpose.",
    ),
    (
        "representative",
        "Representative",
        "Audience translator. Optimize for Lana's stated likes/dislikes; remove anything that will annoy her. You have veto authority. Allow the artists creative freedom but never allow clear conflicts with stated dislikes.",
    ),
    (
        "chameleon",
        "Chameleon",
        "Match the specific style and subject matter implied by the draft; sharpen genre conventions and specificity.",
    ),
]


def enclave_thread_prompt(
    label: str,
    persona: str,
    *,
    preferences_guidance: str | None = None,
    first_prompt_random_values: list[str] | None = None,
) -> str:
    reference_sections: list[str] = []
    if preferences_guidance and str(preferences_guidance).strip():
        reference_sections.append(
            "Preferences guidance (authoritative):\n" + str(preferences_guidance).strip()
        )

    if first_prompt_random_values:
        values = [str(value).strip() for value in first_prompt_random_values if str(value).strip()]
        if values:
            reference_sections.append(
                "First-prompt random values (authoritative):\n"
                + "\n".join(f"- {value}" for value in values)
            )

    reference_block = ""
    if reference_sections:
        reference_block = (
            "\n\nReference material below is authoritative even if it is missing from the conversation context.\n\n"
            + "\n\n".join(reference_sections)
            + "\n\n"
        )

    return (
        f"You are {label}.\n"
        f"Persona: {persona}\n"
        + reference_block
        + "\nYou are a single voice.\n"
        "You do NOT see any other artists' feedback.\n"
        "Critique and refine ONLY the last assistant response in this conversation.\n"
        "Do not add meta commentary.\n\n"
        "Return a structured critique with two sections:\n"
        "## Issues\n"
        "- ...\n\n"
        "## Edits\n"
        "- ... (concrete replacements/rewrites)\n"
        "Keep it succinct and focused. No more than 1000 characters."
    )


def make_tot_enclave_block(stage_name: str) -> Block:
    nodes: list[Any] = []

    for artist_key, label, persona in ENCLAVE_ARTISTS:
        capture_key = f"enclave.{stage_name}.{artist_key}"

        nodes.append(
            ChatStep(
                name=artist_key,
                merge="none",
                capture_key=capture_key,
                prompt=lambda ctx, label=label, persona=persona, artist_key=artist_key: enclave_thread_prompt(
                    label,
                    persona,
                    preferences_guidance=(
                        ctx.outputs.get("preferences_guidance") if artist_key == "representative" else None
                    ),
                    first_prompt_random_values=(
                        ctx.selected_concepts if artist_key == "chameleon" else None
                    ),
                ),
                temperature=0.8,
            )
        )

    def consensus_prompt(ctx: RunContext, stage_name: str = stage_name) -> str:
        notes: list[str] = []
        for artist_key, label, _persona in ENCLAVE_ARTISTS:
            key = f"enclave.{stage_name}.{artist_key}"
            value = ctx.outputs.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing enclave thread output: {key}")
            notes.append(f"## {label}\n{value.strip()}")

        return (
            "You are the enclave consensus editor.\n"
            "Using the independent artist notes below, revise the last assistant response.\n"
            "Keep the original intent and constraints.\n"
            "Return ONLY the revised response (no preamble, no analysis).\n\n"
            "If there is any disagreement amoung the artists, the Representative artist's opinion should have the most weight. Additionally, the Representative has veto authority. Second largest weight goes to the Chameleon artist. However, the chameleon does not have veto authority.\n\n"
            + "\n\n".join(notes)
        )

    nodes.append(ChatStep(name="consensus", prompt=consensus_prompt, temperature=0.8))

    return Block(name="tot_enclave", merge="all_messages", nodes=nodes)
