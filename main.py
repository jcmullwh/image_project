import base64
import json
import logging
import os
import random
import subprocess
import sys
import time
from typing import Any

import pandas as pd

try:
    from ai_backend import ImageAI, TextAI
except ModuleNotFoundError:  # pragma: no cover
    ImageAI = None  # type: ignore[assignment]
    TextAI = None  # type: ignore[assignment]
from message_handling import MessageHandler
from pipeline import Block, ChatRunner, ChatStep, RunContext
from records import append_generation_row
from run_config import RunConfig
from transcript import write_transcript
from upscaling import UpscaleConfig, upscale_image_to_4k
from titles import (
    append_manifest_row,
    generate_title,
    get_next_seq,
    manifest_lock,
    read_manifest,
    utc_now_iso8601,
)
from utils import (
    generate_file_location,
    generate_unique_id,
    load_config,
    save_image,
)

def load_prompt_data(file_path):
    # load the prompt data from the csv file
    
    data = pd.read_csv(file_path)
    
    return data

def generate_first_prompt(prompt_data, user_profile, rng: random.Random):

    likes = user_profile["Likes"].dropna().astype(str).tolist()
    dislikes = user_profile["Dislikes"].dropna().astype(str).tolist()
    # Convert lists to comma-separated strings
    likes_text = ", ".join(likes)
    dislikes_text = ", ".join(dislikes)
    
    preamble = f"The enclave's job is to describe an art piece (some form of image, painting, photography, still-frame, etc. dispalyed in 1792x1024 resolution) for a specific human, 'Lana'. We know that Lana Likes: \
Vibrant colors, symmetry, nature themes, artistic styles, storytelling and interesting world building. \
Lana Dislikes:{dislikes_text}.\
Lana Likes: {likes_text}.\
Create an art piece for Lana that incorporates and thoughtfully blends the below elements."

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

# Function to get a random value from a group of columns
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

def generate_dalle_prompt():
    
    dalle_prompt = "Considering the entire conversation, create a prompt that is optimized for submission to DALL-E3. It MUST be fewer than 3500 characters."
    return dalle_prompt


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
        "Audience translator. Optimize for Lana's stated likes/dislikes; remove anything that will annoy her.",
    ),
    (
        "chameleon",
        "Chameleon",
        "Match the specific style and subject matter implied by the draft; sharpen genre conventions and specificity.",
    ),
]


def enclave_thread_prompt(label: str, persona: str) -> str:
    return (
        f"You are {label}.\n"
        f"Persona: {persona}\n\n"
        "You are a single voice.\n"
        "You do NOT see any other artists' feedback.\n"
        "Critique and refine ONLY the last assistant response in this conversation.\n"
        "Do not add meta commentary.\n\n"
        "Return a structured critique with two sections:\n"
        "## Issues\n"
        "- ...\n\n"
        "## Edits\n"
        "- ... (concrete replacements/rewrites)\n"
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
                prompt=lambda _ctx, label=label, persona=persona: enclave_thread_prompt(
                    label, persona
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
            + "\n\n".join(notes)
        )

    nodes.append(ChatStep(name="consensus", prompt=consensus_prompt, temperature=0.8))

    return Block(name="tot_enclave", merge="all_messages", nodes=nodes)


def configure_stdio_utf8():
    """Force stdout/stderr to UTF-8 so Unicode responses never crash on Windows consoles."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # If reconfigure is unavailable, continue with defaults.
        pass

def setup_operational_logger(log_dir: str, generation_id: str):
    """
    Configure a logger that writes an operational log for traceability.
    Logs go to both stdout and a UTF-8 file under the provided directory.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = generate_file_location(log_dir, f"{generation_id}_oplog", ".log")

    logger_name = f"image_project.{generation_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info("Operational logging initialized for generation %s", generation_id)
    logger.debug("Operational log file: %s", log_file)

    return logger, log_file


def upload_to_photos_via_rclone(
    file_path: str,
    remote: str = "gphotos",
    album: str = "Generated-Art",
    logger: logging.Logger | None = None,
) -> bool:
    dest = f"{remote}:album/{album}"
    if logger:
        logger.info("Uploading %s to %s via rclone", file_path, dest)
    try:
        subprocess.run(["rclone", "copy", file_path, dest], check=True)
    except FileNotFoundError as exc:
        if logger:
            logger.error("Upload skipped: rclone not found (%s).", exc)
        return False
    except subprocess.CalledProcessError as exc:
        if logger:
            logger.error("Upload failed (rclone exit=%s). Continuing.", exc.returncode)
        return False
    except Exception:
        if logger:
            logger.exception("Upload failed due to unexpected error. Continuing.")
        return False

    return True



GENERATION_CSV_FIELDNAMES: list[str] = [
    "generation_id",
    "selected_concepts",
    "final_image_prompt",
    "image_path",
    "created_at",
    "seed",
]


def run_generation(cfg_dict, *, generation_id: str | None = None) -> RunContext:
    cfg, cfg_warnings = RunConfig.from_dict(cfg_dict)

    generation_id = generation_id or generate_unique_id()
    logger, operational_log_path = setup_operational_logger(cfg.log_dir, generation_id)
    logger.info("Run started for generation %s", generation_id)

    for warning in cfg_warnings:
        logger.warning("%s", warning)

    ctx: RunContext | None = None
    transcript_path: str | None = None
    phase = "init"

    try:
        phase = "seed"
        seed = cfg.random_seed
        if seed is None:
            seed = int(time.time())
            logger.info("No prompt.random_seed configured; generated seed=%d", seed)
        else:
            logger.info("Using configured prompt.random_seed=%d", seed)

        rng = random.Random(seed)

        phase = "data_load"
        logger.info("Loading prompt data from %s", cfg.categories_path)
        prompt_data = load_prompt_data(cfg.categories_path)
        logger.info("Loaded %d category rows", len(prompt_data))

        logger.info("Loading user profile from %s", cfg.profile_path)
        user_profile = pd.read_csv(cfg.profile_path)
        logger.info("Loaded %d user profile rows", len(user_profile))

        phase = "prompt_generation"
        prompt_1, selected_concepts = generate_first_prompt(prompt_data, user_profile, rng)
        logger.info("Generated first prompt (selected_concepts=%d)", len(selected_concepts))

        phase = "init_text_ai"
        text_ai_cls = TextAI
        if text_ai_cls is None:  # pragma: no cover
            from ai_backend import TextAI as text_ai_cls  # type: ignore[assignment]

        ai_text = text_ai_cls(model="gpt-5.2", reasoning={"effort": "medium"})
        logger.info("Initialized TextAI with model gpt-5.2")

        system_prompt = (
            "You are a highly skilled enclave of Artists trained to generate meaningful, edgy, artistic images on par "
            "with the greatest artists of any time, anywhere, past or future, Earth or any other planet. The enclave "
            "invents unique images that weave together seemingly disparate elements into cohesive wholes that push "
            "boundaries and elicit deep emotions in a human viewer."
        )

        phase = "context"
        ctx = RunContext(
            generation_id=generation_id,
            cfg=cfg,
            logger=logger,
            rng=rng,
            seed=seed,
            created_at=utc_now_iso8601(),
            messages=MessageHandler(system_prompt),
        )
        ctx.selected_concepts = selected_concepts

        transcript_path = generate_file_location(cfg.log_dir, generation_id + "_transcript", ".json")

        phase = "pipeline"
        def refined_stage(stage_step: ChatStep, *, capture_key: str | None = None) -> Block:
            stage_name = stage_step.name
            if stage_name is None:
                raise ValueError("Stage steps must have names for stable pipeline paths")
            if stage_step.capture_key:
                raise ValueError(
                    f"Stage step {stage_name} must not set capture_key; pass capture_key to refined_stage()"
                )
            draft_step = ChatStep(
                name="draft",
                prompt=stage_step.prompt,
                temperature=stage_step.temperature,
                allow_empty_prompt=stage_step.allow_empty_prompt,
                allow_empty_response=stage_step.allow_empty_response,
                params=dict(stage_step.params),
            )
            enclave = make_tot_enclave_block(stage_name)
            return Block(
                name=stage_name,
                merge="last_response",
                nodes=[draft_step, enclave],
                capture_key=capture_key,
            )

        pipeline_root = Block(
            name="pipeline",
            merge="all_messages",
            nodes=[
                refined_stage(ChatStep(name="initial_prompt", prompt=prompt_1, temperature=0.8)),
                refined_stage(
                    ChatStep(
                        name="section_2_choice",
                        prompt=lambda _ctx: generate_second_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="section_2b_title_and_story",
                        prompt=lambda _ctx: generate_secondB_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="section_3_message_focus",
                        prompt=lambda _ctx: generate_third_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="section_3b_message_clarity",
                        prompt=lambda _ctx: generate_thirdB_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="section_4_concise_description",
                        prompt=lambda _ctx: generate_fourth_prompt(),
                        temperature=0.8,
                    )
                ),
                refined_stage(
                    ChatStep(
                        name="dalle_prompt_creation",
                        prompt=lambda _ctx: generate_dalle_prompt(),
                        temperature=0.8,
                    ),
                    capture_key="dalle_prompt",
                ),
                refined_stage(
                    ChatStep(
                        name="section_5_midjourney_refine",
                        prompt=lambda _ctx: generate_fifth_prompt(),
                        temperature=0.8,
                    )
                ),
            ],
        )

        runner = ChatRunner(ai_text=ai_text)
        runner.run(ctx, pipeline_root)

        phase = "capture"
        dalle_prompt = ctx.outputs.get("dalle_prompt")
        if not dalle_prompt:
            raise ValueError("Pipeline did not produce required output: dalle_prompt")

        phase = "init_image_ai"
        image_ai_cls = ImageAI
        if image_ai_cls is None:  # pragma: no cover
            from ai_backend import ImageAI as image_ai_cls  # type: ignore[assignment]

        ai_image = image_ai_cls()
        logger.info("Initialized ImageAI")

        image_model = "gpt-image-1.5"
        image_size = "1536x1024"
        image_quality = "high"

        os.makedirs(cfg.generation_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.upscale_dir, exist_ok=True)

        phase = "image_pipeline"
        upload_target_path = ""
        seq = -1
        title_result = None

        with manifest_lock(cfg.titles_manifest_path):
            manifest_rows = read_manifest(cfg.titles_manifest_path)
            existing_titles = [row.get("title", "") for row in manifest_rows if row.get("title")]
            existing_titles = list(reversed(existing_titles))

            seq = get_next_seq(cfg.titles_manifest_path)
            title_result = generate_title(
                ai_text=ai_text,
                image_prompt=dalle_prompt,
                avoid_titles=existing_titles,
            )

            caption_text = f"#{seq:03d} - {title_result.title}"
            logger.info("Assigned image identifier %s", caption_text)

            image_result = ai_image.generate_image(
                dalle_prompt,
                model=image_model,
                size=image_size,
                quality=image_quality,
                moderation="low",
            )
            logger.info(
                "Image generation request sent (model=%s, size=%s, quality=%s)",
                image_model,
                image_size,
                image_quality,
            )

            image_data = image_result["image"]
            logger.debug("Received image payload length: %d", len(image_data))
            image_bytes = base64.b64decode(image_data)

            image_full_path_and_name = generate_file_location(
                cfg.generation_dir, generation_id + "_image", ".jpg"
            )
            save_image(
                image_bytes,
                image_full_path_and_name,
                caption_text=caption_text,
                caption_font_path=cfg.caption_font_path,
            )
            logger.info("Saved image to %s", image_full_path_and_name)
            upload_target_path = image_full_path_and_name
            ctx.image_path = upload_target_path

            if cfg.upscale_enabled:
                upscale_out_path = generate_file_location(
                    cfg.upscale_dir,
                    generation_id + "_image_4k",
                    ".jpg",
                )

                upscale_cfg = UpscaleConfig(
                    target_long_edge_px=cfg.upscale_target_long_edge_px,
                    engine=cfg.upscale_engine,
                    realesrgan_binary=cfg.upscale_realesrgan_binary,
                    model_name=cfg.upscale_model_name,
                    model_path=cfg.upscale_model_path,
                    tile_size=cfg.upscale_tile_size,
                    tta=cfg.upscale_tta,
                    allow_fallback_resize=cfg.upscale_allow_fallback_resize,
                )

                logger.info(
                    "Upscaling enabled: engine=%s model=%s target_long_edge_px=%d",
                    upscale_cfg.engine,
                    upscale_cfg.model_name,
                    upscale_cfg.target_long_edge_px,
                )
                upscale_image_to_4k(
                    input_path=image_full_path_and_name,
                    output_path=upscale_out_path,
                    config=upscale_cfg,
                )
                logger.info("Saved 4K upscaled image to %s", upscale_out_path)
                upload_target_path = upscale_out_path
                ctx.image_path = upload_target_path

            append_manifest_row(
                cfg.titles_manifest_path,
                {
                    "seq": int(seq),
                    "title": title_result.title,
                    "generation_id": generation_id,
                    "image_prompt": dalle_prompt,
                    "image_path": upload_target_path,
                    "created_at": utc_now_iso8601(),
                    "model": image_model,
                    "size": image_size,
                    "quality": image_quality,
                    "seed": image_result.get("seed", ""),
                    "title_source": getattr(title_result, "title_source", "llm"),
                    "title_raw": getattr(title_result, "title_raw", ""),
                },
            )
            logger.info("Appended manifest row to %s (seq=%d)", cfg.titles_manifest_path, seq)

        phase = "records"
        append_generation_row(
            cfg.generations_csv_path,
            {
                "generation_id": generation_id,
                "selected_concepts": json.dumps(selected_concepts, ensure_ascii=False),
                "final_image_prompt": dalle_prompt,
                "image_path": upload_target_path,
                "created_at": ctx.created_at,
                "seed": seed,
            },
            GENERATION_CSV_FIELDNAMES,
        )
        logger.info("Appended generation row to %s", cfg.generations_csv_path)

        phase = "rclone"
        if cfg.rclone_enabled:
            remote = cfg.rclone_remote
            album = cfg.rclone_album
            if not remote or not album:
                raise ValueError("rclone.enabled=true but rclone.remote/album missing")

            if not upload_target_path:
                logger.error("Rclone upload enabled but upload target path is empty; skipping upload.")
            else:
                uploaded = upload_to_photos_via_rclone(
                    upload_target_path,
                    remote=remote,
                    album=album,
                    logger=logger,
                )
                if uploaded:
                    logger.info("Uploaded image via rclone to %s", f"{remote}:album/{album}")
                else:
                    logger.error("Rclone upload failed; image remains at %s", upload_target_path)

        phase = "transcript"
        write_transcript(transcript_path, ctx)
        logger.info("Wrote transcript JSON to %s", transcript_path)
        logger.info("Operational log stored at %s", operational_log_path)
        logger.info("Run completed successfully for generation %s", generation_id)

        return ctx
    except Exception as exc:
        logger.exception("Run failed during phase %s", phase)
        if ctx is not None:
            step_name = getattr(exc, "pipeline_step", None)
            pipeline_path = getattr(exc, "pipeline_path", None)
            ctx.error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "phase": phase,
            }
            if step_name:
                ctx.error["step"] = step_name
            if pipeline_path:
                ctx.error["path"] = pipeline_path

            try:
                if transcript_path is None:
                    transcript_path = generate_file_location(
                        cfg.log_dir, generation_id + "_transcript", ".json"
                    )
                write_transcript(transcript_path, ctx)
                logger.info("Wrote transcript JSON to %s", transcript_path)
            except Exception:
                logger.exception("Failed to write transcript during error handling")
        raise
    finally:
        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers.clear()


def main() -> None:
    configure_stdio_utf8()
    generation_id = generate_unique_id()

    try:
        cfg_dict = load_config()
    except Exception:
        # This is a rare exeception to the "no fallbacks" rule. Fallbacks are ONLY acceptable here to prevent loss of logging data
        fallback_logger, fallback_log_path = setup_operational_logger(os.getcwd(), generation_id)
        fallback_logger.exception(
            "Failed to load configuration. Logging to fallback file at %s", fallback_log_path
        )
        raise

    run_generation(cfg_dict, generation_id=generation_id)


if __name__ == "__main__":
    main()
