# image_project
 
## Project Notes:

Personal project exploring prompting and "art" generation. Takes randomly-selected concepts ("Baroque", "Portrait"), creates a cohesive "art" prompt incorporating a specific user's likes and dislikes then generates and saves the image

Uses tree-of-thought prompting, experiments with iterative improvement and identities to help guide prompt development.

## Step-Driven Prompt Pipeline

The generation workflow is implemented as a small, declarative chat pipeline built from **Steps** (`ChatStep`) and nested **Blocks** (`Block`). Each step records `{name, path, prompt, response, params, created_at}` into a JSON transcript for later inspection.

See `docs/pipeline.md` for the execution model, merge modes, and the ToT/enclave wrapping pattern.

### Add a new prompt step

1. Add a new prompt function (or inline prompt factory) in `main.py`.
2. Add a new `ChatStep(...)` and include it in the `pipeline_root` construction in `main.run_generation()` (optionally wrap it as a `merge="last_response"` stage block and/or use `capture_key` to store the final output in `ctx.outputs`).

### Hardening behavior (fail-fast + reliable artifacts)

- Pipeline fails fast on `None` / non-string / empty prompts or responses (unless a step explicitly opts into empties via `allow_empty_prompt` / `allow_empty_response`).
- Config booleans are parsed strictly (e.g. `enabled: "false"` is treated as false; invalid values raise with the full key path).
- Image save failures raise and abort the run (no successful run with missing files).
- Transcript JSON is written on success and on failure (once `RunContext` exists). Failed runs include an `error` object with `type`, `message`, `phase`, and optional `step`.
- If `rclone.enabled: true`, `rclone.remote` and `rclone.album` are required (no empty-string fallbacks).
- `ChatStep.temperature` is the only supported way to set temperature (do not pass `"temperature"` inside `ChatStep.params`).

### Legacy code

- `main.py` contains the canonical implementation (`run_generation()`).
- `main_legacy.py` and `legacy_main.py` contain older experimental orchestration/helpers kept for reference.

## Configuration

Required keys (fail-fast if missing/empty):

- `prompt.categories_path`
- `prompt.profile_path`
- `prompt.generations_path`
- `image.generation_path` (preferred) or `image.save_path` (deprecated alias)
- `image.upscale_path`
- `image.log_path`

Optional keys:

- `prompt.random_seed` (int): makes concept selection deterministic; if omitted, a seed is generated and logged.
- `prompt.titles_manifest_path`: defaults to `<image.generation_path>/titles_manifest.csv` (logged at WARNING when defaulted).
- `image.caption_font_path`: optional `.ttf` for the caption overlay.
  - If explicitly set and the font cannot be loaded, the run fails loudly (no silent fallback).
- Boolean flags (e.g. `rclone.enabled`, `upscale.enabled`) accept booleans, `0`/`1`, and strings `"true"`/`"false"`/`"1"`/`"0"`/`"yes"`/`"no"` (case-insensitive); other values raise.

## Per-Image Identifiers (Seq + Title)

Each generated image is post-processed to add a subtle caption overlay so a viewer can reference images during feedback:

`#042 - Turquoise Citadel` (bottom caption strip)

### Manifest

Each image also appends a row to a CSV manifest so `#NNN` and the title can be resolved back to the underlying generation record.

- Default path: `<image.generation_path>/titles_manifest.csv`
- Override: set `prompt.titles_manifest_path` in `config/config.yaml`

Columns (v1): `seq`, `title`, `generation_id`, `image_prompt`, `image_path`, `created_at`, plus optional metadata like `model`, `size`, `quality`, `title_source`, `title_raw`.

### Sequencing

Sequence numbers are allocated as `max(seq)+1` from the manifest (starts at `1` if missing/empty).

### Failure behavior

Title generation is fail-fast: if the title cannot be produced in a valid format, the run errors rather than silently omitting the identifier.

Optional: set `image.caption_font_path` to a `.ttf` file to control the caption font (otherwise common defaults are tried).

## Run Artifacts

- Image: `<image.generation_path>/<generation_id>_image.jpg` (and optionally `<generation_id>_image_4k.jpg` when upscaling is enabled).
- Generation CSV: `prompt.generations_path` with schema `generation_id`, `selected_concepts` (JSON string), `final_image_prompt`, `image_path`, `created_at`, `seed`.
- Transcript JSON: `<image.log_path>/<generation_id>_transcript.json` with keys:
  - `generation_id`, `seed`, `selected_concepts`, `steps`, `image_path`, `created_at`

## How to run

- Generate: `pdm run generate`
- Tests: `pdm run test` (or `pytest`)

## Examples:

### Random Concepts:

['Psychedelic Nature: Nature scenes with a psychedelic, surreal twist.', "Bird's-Eye View: An overhead perspective, offering a comprehensive view from above.", 'Color Field Painting']

### User Profile:

Likes: 

colorful,
vibrant

Dislikes: 
monochromatic colors, 
apocalyptic themes, 
single character focus, 
abstract without clear story, 
horror elements

### Final Prompt:

**Title:** "The Festival of Life"

**Prompt:**
Create a vibrant, fantastical festival set in an enchanted forest from a bird's-eye view. The forest should be alive with swirling, psychedelic colorsâ€”vivid greens, deep purples, electric blues, and bright pinks. Depict trees with twisted branches that glow, interwoven with bioluminescent and holographic elements, symbolizing a fusion of nature and technology. 

At the heart of the festival, include a diverse array of mythical creatures like fairies with iridescent wings and centaurs, alongside humans in colorful, flowing garments and futuristic beings such as robots and cyborgs. Show dynamic activities filled with communal joy: a centaur and a cyborg collaborating on an art project, fairies teaching a robot to levitate, and humans engaging in virtual reality experiences with mythical beings. 

The setting sun should cast a warm, golden light over the scene, blending with the vibrant colors and bioluminescent glow to enhance the magical and harmonious atmosphere. The overall mood should be joyful, magical, and thought-provoking, with a sense of wonder and futuristic innovation.

### Style and Color:
Blend surrealism and psychedelia with futuristic elements, featuring intricate details and exaggerated, vibrant colors. Use a color scheme of electric blues, vivid greens, deep purples, bright pinks, and warm golden hues.

### Important Keywords:
Unity, diversity, harmony, communal joy, nature and technology fusion, magical mood, futuristic elements, provocative interactions, bird's-eye view.


### Result

![Example Image](example_images/20240527_205225_890de5a3-9201-4f38-8d12-bf90a9c717b0_image.jpg)

### Random Concepts:

['Whodunit: A classic detective story focusing on solving a crime or mystery.', 'Subjective Perspective: Representing a scene from a subjective point of view, often to convey emotional states.', 'Complementary: Colors opposite each other on the color wheel, offering high contrast and vibrant visuals.']

### User Profile:

Likes:
Minimalist design,
Black and white photography,
Complex narratives,
Historical dramas,
Classical music,

Dislikes:
Bright colors,
Modern pop culture,
Simplistic or superficial themes,
Loud environments,
Fast-paced action movies

### Final Prompt

 Title: The Veil of Dreams

**Visual Description:**
- **Central Scene:** A 1930s detective, dressed in period attire, is asleep at a wooden desk. The desk is cluttered with notes, black-and-white photographs, and old maps, signifying his deep involvement in a mystery. His facial expression reflects a mix of peace and slight unease.
- **Dreamscape Integration:** Above and around the detective, an ethereal dreamscape unfolds. Floating islands, surreal architecture, and mystical creatures in vivid violet and vibrant yellow hues seamlessly merge into the real-world setting. Books float mid-air, spectral figures lurk in shadows, and parts of the dream-like architecture break through the walls and floor, creating a striking, surreal atmosphere.
- **Symbolic Details:** The desk includes a broken clock, scattered keys, and torn pages with cryptic messages, symbolizing hidden truths and fragmented thoughts. Faint, ghostly figures from the dreamscape subtly interact with the real-world elements, further blurring the lines between reality and dreams.
- **Lighting and Shadows:** Dramatic lighting casts stark contrasts and eerie shadows, enhancing the dreamlike and unsettling mood of the scene. Monochromatic tones in shades of brown and sepia dominate the real-world elements, contrasting sharply with the vivid violet and vibrant yellow hues of the dreamscape.
- **Action and Emotion Words:** Asleep, floating, lurking, breaking, hinting, interacting. Peaceful, uneasy, ethereal, vivid, disorienting, surreal, dramatic, eerie, transparent.
- **Art Style:** A blend of realistic and surrealist artistry, with the detective’s environment rendered in a detailed, realistic manner and the dreamscape featuring fluid, abstract forms.

**Narrative and Thematic Symbolism:**
- **Subconscious Exploration:** The detective’s journey into his subconscious, where intuition and imagination guide him through a disorienting blend of reality and dreams. The dreamscape reveals not just fantastical elements but also hidden truths and emotional connections.
- **Emotional and Psychological Tension:** Tension between the conscious mind and subconscious revelations is depicted through the merging of dream elements with reality, creating a sense of disorientation and introspection.

### Result

![Example Image](example_images/20240530_154914_1ec45606-4a4c-4e27-955c-ee9d1caa107c_image.jpg)

### Notes:

  Current prompting methods seem to repeatedly hone in on sunset/sunrise and environmental themes. Trees are frequently incorporated. Would be interesting to explore further.

## Future Directions:
- integrate like/dislike mechanism
    - MVP like/dislike input mechanism
    - MVP like/dislike tracking
    - language tracking ("I don't like how it's too dark")
    - language tracking -> like/dislike variables

- Frequency adjustments based on likes/dislikes and recent-ness

- Implement injection of local/current things (date, season, location, etc.)

- Programmatically modify prompt structure and implement variations.
- Implement prompting A/B testing

- integrate grading mechanism based on how well it aligns with the selected concepts and additional requirements

- automatically generate categories concepts at first install

- Display Frontend

- Integrate with midjourney API when available
    - Automate assessment and selection of upscaling/variation images

## 4K Upscaling (Optional)

This project can optionally run a post-processing step to upscale the generated
image to a "4K" long-edge target (default: 3840px) using the open-source
Real-ESRGAN NCNN Vulkan portable executable.

### Install Real-ESRGAN (NCNN Vulkan)

1. Download the portable executable for your OS from the Real-ESRGAN project:
   https://github.com/xinntao/Real-ESRGAN

2. Download the model files (param/bin) and place them in a `models` directory
   next to the executable, or provide an explicit `model_path` in config/CLI.

3. Ensure the binary can be found by the project by doing one of:
   - Put it on your PATH (so `realesrgan-ncnn-vulkan` is executable), OR
   - Set environment variable `REALESRGAN_NCNN_VULKAN_PATH` to the full path,
     OR
   - Set `upscale.realesrgan_binary` in config.

Note: The Vulkan backend generally requires a Vulkan-compatible GPU.

### Configure

Add an `upscale` section to your config (example YAML):

```yaml
upscale:
  enabled: true
  target_long_edge_px: 3840
  engine: realesrgan-ncnn-vulkan
  # Optional. If omitted, PATH + REALESRGAN_NCNN_VULKAN_PATH are checked.
  realesrgan_binary: null
  # Optional. Directory containing the *.param/*.bin model files.
  model_path: null
  model_name: realesrgan-x4plus
  tile_size: 0
  tta: false
  # If true and Real-ESRGAN isn't available, fall back to a Lanczos resize.
  # Default is false (fail loudly).
  allow_fallback_resize: false
```

When enabled, the script saves an additional file named:

`<generation_id>_image_4k.jpg`

### Manual Upscaling CLI

To manually test upscaling on any image, use the helper script:

```
python scripts/manual_upscale.py path/to/image.jpg
```

- Writes the result next to the input as `image_4k.jpg` (same extension preserved).
- Defaults to a 3840px long edge; override with `--target-long-edge 4096`, etc.
- Provide a custom Real-ESRGAN binary with `--realesrgan-binary /path/to/realesrgan-ncnn-vulkan`.
- Provide a custom models directory with `--model-path /path/to/models` (folder containing *.param/*.bin).
