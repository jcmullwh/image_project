Below is a “handoff spec” you can give to an implementation agent who hasn’t seen this chat and doesn’t know your repo. It’s written to be minimally invasive and to fail loudly when invariants break.


---

Task: Add human-friendly per-image identifiers using sequence + LLM title

Goal

For every generated image, add a small, non-distracting, reliably visible identifier so a single viewer can reference images during feedback.

The identifier must be:

Human-friendly: e.g., #042 — Turquoise Citadel

Always present: not dependent on the image model drawing it

Non-intrusive: should not look like a “badge” or a giant watermark

Mapped to canonical internal IDs: so we can always resolve the reference to the real generation record


We are generating dozens to a few hundred images for one person, so sequential numbering is preferred over UUIDs.

Non-goals

Do not implement steganographic watermarks.

Do not build a UI.

Do not introduce complex DB storage; a CSV manifest is enough for v1.

Do not “hope the model draws the title” in the image; the overlay must be post-processing.



---

Functional requirements

FR1 — Sequence number

Each generated image must be assigned a monotonically increasing integer seq:

Format for display: #001, #002, …

Persist across runs by reading an on-disk manifest and allocating max(seq)+1.

If the manifest is missing/empty, start at 1.


FR2 — LLM-generated title

Each image must have a short title generated via your existing text/LLM backend (whatever the current abstraction is):

Input to title generation: the final image prompt (the exact prompt used for the image model), plus optionally a small “avoid list” of recent titles.

Output constraints (hard):

2–4 words

Title Case

No quotes

No proper nouns (no people/place/brand names)

No punctuation except optional hyphen

Return only the title text



Collision handling:

If generated title duplicates an existing title in the manifest (case-insensitive), reprompt up to 2 times.

If still duplicated or invalid, append a tasteful disambiguator (e.g., II, III) and record the final title.


Failure handling (minimal working):

If title generation fails due to API error/timeouts: either

(Preferred) fail the whole generation with a clear error, OR

(If you already have a “best-effort” mode) fall back to Untitled and mark it in manifest (title_source="fallback").
Pick one consistent behavior and document it.



FR3 — Always-visible caption overlay

When saving the final image file, overlay a small caption containing sequence + title:

Caption text: #042 — Turquoise Citadel

Placement: bottom-center or bottom-right.

Presentation: “gallery caption” style (subtle). Examples:

a thin matte strip at the bottom (e.g., 3–4% of image height) with slightly darkened background and light text; OR

text drawn onto a small translucent rectangle.


Must be applied after image bytes are returned (PIL post-process), so it’s always present.


FR4 — Manifest row written per image

Append one row per image to a CSV manifest that can resolve viewer references. Minimum columns:

seq (int)

title (str)

generation_id (your existing unique identifier)

image_prompt (final prompt sent to the image model)

image_path (full or relative path)

created_at (ISO8601) Optional but useful:

model / size / quality / seed (if present)

title_source (llm|fallback)

title_raw (if you want auditability)


Do not rely on an implicit column order with list-based CSV writes; use a DictWriter with explicit fieldnames.


---

Implementation instructions (agent-facing)

Step 0 — Locate pipeline entry points

Search the repo for:

Where images are generated (call to DALL·E / GPT-image / your backend wrapper)

Where image bytes are saved to disk (likely a save_image() helper using Pillow)

Where per-generation logs/metadata are written (CSV, JSON, etc.)


You’ll integrate at three points:

1. right after the final image prompt is finalized (generate title)


2. right before saving the image (overlay caption)


3. after saving (append manifest row)



Step 1 — Add manifest utilities

Create or update a small module (existing utils.* or new manifest.py) with:

read_manifest(manifest_path) -> list[dict]

Returns empty list if file does not exist.

Uses csv.DictReader.

Must handle missing columns gracefully (older manifests).


get_next_seq(manifest_path) -> int

Reads manifest, returns max(seq)+1 (robust to blank/invalid values).


append_manifest_row(manifest_path, row: dict, fieldnames: list[str])

Creates file with header if missing.

Ensures required fields exist; otherwise raise.


Concurrency (minimal):

If you already have a locking utility, use it.

Otherwise add a simple lock file approach (manifest.csv.lock) to protect “read max + append”.


Step 2 — Add title generation

Implement:

generate_title(image_prompt: str, avoid_titles: list[str]) -> str

Uses existing text backend (whatever abstraction you currently use).

Enforces constraints in code:

strip quotes

collapse whitespace

validate word count

validate allowed characters

reject if contains obvious proper-noun signals (e.g., any word with internal capital not at start, or presence of known banned tokens). Keep it simple; don’t over-engineer.



Collision loop:

Load existing titles (or last N titles) from manifest.

Retry with stronger “must be distinct” instruction.

If still collision: suffix II/III.


Important: keep the prompt for title generation deterministic and short. Example structure:

System/Instruction:

“Return a 2–4 word Title Case name…” User:

“Prompt: …”

“Do not reuse these titles: …”


Step 3 — Caption overlay in save path

Modify (or wrap) the image-saving function so you can pass caption_text.

save_image(image_bytes, out_path, caption_text: str | None)

Decode bytes to PIL Image

If caption_text:

draw caption strip/box + text using PIL’s ImageDraw

use a default font if a packaged font isn’t available; optionally allow configuring a TTF path


Save to the same output format you already use (likely JPG)

Keep parameters consistent (quality/subsampling) if your pipeline already sets them.


Keep the overlay subtle and consistent across image sizes:

Use proportional sizing (e.g., font size ~2–3% of image height)

Avoid hard-coded pixel values that break at 512 vs 2048+.


Step 4 — Tie into generation loop

In the main generation loop/function:

1. seq = get_next_seq(manifest_path)


2. title = generate_title(final_image_prompt, avoid_titles=recent_titles)


3. Generate image bytes using the image model


4. Save image with caption f"#{seq:03d} — {title}"


5. Append manifest row



Do not change the internal generation_id semantics. The manifest is the bridge.

Step 5 — Documentation

Update README.md (or add docs/identifiers.md) with:

What the visible identifier looks like

How the viewer should give feedback (“just tell me #042 or the title”)

Where the manifest lives and what columns mean

How sequence allocation works (global monotonic)

What happens if title generation fails (fail-fast vs fallback)



---

Testing requirements (must add)

Use pytest (or your existing framework). Add tests that do not require external APIs.

Unit tests (fast)

1. Sequence allocation



Given no manifest → next_seq = 1

Given manifest with seq 1..N → next_seq = N+1

Handles malformed rows (blank seq, non-int) safely


2. Manifest append



Creates file + header when missing

Appends row; later read returns it exactly


3. Title validation



Accepts valid titles

Rejects too-long, quotes, punctuation, 1-word, 5+ words

Collision handling: if existing title, retries, then appends II deterministically (mock the LLM return)


4. Caption overlay



Given a blank test image, overlay caption, save, reload:

verify output dimensions unchanged

verify some pixels in caption region differ from original (basic sanity)

do not attempt OCR



Integration test (offline)

Mock your AI backends:

Mock “image generation” to return known bytes (a generated PIL image to bytes).

Mock “title generation” to return a stable title. Run one generation end-to-end into a temp directory:

asserts:

image file exists

caption overlay applied (pixel diff)

manifest row exists with correct seq/title/prompt/path




---

Acceptance criteria (definition of done)

Running the generation workflow produces images with visible caption #NNN — Title.

seq increments across runs using the manifest.

A manifest CSV exists and can resolve seq/title → generation_id/image_path/prompt.

Tests pass locally without network access.

Documentation explains the scheme and how to provide feedback.



---

Notes / pitfalls to avoid

Do not embed long IDs in the image.

Do not depend on the image model to render text.

Do not use list-based CSV writes (column drift will break you later).

Keep the overlay subtle; the point is referenceability, not branding.


If you want the agent to keep changes ultra-minimal: tell them to avoid renaming files/functions and implement as small wrappers around existing save + logging utilities.