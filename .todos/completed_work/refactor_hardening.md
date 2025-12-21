Below is an agent-facing change spec for the next hardening pass. It’s written to eliminate the remaining “silent fallback” behaviors, improve artifact reliability (especially transcripts on failure), and reduce test brittleness—without over-engineering or rewriting your prompts.

This is based on the current implementation in main.py, pipeline.py, records.py, utils.py, and test_refactor_main.py.


---

Task: Hardening pass — fail-fast pipeline, reliable artifacts, less brittle tests

Goal

Make the refactor behave like a robust experimental harness:

No silent fallbacks: if a prompt factory returns None, or the model returns None, or image saving fails, we must fail loudly with actionable errors.

Reliable artifacts: JSON transcript should be written on success and on failure (once RunContext exists).

Cleaner future experimentation: adding steps/agents shouldn’t require rewriting tests that rely on step indices or specific system prompt substrings.

Keep it minimal: don’t redesign prompt content; keep orchestration simple and readable.


Non-goals

Do not change the prompt content/wording beyond what’s necessary to support testing (tests may monkeypatch prompt functions).

Do not implement multi-agent enclaves.

Do not add new infra (DB/UI/service).

Do not introduce “best-effort” behavior that hides errors.



---

Why these changes are needed (current gaps)

1. Silent prompt/response fallback in pipeline
ChatStep.render_prompt() and ChatRunner.run_step() convert None to "", which can cause the system to send empty prompts or record empty responses without alerting you. 


2. Silent image save failure
utils.save_image() catches exceptions and prints instead of raising, allowing runs to “succeed” while not actually writing an image. Same issue exists for generate_file_location() returning None implicitly on failure. 


3. Profile NaN footgun
generate_first_prompt() uses dropna() for Likes but not for Dislikes, so a blank dislike can crash with ", ".join(...) or, worse, produce confusing output types. 


4. Transcript is not guaranteed on pipeline failure
run_generation() only attempts transcript writing in the exception handler around the title/manifest/image pipeline, not for failures during the LLM step pipeline. 


5. Integration test brittleness
The offline integration test assumes the DALL·E capture is step index 7 and infers title-generation mode by a magic substring in the system prompt—both are fragile against normal experimentation.




---

Functional requirements

FR1 — Fail-fast prompt validation (no empty prompts by accident)

Requirements

If a step’s prompt factory returns None, raise ValueError with the step name.

If it returns a non-string type, raise TypeError (do not silently str() it).

If it returns empty/whitespace-only text, raise ValueError by default.


Allowing empty prompts deliberately (escape hatch)

Add a per-step opt-in flag:

ChatStep.allow_empty_prompt: bool = False


If allow_empty_prompt=True, empty/whitespace prompts are permitted (but still not None).

Acceptance

A bug that returns None or "" cannot produce “mysterious model behavior”; it fails immediately with a clear error that points to the step.


Touches: pipeline.py (ChatStep.render_prompt, ChatRunner.run_step). 


---

FR2 — Fail-fast response validation (no empty responses by accident)

Requirements

If ai_text.text_chat(...) returns None, raise ValueError (include step name).

If it returns non-string type, raise TypeError (don’t str() it).

If it returns empty/whitespace-only text:

default behavior: raise ValueError

optional escape hatch: ChatStep.allow_empty_response: bool = False



Why

The current behavior silently turns None into "" and proceeds. That’s indistinguishable from a valid but empty response, and it poisons downstream steps. 


---

FR3 — Always write transcript JSON once RunContext exists

Requirements

In run_generation(), once ctx has been constructed, any exception (pipeline steps, image generation, save, manifest writing, CSV writing) must trigger:

1. recording an error object on the context


2. best-effort transcript JSON write


3. re-raising the original exception




Error object (minimum)

Add to RunContext:

error: dict[str, Any] | None = None


When an exception occurs, set:

ctx.error = { "type": exc.__class__.__name__, "message": str(exc) }

(Optional but useful) add "where" like "phase": "pipeline" / "phase": "image_pipeline" or "step": step.name when applicable.


Transcript behavior on transcript-write failure

If transcript writing fails during error handling:

log exception via logger.exception(...)

do not swallow the original error (re-raise original).



Touches: main.py (run_generation), pipeline.py (optionally annotate error with step context), transcript.py (include ctx.error if present).


---

FR4 — Utilities must raise (no print-and-continue)

Requirements

Update utils.py:

1. save_image(...)



Must raise on any error.

Must not catch and print exceptions without re-raising.

Must not reference requests.exceptions (saving an image shouldn’t catch network errors).

Must align with main.run_generation() usage (it currently calls save_image(..., caption_text=..., caption_font_path=...)).

If your real utils.save_image already supports captions, keep it; just ensure it raises.

If not, either:

add those kwargs (and implement captioning), or

remove those kwargs from main.py and explicitly note that captioning is unavailable (less preferred since your pipeline expects identifiers).




2. generate_file_location(...)



Must raise if path join fails; must not return None silently. 


3. Remove or gate prints



Replace print(...) statements in utilities with either:

returning values, or

logging by passing a logger (optional).


For now, simplest: remove prints.


Acceptance

If the filesystem is unwritable / invalid output path / PIL fails, the run fails clearly and does not proceed to write misleading CSV rows or manifests.



---

FR5 — Fix Dislikes NaN/type safety in generate_first_prompt

Requirements

In generate_first_prompt():

Likes:

likes = user_profile["Likes"].dropna().astype(str).tolist()


Dislikes:

dislikes = user_profile["Dislikes"].dropna().astype(str).tolist()



Then join.

This prevents crashes from NaN and prevents accidental non-string entries causing join failures. 


---

FR6 — Rclone config must not “default to empty strings” when enabled

Requirements

In run_generation() you currently call:

remote=cfg.rclone_remote or ""
album=cfg.rclone_album or ""

That is a silent fallback if those fields are missing. 

Implement:

If cfg.rclone_enabled:

require cfg.rclone_remote and cfg.rclone_album be non-empty

otherwise raise ValueError("rclone.enabled=true but rclone.remote/album missing")



Preferably enforce this in RunConfig.from_dict() so config is validated centrally. (RunConfig already returns warnings; use that mechanism if you want a transitional “warn then error later,” but default should be strict.)

Then call upload_to_photos_via_rclone(..., remote=cfg.rclone_remote, album=cfg.rclone_album) with no or "".


---

FR7 — Always close logger handlers (even on exceptions)

Requirements

run_generation() currently flushes/closes handlers only on success path. If any exception occurs before the cleanup code, file handles can leak. 

Fix:

Wrap the entire body after logger creation in a try: ... finally: that always flushes/closes handlers and clears them.

Do not let handler flush/close failures hide the underlying run error (swallow handler close errors, but not run errors).



---

FR8 — Tests must not depend on step indices or hidden prompt internals

Requirements

Update test_integration_offline_run_generation_writes_artifacts to avoid:

assuming the DALL·E step is index 7

detecting title generation by a magic substring in the “system” message


Current brittleness: 

Replace with monkeypatch approach (recommended)

In the test:

1. Monkeypatch main.generate_dalle_prompt to return a sentinel string, e.g. "__TEST_DALLE_PROMPT_REQUEST__".

The pipeline will call this function via your lambda steps. 



2. FakeTextAI returns "A test image prompt" when the last user message equals the sentinel; otherwise return deterministic strings.


3. Monkeypatch main.generate_title to return a simple object with a .title attribute:

e.g. types.SimpleNamespace(title="Test Title", title_source="test", title_raw="Test Title")

This avoids coupling to the internal behavior of titles.generate_title (which may change prompt structure).




This makes the test resilient to step list growth/reordering.

Add new tests (fast unit tests)

Add unit tests for new fail-fast behavior:

Pipeline prompt validation:

a step whose prompt factory returns None raises ValueError

a step returning "   " raises ValueError

a step returning 123 raises TypeError


Pipeline response validation:

FakeTextAI returning None raises ValueError

FakeTextAI returning "" raises ValueError


Transcript on pipeline failure:

Make FakeTextAI raise on step “b”

Assert transcript JSON exists and contains error and already-recorded steps.



These directly enforce the “no silent fallbacks” policy and the “transcript on failure” requirement.


---

Implementation instructions (agent-facing)

Step 0 — Identify files to change

pipeline.py: prompt/response validation + (optional) error propagation hooks 

main.py: generate_first_prompt, run_generation error handling, rclone gating, logger cleanup 

utils.py: make save_image and generate_file_location raise; align signature with usage

test_refactor_main.py: make integration test robust; add new tests 

transcript.py (not shown): include ctx.error if present (backwards compatible)


Step 1 — Update ChatStep and pipeline validation

In pipeline.py:

Add fields:

allow_empty_prompt: bool = False

allow_empty_response: bool = False



Update render_prompt():

if text is None: raise ValueError(f"Step {name} produced None prompt")

if not isinstance(text, str): raise TypeError(...)

if not text.strip() and not allow_empty_prompt: raise ValueError(...)

return text (original string)


Update ChatRunner.run_step():

After text_chat returns:

validate response with same rules (None/type/empty)


Only append to ctx.messages after validation.

Only append step record after validation.


Step 2 — Make run_generation() write transcript on any failure after ctx exists

In main.py run_generation():

Create ctx = None before entering the run.

After constructing ctx, wrap the remainder in:


try:
   ... run steps, generate image, write CSV, write transcript ...
except Exception as exc:
   if ctx is not None:
      ctx.error = { ... }
      try: write_transcript(...)
      except: logger.exception(...)
   raise
finally:
   close handlers

Also:

remove the narrow try/except that only wraps title/manifest/image pipeline, or keep it but ensure it re-raises into the outer exception handler (avoid double transcript writes if you want).


Step 3 — Fix dislikes NaN/type handling

In generate_first_prompt():

apply .dropna().astype(str) to dislikes (and likes, for consistency). 


Step 4 — Fix rclone config handling

Prefer to enforce in RunConfig.from_dict() (central validation). If cfg.rclone_enabled:

require rclone.remote, rclone.album

error if missing/blank


Then in run_generation():

call upload_to_photos_via_rclone(... remote=cfg.rclone_remote, album=cfg.rclone_album) (no or ""). 


Step 5 — Make utilities raise (no print-and-continue)

In utils.py:

update save_image to raise exceptions.

update generate_file_location to raise exceptions.

remove prints (or gate behind a verbose flag if you really want them, but default should not print). 


Important: ensure signature matches main.run_generation() calls. Right now main.py calls save_image(..., caption_text=..., caption_font_path=...). 

Step 6 — Fix tests

In test_refactor_main.py:

update integration test to:

monkeypatch main.generate_dalle_prompt to sentinel

monkeypatch main.generate_title to stable object

stop relying on step indices and system-prompt substring checks


add new fail-fast tests as described in FR8.



---

Documentation updates

Update README or a short docs/hardening.md describing:

Pipeline now fails on:

None / empty prompts

None / empty responses

image save failures


Transcript JSON is written on failure once ctx exists (and contains an error section).

If rclone enabled, remote/album are mandatory.


(Keep it short—this is mainly “behavioral contract” documentation.)


---

Acceptance criteria (Definition of Done)

A step prompt factory returning None causes a clear exception before calling the model.

A model returning None or empty text causes a clear exception before recording a successful step.

Image save failures raise and abort; no misleading “success” artifacts are written.

Transcript JSON is created for both successful runs and failed runs (after ctx is created), and failed-run transcript includes error.

Integration test does not depend on step indices or internal title-generation prompt strings.

All pytest tests pass offline.



---

Pitfalls to avoid

Don’t “fix” prompt validation by defaulting None → "". That’s exactly what we’re removing. 

Don’t catch exceptions in save_image and only print—must raise. 

Don’t add validation so strict it blocks intentional experiments; provide explicit opt-ins (allow_empty_prompt/response) rather than implicit behavior.

Don’t write transcripts twice on error unless you intend to (ensure the error handler structure is clean and deterministic).