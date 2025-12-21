Task: Refactor `main.py` into a minimal, step-driven “prompt pipeline” with strict config validation and structured run artifacts (JSON transcript + stable metadata)

## Goal

Make the generation script **cheap to iterate on** without rewriting orchestration every time you add/reorder/remove a prompt phase.

Specifically:

* **Open to extension:** adding a new prompt phase should usually be “add 1 entry to a list” (not copy/paste another `message_with_log(...)` block).
* **Closed to modification:** the “engine” of the pipeline should not require changes when experiments change.
* **No silent fallbacks:** missing/invalid config should fail fast with clear errors; any “default” behavior must be explicit + logged.
* **Living script, clarity > cleverness:** keep the minimal working version straightforward, testable, and well-documented.

This task is written to be executable by an agent who has not seen this chat. It assumes the current repo resembles the uploaded files, especially `main.py`, `utils.py`, and `message_handling.py`.

---

## Current behavior summary (what must remain working)

Today, `main.py` does roughly:

1. Load YAML config via `utils.load_config()` 
2. Load categories CSV + user profile CSV (pandas). 
3. Build an initial prompt using randomly selected category concepts (via `random.choice`). 
4. Run a multi-step LLM conversation using `TextAI.text_chat(...)` and `MessageHandler` to accumulate messages.
5. Produce a final image prompt (DALL·E oriented), then call `ImageAI.generate_image(...)`, decode base64, and save a JPG via `utils.save_image(...)`.
6. Write some metadata to CSV (`utils.save_to_csv`).
7. Write a transcript log (currently `str(messages_log.messages)` written to a `.txt` file). 

**Do not change the visible output semantics** (it should still generate an image and write artifacts) unless explicitly required below.

---

## Non-goals (do NOT do these in this task)

* Do not redesign prompts or reword prompt content.
* Do not implement multi-agent enclaves yet.
* Do not introduce a database, UI, or web service.
* Do not introduce new runtime dependencies (keep it stdlib + existing deps).
* Do not change the underlying `ai_backend` behavior or swap models unless required for the refactor.
* Do not “best-effort” missing configuration by silently defaulting to CWD, empty strings, etc.

---

## Functional requirements

### FR1 — Strict config validation (fail loud, fail early)

**Problem today:** config lookups are scattered; some values can be missing and the run will proceed until later failure (or write to unexpected locations).

**Requirement:**

* Create a single config parsing/validation layer that:

  * Validates required keys exist and are non-empty strings.
  * Raises `ValueError` with a precise message containing the full key path (e.g., `Missing required config: prompt.categories_path`).
  * Performs minimal normalization (e.g., expanduser, abspath) if helpful.
* Every path that controls artifact locations must be validated. At minimum, validate keys currently used:

  * `image.save_path` (or the repo’s current equivalent)
  * `prompt.categories_path`
  * `prompt.profile_path`
  * `prompt.generations_path`
* If you introduce any new config keys for the refactor, they must be documented and tested.

**No silent fallback policy:**

* If a config value is missing → raise.
* If you must keep a default for compatibility, it must be:

  * explicitly logged at **WARNING** level, and
  * easy to disable later (e.g., `strict_config: true` default).

### FR2 — Step-driven prompt pipeline (declarative steps list)

**Problem today:** `main()` has many repeated `message_with_log(...)` calls, making adding/reordering steps tedious and error-prone. 

**Requirement:**

* Introduce a minimal abstraction for prompt phases:

  * A `ChatStep` (dataclass) that defines:

    * `name: str` (unique)
    * `prompt: str | Callable[[RunContext], str]` (static or computed)
    * `temperature: float` (and any other model params you already pass)
    * optional `capture_key: str | None` (store output in context for later steps)
  * A `ChatRunner` that executes steps consistently:

    * Appends user prompt message
    * Calls `TextAI.text_chat(...)`
    * Appends assistant response message
    * Records step metadata (name, prompt text, response length, etc.)
* Refactor `main()` so that the sequence of prompt steps is defined in **one list**, in the existing order, and executed in a `for` loop.

**Preserve behavior:** step order and prompt strings must remain the same as current `main.py` unless you have a bugfix justification.

### FR3 — Separate “conversation state” from “run recording”

**Problem today:** logging is interleaved with conversation; transcript is written as a Python `str(list)` dump. 

**Requirement (minimal):**

* Keep the **conversation state** as `MessageHandler` (or equivalent).
* Add a **RunRecorder** concept (can be very small) that collects:

  * step name
  * prompt text (as sent)
  * response text (as received)
  * model params used (temperature, model name if available)
  * timestamps (optional but helpful)
* Write the transcript to disk as JSON (not Python repr). Include:

  * `generation_id`
  * `seed` (see FR4)
  * `selected_concepts` (the random concepts chosen for the run)
  * steps array with `{name, prompt, response, params, created_at}`
* Do not silently skip transcript writing; if it fails, raise an exception (and log it).

### FR4 — Reproducible randomness (seeded RNG, logged)

**Problem today:** random concept selection uses global `random.choice(...)` so runs are not reproducible. 

**Requirement:**

* Introduce a run seed:

  * Prefer `prompt.random_seed` (or similar) in config.
  * If no seed is provided, generate one (e.g., from time) but **log it at INFO** so it isn’t “silent randomness.”
* Use a local `random.Random(seed)` instance and pass it into concept selection:

  * Update `get_random_value_from_group(group, data, rng)` so it uses `rng.choice(...)`.
* Include the seed and selected concepts in JSON transcript metadata.

### FR5 — Eliminate “print-driven” output; use logger (still human-friendly)

**Problem today:** `print()` is used heavily for section separators and responses. 

**Requirement:**

* Replace section `print(...)` calls with structured logging:

  * `logger.info("Step: %s", step.name)`
  * `logger.info("Received response for %s (chars=%d)", step.name, len(resp))`
* Avoid logging full LLM responses at INFO by default (logs become unreadable).

  * Put full responses in JSON transcript and/or `logger.debug`.
* Keep the developer experience good:

  * INFO logs should show step names and progress clearly.

### FR6 — Fix CSV schema mismatch and remove implicit list-based writes

**Problem today:** `utils.save_to_csv()` writes headers for 4 columns but `main.py` writes only 3 values (`[generation_id, gen_keywords, dalle_prompt]`). This produces malformed CSV.

**Requirement (minimal working):**

* Replace `save_to_csv(list)` usage with a dict-based writer:

  * New function: `append_generation_row(path, row: dict, fieldnames: list[str])`
  * Use `csv.DictWriter`
  * Explicit fieldnames and required keys enforced (raise on missing)
* Define a stable schema. Minimum recommended columns:

  * `generation_id`
  * `selected_concepts` (stringified safely, e.g. JSON string)
  * `final_image_prompt`
  * `image_path`
  * `created_at` (ISO8601)
* Update any documentation referencing the old CSV format.

---

## Implementation instructions (agent-facing)

### Step 0 — Read and map the current pipeline

Locate in `main.py`:

* where config is loaded (`load_config`)
* where prompts are generated (`generate_first_prompt`, etc.)
* where `TextAI.text_chat()` is called
* where `ImageAI.generate_image()` is called and image is saved (`save_image`)
* where artifacts are written (CSV + transcript `.txt`)

Write down the exact order of current prompt steps so you can preserve it during refactor.

### Step 1 — Add `run_config.py` (or `config_schema.py`)

Implement a small dataclass:

* `RunConfig` with fields for all required paths and optional settings (seed, model, temperatures).
* `RunConfig.from_dict(cfg: dict) -> RunConfig`

  * Raises `ValueError` with clear messages.
  * Normalizes paths.
  * Does not create directories (leave that to runtime).

**Pitfall to avoid:** do not spread `config['prompt']['x']` lookups across the code after this change.

### Step 2 — Add `pipeline.py` with `ChatStep`, `RunContext`, `ChatRunner`

Minimum recommended shapes:

* `RunContext`

  * `generation_id: str`
  * `cfg: RunConfig`
  * `logger: logging.Logger`
  * `rng: random.Random`
  * `selected_concepts: list[str]`
  * `messages: MessageHandler`
  * `outputs: dict[str, Any]` (captures like “dalle_prompt”)
  * `steps: list[dict]` (recorded steps for transcript)

* `ChatStep`

  * `name`
  * `prompt_factory(ctx) -> str` OR `prompt: str`
  * `params: dict` (temperature, etc.)
  * `capture_key: Optional[str]`

* `ChatRunner.run_step(ctx, step) -> str`

  * Generates prompt text
  * Mutates `ctx.messages` by appending user/assistant entries (consistent with `MessageHandler.continue_messages`) 
  * Calls `TextAI.text_chat(ctx.messages.messages, **params)` 
  * Records `{name, prompt, response, params, created_at}` into `ctx.steps`
  * Stores captured output if `capture_key` is set

**Pitfall to avoid:** don’t accidentally use copies of `MessageHandler` such that later steps lose conversation history.

### Step 3 — Refactor concept selection to be seeded + deterministic

Modify in `main.py` (or move into a new helper module):

* `get_random_value_from_group(group, data, rng)` — uses `rng.choice`
* `generate_first_prompt(..., rng)` — uses the injected RNG

Record `selected_concepts` in `ctx.selected_concepts`.

**Pitfall to avoid:** continuing to use `random.choice` anywhere in the pipeline that affects outputs.

### Step 4 — Implement JSON transcript writing (`transcript.py`)

Add:

* `write_transcript(path: str, ctx: RunContext) -> None`

  * Writes UTF-8 JSON with indent
  * Includes:

    * `generation_id`
    * `seed`
    * `selected_concepts`
    * `steps` (as recorded)
    * `image_path` (if created)
    * `created_at`
  * If writing fails → raise.

Replace the existing `.txt` transcript behavior. (If you must keep the `.txt` for backwards compatibility, write both, but document it and ensure neither is silent.)

### Step 5 — Fix generation CSV writing

Implement a new utility (preferably in `utils.py` or a new `records.py` module):

* `append_generation_row(csv_path, row: dict, fieldnames: list[str])`

  * Uses `csv.DictWriter`
  * Creates file + header if missing
  * Validates required keys

Update `main()` to write a correct row including `image_path`.

**Pitfall to avoid:** do not keep the broken list-based `save_to_csv` call as-is. 

### Step 6 — Update `main()` orchestration minimally

`main()` should become:

1. Load + validate config into `RunConfig`.
2. Initialize logger.
3. Initialize RNG with seed.
4. Load prompt data + user profile.
5. Build initial prompt + selected concepts.
6. Initialize `TextAI`, `MessageHandler`, `RunContext`.
7. Define `steps = [...]` list (existing order).
8. Execute the step loop.
9. Pull captured `dalle_prompt` from `ctx.outputs`.
10. Generate image + save.
11. Write generation CSV row.
12. Write transcript JSON.
13. Log completion.

**Strong guidance:** Keep prompt text functions (`generate_second_prompt`, etc.) unchanged for MWV.

---

## Testing requirements (must add; offline; no network)

Project test runner is `python -m unittest discover -s tests` per `pyproject.toml`. 
Use `unittest` + `unittest.mock` (do NOT require external API calls).

### Unit tests

1. **Config validation**

   * Missing `prompt.categories_path` → raises `ValueError` with message containing that key.
   * Missing `image.save_path` → raises.
   * Empty strings treated as missing.

2. **Seeded randomness determinism**

   * With a fixed seed and a tiny DataFrame of categories, the selected concepts are stable across runs.

3. **Pipeline step execution order**

   * Mock `TextAI.text_chat` to return deterministic strings (e.g., `f"resp:{step_name}"`).
   * Assert `ctx.steps` records appear in the same order as `steps` list.
   * Assert capture behavior: the step with `capture_key="dalle_prompt"` ends up in `ctx.outputs["dalle_prompt"]`.

4. **Transcript JSON validity**

   * Run a minimal ctx through `write_transcript`
   * Load JSON back and assert required keys exist.

5. **CSV writer**

   * Creates file with header
   * Appends row and reads back with `csv.DictReader`
   * Fails (raises) if required keys missing

### Integration test (offline end-to-end)

* Patch `load_config()` to return an in-memory dict with temp directories. 
* Create temporary CSVs for:

  * categories
  * user profile
* Patch `TextAI` and `ImageAI`:

  * `TextAI.text_chat` returns deterministic “responses”
  * `ImageAI.generate_image` returns a base64 encoded small test image (generate via PIL in-memory)
* Run the pipeline function (ideally refactor `main()` so core logic is callable as `run_generation(cfg_dict)` for testability).
* Assert:

  * image file exists
  * transcript JSON exists and parses
  * CSV row exists with `generation_id` and `image_path`

**Pitfall to avoid:** calling the real OpenAI backend in tests.

---

## Documentation updates (must do)

Update `README.md` (or add `docs/pipeline.md`) to include:

* What the pipeline is (step-driven list)
* How to add a new prompt step (add to steps list; optional capture_key)
* Config keys required (explicit list)
* Artifacts written:

  * image output path
  * generation CSV schema + location
  * transcript JSON schema + location
* How to run:

  * `pdm run generate`
  * `pdm run test` / unittest discover

---

## Acceptance criteria (Definition of Done)

* Running the generation workflow still produces:

  * a saved JPG image
  * a correct generation CSV row (no schema mismatch)
  * a JSON transcript containing step-by-step prompts/responses and run metadata
* Adding a new prompt phase requires editing **only** the `steps = [...]` list (and optionally adding a prompt function).
* Missing required config keys fail immediately with explicit error messages (no “defaults to cwd” behavior without a warning).
* All tests pass locally with no network access using the existing unittest runner.

---

## Notes / pitfalls to avoid

* **Do not break conversation continuity.** The same `MessageHandler` instance must accumulate messages across all steps. 
* **Do not log megabytes at INFO.** Keep INFO concise; store full prompts/responses in transcript JSON.
* **Do not “helpfully” default paths.** If a value is missing, raise; if you must default for compatibility, log a WARNING and document it.
* **Do not change prompt text.** Any changes to prompt content should be a separate task (to keep refactor risk low).

