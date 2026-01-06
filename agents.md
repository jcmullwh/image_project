# PROJECT_SPEC_TEMPLATE.md

Project-specific spec template and guidance for this repo (the “image_project” pipeline).

Use this when writing implementation specs (for humans or agents). It is tuned to the actual architecture and recurring failure modes in this codebase.

This doc assumes the canonical entry point is `main.py`’s `run_generation()` and the Step/Block prompt pipeline built there. :contentReference[oaicite:0]{index=0}

---

## What this repo is

A **living experimentation harness** that:

- builds multi-step LLM prompt pipelines (Step/Block) to produce an image prompt,
- generates an image and saves it with a caption,
- writes structured artifacts (JSON transcript + CSV records + titles manifest),
- supports iterative experimentation (adding/removing stages, adding refinement loops, future toolcalls/agentic subroutines).

The primary “product” is not just the image—it’s the **repeatable run** and the **audit trail**.

---

## Core invariants

### 1) No silent fallbacks
If something is missing, invalid, or fails, the system must:
- raise with a clear error (preferred), OR
- warn loudly and follow a documented best-effort behavior (ANY fallback MUST be human approved)

Avoid patterns that:
- set required modules to `None` and continue,
- swallow exceptions and keep running,
- default output paths to CWD without warning,
- accept mis-typed config (notably booleans).

### 2) Clarity beats cleverness
A reader should be able to understand “what runs” from the flow definition without decoding predicates, magic indices, or hidden behavior.

### 3) Determinism when randomness is used
When randomness affects prompts:
- seed must be recorded and logged,
- seed must be stored in transcript,
- runs must be reproducible with the same seed.

### 4) Transcript is canonical for debugging
Transcript JSON must contain:
- step-by-step prompts/responses + params,
- nested pipeline paths,
- seed, selected concepts,
- error metadata on failure.

Transcript is not model context; it’s a recorder.

### 5) Offline testability
All tests must run offline:
- mock/fake LLM and image backends
- no network calls
- no real external binaries (use fakes or platform-safe stubs)

---

## Repository architecture map (current)

### Canonical run path
`main.py`:
- `main()` loads config → calls `run_generation(cfg_dict, generation_id=...)` :contentReference[oaicite:1]{index=1}
- `run_generation()` phases:
  - seed selection
  - prompt data/profile load
  - prompt pipeline (Step/Block + ChatRunner)
  - capture `dalle_prompt`
  - image generation + save
  - titles manifest lock + seq allocation + title generation
  - optional upscaling
  - CSV record append
  - optional rclone upload
  - transcript write
  - on exception: set `ctx.error` and write transcript best-effort :contentReference[oaicite:2]{index=2}

### Pipeline execution
- `pipeline.py` provides `ChatStep`, `Block`, `ChatRunner`, `RunContext` (used from `main.py`). :contentReference[oaicite:3]{index=3}
- The pipeline is built as a root `Block(name="pipeline", ...)` and run via `runner.run(ctx, pipeline_root)`. :contentReference[oaicite:4]{index=4}
- Stages are commonly wrapped with a “refinement loop” block (`tot_enclave`) via `refined_stage(...)`. :contentReference[oaicite:5]{index=5}

### Artifacts
From `run_generation()`:
- Operational log: `<generation_id>_oplog.log` in `cfg.log_dir` :contentReference[oaicite:6]{index=6}
- Transcript JSON: `<generation_id>_transcript.json` in `cfg.log_dir` :contentReference[oaicite:7]{index=7}
- Generated image: `<generation_id>_image.jpg` in `cfg.generation_dir` (and optional `_image_4k.jpg` in `cfg.upscale_dir`) :contentReference[oaicite:8]{index=8}
- Titles manifest CSV (seq/title mapping): `cfg.titles_manifest_path` (locked during allocation/write) :contentReference[oaicite:9]{index=9}
- Generations CSV: `cfg.generations_csv_path` via `append_generation_row(...)` :contentReference[oaicite:10]{index=10}

---

## Spec template for this project

### Task
One sentence. Example: “Threaded enclave opinions: one response per artist, independent threads, then consensus.”

### Goal
Describe user-visible and developer-visible outcomes.

- What changes in outputs (transcript structure, prompts, images, manifest rows)?
- What becomes easier to experiment with?

### Context
What exists now and why it’s insufficient.

If pipeline-related, state:
- where the steps are defined (usually `run_generation()` pipeline section),
- what is currently captured (e.g., `capture_key="dalle_prompt"`),
- what currently gets merged back into conversation (e.g., stage wrapper uses `merge="last_response"`). :contentReference[oaicite:11]{index=11}

### Constraints
Include the ones that apply:

- No silent fallbacks
- Offline tests
- Deterministic randomness
- Cross-platform behavior (esp. upscaling tool stubs)
- Backwards compatibility for artifact schemas (if applicable)

### Non-goals
Explicitly list what you are not doing.

### Functional requirements
Numbered FRs. Each FR must specify:

- behavior
- inputs/outputs
- failure behavior (raise vs warn+fallback)
- observability (what will appear in transcript/logs)

#### Project-specific FR items to include when relevant
- **Config validation:** if you add config keys, define their path and validation rules. Fail on invalid types. (Especially booleans; do not rely on `bool(value)`.)
- **Pipeline paths:** if new steps are repeated, ensure transcript path uniqueness.
- **Capture keys:** if you capture outputs, specify naming scheme and collision behavior (raise on collision or stage-scoped keys).
- **Error attribution:** define what `ctx.error` must include (phase + pipeline path where possible). :contentReference[oaicite:12]{index=12}

### Proposed conversation flow (most important section)
Show “just the flow”. Use Steps/Blocks exactly as they will appear in code.

Rules for this repo:
- Avoid predicate lambdas to select steps (“where=lambda …”) in flow code.
- Prefer explicit lists (`except_steps=[...]`) or tags if selection is needed.
- If using closures inside loops, bind loop variables safely (default args / helper functions).

### Implementation plan
Ordered steps, including which files change. Typical touch points:
- `main.py` for pipeline construction
- `pipeline.py` if runner capabilities must expand
- `run_config.py` for config validation
- `transcript.py` if transcript fields change
- `tests/` for offline tests

### Error handling + observability contract
For each new failure point:
- What exception is raised?
- What error fields are recorded in transcript (`ctx.error`)?
- What log line appears at INFO/WARN?

For pipeline failures, ensure the exception includes enough context to record:
- `phase` (from `run_generation`)
- `pipeline_path` or step identifier (so transcript tells you where it died) :contentReference[oaicite:13]{index=13}

### Data/artifact changes
If you change:
- generations CSV schema
- titles manifest schema
- transcript schema
Document:
- the new fields
- backward compatibility/migration policy

### Testing requirements (pytest, offline)
**Rules for this repo:**
- No network
- Fake TextAI/ImageAI
- Do not test by matching prompt wording
- Do not depend on step indices
- Use temp dirs; assert artifacts exist and parse

Minimum recommended tests for pipeline behavior changes:
1) Unit tests for new semantics/invariants (e.g., merge behavior, isolation)
2) Integration test that runs `run_generation()` with mocked backends and asserts:
   - image file exists
   - transcript exists and contains expected structure
   - CSV row exists and matches schema

### Documentation updates
Update README or `docs/pipeline.md` with:
- what changed
- how to configure it
- how to interpret transcript paths/fields

### Acceptance criteria
Concrete, verifiable outcomes. Example:
- “Transcript contains one step record per enclave thread per stage.”
- “Thread steps do not influence each other (validated by a test inspecting model input messages).”
- “All tests pass offline.”

### Pitfalls to avoid (this repo’s greatest hits)
- Using `bool("false")` for config flags (turns “false” into True)
- Silent fallbacks on file paths or fonts
- Logging local step names only (repeated steps become ambiguous)
- Tests tied to prompt strings or step indices
- Closure late-binding bugs in loop-built prompt callables
- Writing partial artifacts and continuing after failures

---

## Project-specific guidance on “good specs”

### A good spec here is:
- **Flow-first:** the Step/Block shape is shown clearly with minimal code
- **Explicit about merge/capture semantics:** what enters conversation vs what is only recorded/captured
- **Strict about config:** types, defaults, and failures are spelled out
- **Testable offline:** tests validate behavior by inspecting inputs/outputs, not matching prompt text
- **Observable:** transcript/log effects are described so debugging doesn’t require guesswork

### A bad spec here:
- introduces hidden behavior or clever selection lambdas in the flow,
- relies on silent defaults (“if missing, just…”),
- makes testing dependent on prompt wording,
- does not say what happens on failure.

---

## Appendix: Current canonical pipeline outline (for reference)

Inside `run_generation()` the pipeline is built as:

- `pipeline_root = Block(name="pipeline", merge="all_messages", nodes=[ refined_stage(...), ... ])`
- each `refined_stage` wraps:
  - a `draft` ChatStep and a `tot_enclave` Block
  - and commits `merge="last_response"` so the stage retains only the refined final output
- `dalle_prompt_creation` is captured to `ctx.outputs["dalle_prompt"]` :contentReference[oaicite:14]{index=14}

Use this as the reference point when proposing modifications.

---
