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