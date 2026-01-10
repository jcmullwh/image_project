# Task

Implement an **optional black-box scoring + selection stage** inside the `image_project` generation pipeline:

* The pipeline generates multiple **discrete “idea cards”** (each with multiple options).
* A separate **LLM judge** scores each idea card **numerically** (strict JSON).
* The system **selects** one candidate (with optional exploration/diversity) and proceeds to final prompt generation.
* **No scoring rationale/feedback is returned or merged into downstream prompt context**.

This adds an internal quality “gradient” (best-of-N / selection pressure) without exposing preference details that cause over-steering. 

---

# Goal

## User-visible outcomes

* Generated images remain concept-aligned but show **more variety** across runs.
* If a user profile contains specific tastes (e.g., exact colors), the system avoids collapsing into those motifs every run by:

  * keeping the *generator* blind to overly-specific preference tokens, and
  * using black-box scoring + controlled exploration to balance alignment and novelty.

## Developer-visible outcomes

* Transcript JSON includes a clear record of:

  * generated candidates (as a step response),
  * judge scores (as a step response),
  * selection metadata (as structured transcript fields).
* When scoring is disabled, the pipeline behavior is unchanged. 

---

# Context

The repo is a step-driven prompting harness built around `ChatStep` and nested `Block`s, orchestrated in `main.py`’s `run_generation()`, and recorded into a canonical transcript JSON with `{name, path, prompt, response, params, created_at}` per step. 

It already emphasizes fail-fast behavior, deterministic seeding, and reliable artifacts (image file + transcript + CSV records). 

A “grading mechanism” is listed as a future direction; this spec implements it as **black-box scoring** that does **not** inject steering feedback into the generator. 

---

# Constraints

* **No silent fallbacks**: invalid config, invalid judge JSON, or missing outputs must raise with clear errors and be attributed in `ctx.error` when possible. 
* **Offline testability**: tests must not hit the network; use fake LLM/image backends. 
* **Determinism**: any randomness (candidate sampling, exploration selection) must be seeded from the run seed and recorded. 
* **Transcript remains canonical** for debugging; scoring must be observable there. 
* **Do not leak judge feedback** into downstream generation context (prevents “preference collapse” and rubric gaming). 

---

# Non-goals

* Do not implement image-based judging (vision model scoring) in this iteration.
* Do not change the titles manifest / sequencing / caption overlay logic. 
* Do not introduce any frontend/UI for score display.
* Do not tune prompts by matching specific strings in tests.

---

# Functional requirements

## FR1 — Config: enable/disable + validation

Add a new config section (suggested path) and validate types strictly:

```yaml
prompt:
  scoring:
    enabled: false
    num_ideas: 6              # int >= 2
    exploration_rate: 0.15    # float in [0, 0.5]
    judge_temperature: 0.0    # float, must be 0.0 by default
    judge_model: null         # optional string; if null use primary text model
    generator_profile_abstraction: true
    novelty:
      enabled: true
      window: 25              # int >= 0 (0 disables history usage)
```

Validation rules:

* `enabled` parsed strictly (do not rely on Python truthiness). 
* `num_ideas` must be an int ≥ 2; otherwise raise `ValueError` with full key path.
* `exploration_rate` must be float in `[0, 0.5]` (cap at 0.5 to keep behavior sane); else raise.
* `judge_temperature` must default to 0.0; if set to nonzero, allow but warn loudly (or fail if you want stricter determinism).
* Unknown keys are allowed only if the config system already supports that; otherwise fail-fast.

Observability:

* Log whether scoring is enabled and the key parameters at INFO.
* Transcript includes `scoring.enabled` and `scoring.config_snapshot`.

Failure behavior:

* Invalid types/values raise before any LLM calls. 

---

## FR2 — Candidate representation: “Idea cards with options”

When scoring is enabled, the pipeline must produce **N idea cards** in **strict JSON**.

Each idea card must include:

* `id` (string; unique within the candidate set)
* `hook` (one sentence pitch)
* `narrative` (short paragraph)
* `options` object with at least:

  * `composition` (list[str], length ≥ 2)
  * `palette` (list[str], length ≥ 2)
  * `medium` (list[str], length ≥ 1)
  * `mood` (list[str], length ≥ 1)
* `avoid` (list[str]) optional

Hard requirements:

* The JSON must be parseable; otherwise raise and attribute failure to the generating step.
* Idea IDs must be unique and stable (e.g., `"A".."F"`).
* The generator should be encouraged to keep options meaningfully distinct (not just synonyms).

Observability:

* The full JSON response is recorded in the transcript as a normal step response. 

---

## FR3 — Generator preference de-steering via profile abstraction

To avoid “user likes X → everything becomes X,” do **not** feed the raw likes/dislikes list verbatim into candidate generation.

If `prompt.scoring.generator_profile_abstraction: true`:

* Add a step to produce a **generator-safe profile summary**:

  * Allowed: broad adjectives, high-level style constraints, general “avoid” constraints.
  * Disallowed: explicit colors, named motifs, named artists, repeated n-grams from the raw likes list.
* Candidate generation uses **only**:

  * selected concepts
  * generator-safe summary
  * global constraints (e.g., “avoid horror if disliked” can remain, but avoid listing exact likes).

Judge scoring can still use the full raw profile (likes/dislikes). 

Failure behavior:

* If abstraction step outputs empty/invalid content and step does not allow empties, fail fast per existing repo behavior. 

---

## FR4 — Black-box judge scoring: numeric-only, strict JSON, no rationale

Add a judge step that takes:

* selected concepts
* raw user profile (full likes/dislikes)
* candidate idea cards JSON
* optional recent-history summary (if novelty enabled)

The judge outputs strict JSON only:

```json
{
  "scores": [
    {"id": "A", "score": 0},
    {"id": "B", "score": 78}
  ]
}
```

Constraints:

* `score` must be an int in `[0, 100]`.
* No additional keys, no explanations, no prose.

Judge call parameters:

* Temperature must default to 0.0 for stability.
* Use the same step recording behavior as all other steps (prompt/response stored). 

Failure behavior:

* If judge output is not parseable JSON or violates schema: raise `ValueError` (“invalid_judge_output”) and record `ctx.error.phase="prompt_pipeline"` plus step name/path if available. 

---

## FR5 — Selection algorithm in code (deterministic, diversity-aware)

After parsing scores:

* Select one candidate ID via a deterministic function using a seeded RNG:

  * With probability `1 - exploration_rate`, choose best score.
  * With probability `exploration_rate`, sample from the top quartile (or top K) weighted by score.

If novelty is enabled:

* Apply a lightweight penalty or filter using recent prompt history (see FR6) so that repeatedly overused motifs/colors are less likely to win.
* The novelty logic must not depend on LLM calls (keeps selection deterministic and offline-testable).

Selection metadata must be recorded in transcript under a new top-level field, e.g.:

```json
"blackbox_scoring": {
  "selected_id": "B",
  "selected_score": 81,
  "exploration_rate": 0.15,
  "exploration_roll": 0.07,
  "selection_mode": "explore",
  "score_table": [{"id":"A","score":74},{"id":"B","score":81}]
}
```

Failure behavior:

* If a score references an unknown id or an id is missing: raise.
* If all candidates invalid: raise.

---

## FR6 — Optional novelty input from recent generations

If `prompt.scoring.novelty.enabled: true` and `window > 0`:

* Load up to `window` most recent rows from the generations CSV (`prompt.generations_path`) and extract a simple “recent motifs” summary:

  * Example: top repeated tokens beyond stopwords, plus explicit detection of high-frequency terms like “sunset/sunrise,” etc.
  * Keep it deterministic (no LLM feature extraction in MVP).
* Provide this summary to the judge (as context) and/or apply as a code-level penalty in FR5.

Failure behavior:

* If history file missing, empty, or unreadable:

  * Warn loudly and continue with novelty disabled for this run (documented fallback), **but do not silently proceed**. 
* The warning must include the resolved path and exception string.

---

## FR7 — Downstream context isolation: judge outputs must not steer generation

Judge responses and scoring details must **not** be merged into the prompt conversation context used for final prompt creation.

Two acceptable implementations (pick one; do not do both):

1. **Add/Use `merge="none"` for Blocks**: run the entire `blackbox_scoring` block with merge mode that does not append any messages back into the parent message list.
2. **Run scoring in an isolated sub-run**: run scoring steps with a fresh message list and only pass structured outputs forward.

In either approach:

* Candidate generation may see upstream context (selected concepts and generator-safe profile summary), but judge/scoring text should not appear in messages for later steps.

Observability:

* Transcript still contains judge prompt/response (for audit), but downstream steps’ input messages do not include judge output.

---

## FR8 — Backward compatibility

When `prompt.scoring.enabled: false`:

* The pipeline should behave exactly as before:

  * same capture key for final `dalle_prompt`
  * same artifacts written (image, transcript, CSV rows, titles manifest). 

No existing schema changes are required for MVP; store scoring metadata in transcript only.

---

# Proposed conversation flow

Below is the intended **Step/Block flow** as it should appear in code (names are part of the contract; avoid lambdas that filter steps).

## When scoring is disabled (unchanged)

* `Block(name="pipeline", merge="all_messages", nodes=[ ...existing refined stages..., dalle_prompt_creation(capture_key="dalle_prompt"), ... ])` 

## When scoring is enabled (new)

**High-level structure**: run a scoring sub-block that does not merge into downstream context, then run final prompt creation.

### Block: `blackbox_scoring` (merge="none" or isolated)

* `ChatStep(name="profile_abstraction", ...)`

  * outputs `ctx.outputs["generator_profile_hints"]`
* `ChatStep(name="idea_cards_generate", ...)`

  * outputs `ctx.outputs["idea_cards_json"]`
* `ChatStep(name="idea_cards_judge_score", temperature=0.0, ...)`

  * outputs `ctx.outputs["idea_scores_json"]`

**Then in Python (between blocks)**:

* Parse idea cards + scores
* Select winner deterministically (FR5)
* Store:

  * `ctx.outputs["selected_idea_card"]`
  * `ctx.blackbox_scoring = {...}` (new transcript field)

### Block: `prompt_pipeline` (merge="all_messages")

* `refined_stage(name="final_prompt_from_selected_idea", draft_step=..., tot_enclave=...)`

  * Uses:

    * `selected_concepts`
    * raw profile
    * `ctx.outputs["selected_idea_card"]`
  * Produces a polished final image prompt
* `ChatStep(name="dalle_prompt_creation", capture_key="dalle_prompt", ...)` 

### Image generation + artifacts (unchanged)

* Generate image, save, titles manifest, generations CSV append, transcript write. 

---

# Implementation plan

## Files to change / add

1. **`run_config.py` (or wherever config validation lives)**

   * Add `prompt.scoring.*` keys and strict validation rules (FR1). 

2. **`prompts.py`**

   * Add prompt factories:

     * `profile_abstraction_prompt(profile) -> str`
     * `idea_cards_generate_prompt(concepts, generator_profile_hints, num_ideas) -> str`
     * `idea_cards_judge_prompt(concepts, raw_profile, idea_cards_json, recent_summary) -> str`
     * `final_prompt_from_selected_idea_prompt(concepts, raw_profile, selected_idea_card) -> str`

3. **New module `blackbox_scoring.py`**

   * Pure-python utilities:

     * JSON schema validation for idea cards and scores
     * Selection algorithm (epsilon-greedy)
     * Novelty extraction from generations CSV (optional)
     * Deterministic RNG plumbing (`random.Random(seed)`)

4. **`main.py`**

   * Wire in the scoring flow:

     * Build `blackbox_scoring` block and run it when enabled
     * Do python-side selection between blocks
     * Then run the existing refinement/final prompt stages
     * Keep a clean separation so sections like `blackbox_scoring` and `refinement` can be turned on/off

5. **`pipeline.py`** (only if needed)

   * If the existing merge system cannot isolate scoring output, implement `merge="none"` (Block-level) and test it.
   * Keep behavior unchanged for other merge modes. 

6. **`transcript.py`** (or transcript writer)

   * Add serialization for `ctx.blackbox_scoring` (new top-level field).
   * Ensure transcript still writes on failure as before. 

7. **`tests/`**

   * Add offline unit + integration tests (see below).

---

# Error handling + observability contract

## New failure points

1. **Idea cards JSON invalid**

   * Raise `ValueError("invalid_idea_cards_json: ...")`
   * `ctx.error.phase = "prompt_pipeline"`
   * `ctx.error.step = "<path/to/idea_cards_generate>"` if available
   * Transcript must still write on failure. 

2. **Judge output invalid**

   * Raise `ValueError("invalid_judge_output: ...")`
   * Include the first ~200 chars of the invalid output in the exception message for debugging (not the full thing).
   * Record same attribution fields as above. 

3. **Selection inconsistency**

   * Missing IDs, duplicate IDs, empty candidate set:
   * Raise with a specific message, include counts and offending ids.

4. **Novelty history load failure**

   * WARN loudly with path + exception and continue with novelty disabled (documented fallback).
   * This is the only “warn+fallback” allowed here.

## Logs

At INFO:

* “Blackbox scoring enabled: num_ideas=…, exploration_rate=…, novelty=…”
* “Selected candidate: id=…, score=…, selection_mode=exploit|explore”

At WARN (only novelty fallback):

* “Novelty enabled but history unavailable; disabling novelty for this run: <path> <exc>”

---

# Data/artifact changes

## Transcript JSON (additive)

Add a new top-level field `blackbox_scoring` (or `scoring`) containing:

* config snapshot (or key subset)
* score table
* selected id + selected score
* exploration roll and mode
* novelty summary used (if any)

Backwards compatibility:

* Additive only; existing consumers can ignore unknown fields. 

## Generations CSV

MVP: **no schema change**.
Optional later: append columns like `scoring_enabled`, `selected_candidate_id`, `selected_candidate_score`.

---

# Testing requirements (pytest, offline)

## Unit tests

1. **Idea card schema parsing**

   * Valid JSON parses into normalized internal structure.
   * Missing fields / duplicate IDs raise.

2. **Judge score parsing**

   * Accepts valid score JSON.
   * Rejects non-int scores, out-of-range, missing ids.

3. **Selection determinism**

   * With fixed seed and fixed scores:

     * selection is stable across runs
     * exploration branch is exercised by forcing `exploration_rate=1.0` in test

4. **Novelty extraction**

   * Given a temp generations CSV with repeated motifs, returns deterministic summary.

## Integration test

Run `run_generation()` with:

* Fake TextAI backend that returns:

  * profile abstraction JSON/text
  * idea cards JSON
  * judge score JSON
  * final prompt text
* Fake ImageAI backend that “writes” a dummy image file.

Assert:

* image exists
* transcript exists and includes:

  * the idea generation step
  * the judge scoring step
  * `blackbox_scoring.selected_id`
* **Downstream isolation**:

  * validate that the final prompt creation step input messages do not include judge output text (inspect recorded “prompt” field for that step or the fake backend’s captured inputs).

Do not test by matching entire prompt wording; validate structure and presence of expected artifacts. 

---

# Documentation updates

* Update README “Future Directions” / pipeline section to mention:

  * how to enable `prompt.scoring.enabled`
  * what gets recorded in transcript
  * the “no feedback / black-box numeric scoring” principle 

Optionally add `docs/scoring.md` describing:

* selection algorithm
* novelty behavior
* troubleshooting invalid JSON issues

---

# Acceptance criteria

1. With `prompt.scoring.enabled: true`, a run produces:

   * transcript JSON with recorded steps and a `blackbox_scoring` object
   * a final image and the normal CSV/manifest artifacts 

2. Judge responses are **not merged** into downstream conversation context:

   * verified by an integration test that inspects the final prompt step’s model input.

3. With scoring disabled, behavior is unchanged:

   * same artifacts, same capture key for `dalle_prompt`. 

4. All tests pass offline with fake backends.

---

# Pitfalls to avoid

* Treating config strings like `"false"` as truthy (`bool("false") == True`). Use strict parsing. 
* Letting judge output leak into later prompt context (will cause over-steering and rubric gaming).
* Depending on step indices in tests; use step names/paths and structural assertions.
* Adding selection logic inside an LLM step (breaks determinism and “black-box” intent).
* Silent fallback when JSON is invalid; fail fast with clear attribution. 
