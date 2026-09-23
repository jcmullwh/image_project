# PROJECT_SPEC — Finish delineation: move prompt policy into `prompts/`, make stages macro-only, enforce boundaries, and fix StageIO/observability gaps

## Task

Complete the latest refactor iteration by addressing the remaining “blur” and semantic inconsistencies:

1. **Move all prompt policy out of `framework/` and `impl/current/` into a dedicated `image_project/prompts/` package** (pure prompt/parse helpers).
2. **Remove `image_project.impl.*` imports from `image_project.stages/*`** (stages depend on `framework` + `foundation` + `prompts`, not `impl`).
3. **Enforce the “stage is a macro that builds a stage block” model** by removing “inline `Block` stage nodes” from plan authoring: plans must return **StageInstances only**.
4. **Fix plan/plugin code paths that bypass StageRef semantics** (e.g., block-returning variations and blackbox isolated idea-card block generators).
5. **Make stage IO/observability trustworthy** for key composite stages by:

   * requiring StageIO declarations where meaningful, and
   * recording **discovered capture keys per stage instance** in the pipeline metadata (so composite/dynamic captures are visible and validated).

This work should reduce cognitive load without adding new abstraction layers beyond what already exists (`StageRef → Block`).

---

## Goal

### Developer-visible outcomes

* Clear, enforceable dependency directions:

  * `foundation` → imports nothing higher-level
  * `framework` → imports `foundation` (+ `prompts` allowed if needed), but not `stages`/`impl`
  * `stages` → imports `framework`/`foundation`/`prompts`, but not `impl`
  * `impl` → contains experiments/plans/plugins only; not a “prompt policy dumping ground”
* IDE navigation improves further:

  * Stage modules import prompt builders from `image_project.prompts.*`, not `impl/current/prompting.py`.
* Config semantics become uniform:

  * `prompt.stage_configs.defaults/instances` apply consistently to all stage nodes (no “works unless your plan returns raw Blocks”).

### Run-visible outcomes

* Transcript remains the canonical audit trail, and pipeline metadata becomes more helpful:

  * `outputs.prompt_pipeline.stage_io_effective[<stage_instance>]` includes discovered capture keys for that stage instance.
  * Stage kind ↔ instance mapping remains explicit.
* No prompt policy resides inside “structural” framework code.

---

## Context (what exists now and why it’s insufficient)

The current snapshot has made real progress (StageRefs, stage configs, patterns), but several issues remain:

* **Prompt policy still lives in framework**:

  * `framework/inputs.py:make_dislike_rewrite_filter()` embeds LLM prompt text.
  * `framework/artifacts.py:generate_title()` embeds system/user prompt policy.
* **Stages import `impl/current`** (primarily `impl/current/prompting.py` and `impl/current/blackbox_refine_prompts.py`), which undermines the intended layering and keeps the “giant prompts file” as the de facto prompt source of truth.
* **Plans can still return raw Blocks** because `StageNode = StageInstance | Block`, which:

  * bypasses StageRef build/config/IO semantics,
  * allows reusing existing stage IDs for different behavior (e.g., `standard.initial_prompt` but different prompt), which breaks the “stage id means something stable” expectation.
* **StageIO is present but not consistently useful**: key composite stages (`refine.tot_enclave`, `blackbox_refine.iter`) currently declare no IO, and actual capture keys for dynamic/composite stages are not surfaced in pipeline metadata.

---

## Constraints

* **No silent fallbacks**: if a plan tries to bypass StageRef semantics, it must fail clearly.
* **Offline tests only**.
* **Determinism** remains unchanged: any randomness must remain seeded and recorded.
* **Transcript is canonical**: do not reduce transcript fidelity.
* **Python version constraint**: repo currently pins `requires-python ==3.13.*` in `pyproject.toml`; changes must remain compatible with that unless explicitly changed (this spec does not change the Python version target).

---

## Non-goals

* Not redesigning experimentation runners or CLI flows.
* Not implementing post-image loops (OCR, vision scoring) in this step.
* Not rewriting all stage prompt content (only relocating and re-homing it).
* Not adding a new “meta pipeline” abstraction.

---

## Functional requirements

### FR1 — Introduce a dedicated prompt-policy package

1. Add `image_project/prompts/` as the canonical home for prompt policy and parsing helpers.
2. `image_project/prompts/*` modules must be **pure**:

   * no imports from `image_project.stages`, `image_project.app`, or `image_project.impl`
   * minimal dependencies (stdlib + small shared helpers ok)
3. Move prompt policy out of:

   * `image_project/impl/current/prompting.py`
   * `image_project/impl/current/blackbox_refine_prompts.py`
   * `image_project/framework/inputs.py` (prompt strings)
   * `image_project/framework/artifacts.py` (title prompt strings)
4. Organize prompts by domain (example layout; adjust as needed):

   * `prompts/preprompt.py`
   * `prompts/standard.py`
   * `prompts/postprompt.py`
   * `prompts/blackbox.py`
   * `prompts/blackbox_refine.py`
   * `prompts/titles.py`
   * `prompts/concept_filters.py`

### FR2 — Framework contains no embedded prompt policy

1. Remove embedded prompt templates from:

   * `framework/inputs.py:make_dislike_rewrite_filter()`
   * `framework/artifacts.py:generate_title()`
2. These functions may import prompt builders from `image_project.prompts.*`, but must not embed multi-line prompt policy in the framework module itself.
3. No other prompt-policy strings may be added to `framework/` going forward.

### FR3 — Stages do not import `impl`

1. Remove all `image_project.impl.*` imports from `image_project/stages/**`.
2. Stages must import prompt builders from `image_project.prompts.*` (or group-local stage helper modules), not from `impl/current/prompting.py` or `impl/current/blackbox_refine_prompts.py`.
3. Add an import-boundary test enforcing:

   * `stages/` must not import `image_project.impl.*`.

### FR4 — Plans must author **StageInstances only**

1. Remove the “inline stage block” escape hatch:

   * Change `StageNode` from `StageInstance | Block` to **`StageInstance` only**.
   * Update `SequencePromptPlan.stage_nodes(...)` and all plan implementations to return `list[StageInstance]`.
2. `compile_stage_nodes(...)` must no longer accept raw `Block` nodes.
3. Failure mode:

   * if a plan returns a `Block` (or any non-StageInstance), raise a clear `TypeError` pointing to the offending plan.

### FR5 — Fix block-returning plan/plugin helpers to preserve StageRef semantics

#### FR5.1 `SimpleNoConceptsPromptPlan` must not reuse `standard.initial_prompt` with different behavior

* Current behavior: returns a block named `standard.initial_prompt` with a freeform prompt, bypassing `STANDARD_INITIAL_PROMPT` StageRef.
* Required change:

  1. Create a dedicated stage kind (new StageRef) for this behavior, e.g.:

     * `standard.initial_prompt_freeform`
  2. Plan uses that stage ref via `.instance()`.

This preserves the invariant: **stage IDs map to stable, single meanings**.

#### FR5.2 `build_blackbox_isolated_idea_card_specs` must not return raw Blocks

* Current behavior: returns `list[Block]` with per-idea stage IDs like `blackbox.idea_card_generate.<idea_id>` and step capture keys like `blackbox.idea_card.<idea_id>.json`.
* Required change:

  1. Replace it with `build_blackbox_isolated_idea_card_instances(...) -> list[StageInstance]`.
  2. Implement a stage kind for isolated idea-card generation, e.g.:

     * kind id: `blackbox.idea_card_generate`
     * instance ids: `blackbox.idea_card_generate.<idea_id>`
  3. Stage builder derives the idea id from the instance id (stable parse rule), and keeps the same capture keys as today to avoid breaking downstream logic.
  4. This stage kind may remain “internal” (not necessarily documented for end users), but it must be a StageRef-based stage node to preserve consistent config/metadata behavior.

### FR6 — Stage IO/observability must be trustworthy for key composite stages

1. Ensure key composite stages declare meaningful `io=` where applicable (even if minimal):

   * `stages/refine/tot_enclave.py`
   * `stages/blackbox_refine/loop.py` (blackbox refine iteration stage)
2. Add **run-time effective IO metadata** to pipeline metadata:

   * record discovered capture keys *per stage instance*, not just per kind.
3. Add to `outputs.prompt_pipeline`:

   * `stage_io_effective[<stage_instance_id>] = { "requires": [...], "provides_declared": [...], "captures_declared": [...], "captures_discovered": [...] }`
4. Collision behavior:

   * if discovered capture keys collide across stages, fail at compile time (unless an explicit allowlist exists; default is fail).

### FR7 — Fix misleading StageRef metadata

1. Update `stages/blackbox_refine/seed_from_draft.py`:

   * `source=` must reference the stage module/builder function (not a plan plugin class).

### FR8 — Documentation update: layering rules reflect reality

1. Update `docs/where_things_live.md` to include `image_project/prompts/` and explicitly allow:

   * `framework` importing `prompts` (for helper LLM calls like titles), while disallowing embedded prompt policy in framework.
2. Document the “stage macro” rule:

   * plans return StageInstances only; no inline blocks.

---

## Proposed conversation flow (what runs)

This is unchanged in shape, but becomes easier to read because stages import prompts from `prompts/*` and plans return StageInstances only.

### Example: ToT refinement stage (explicit stage, generic pattern)

```python
Block(name="refine.tot_enclave_01", merge="last_response", meta={"stage_kind":"refine.tot_enclave"}, nodes=[
  Block(name="tot_enclave", merge="all_messages", nodes=[
    ChatStep(name="critic.hemingway", merge="none", capture_key="refine.tot_enclave_01.critic.hemingway", prompt=...),
    ChatStep(name="critic.octavia", merge="none", capture_key="refine.tot_enclave_01.critic.octavia", prompt=...),
    ChatStep(name="final_consensus", merge="last_response", prompt=...),
  ])
])
```

### Example: blackbox isolated idea cards (StageInstances only; no plan-time blocks)

Plan returns instances:

* `blackbox.idea_card_generate.idea_01`
* `blackbox.idea_card_generate.idea_02`
* …

Each stage instance compiles into a stage block that captures:

* `blackbox.idea_card.idea_01.json`
* `blackbox.idea_card.idea_02.json`
* …

---

## Implementation plan

### 1) Create `image_project/prompts/` and move prompt policy there

Files to add (example):

* `image_project/prompts/__init__.py`
* `image_project/prompts/preprompt.py`
* `image_project/prompts/standard.py`
* `image_project/prompts/postprompt.py`
* `image_project/prompts/blackbox.py`
* `image_project/prompts/blackbox_refine.py`
* `image_project/prompts/titles.py`
* `image_project/prompts/concept_filters.py`

Move/translate functions/constants:

* from `impl/current/prompting.py`: split into the above modules (prompt builders + parsing helpers).
* from `impl/current/blackbox_refine_prompts.py`: move into `prompts/blackbox_refine.py`.
* keep `impl/current/prompting.py` as a **compat shim** that re-exports from `image_project.prompts.*` (optional but recommended), with a loud comment and/or DeprecationWarning in docstring. Ensure **no stage imports it**.

### 2) Remove prompt templates from framework (use prompt builders from `prompts/*`)

* Update `framework/inputs.py`:

  * extract the multi-line prompt into `prompts/concept_filters.py` (return either text content or full messages).
  * keep the LLM call in framework if that’s where the ConceptFilter interface lives.
* Update `framework/artifacts/manifest.py`:

  * move the title-generation prompt policy to `prompts/titles.py`.
  * `generate_title()` constructs messages using `prompts.titles.build_title_messages(...)`.

### 3) Update all stages to import from `image_project.prompts.*`

* For every file currently importing `image_project.impl.current.prompting` or `image_project.impl.current.blackbox_refine_prompts`:

  * replace with `image_project.prompts.*` imports.
* Update `StageRef.source` strings to reflect new module paths.

### 4) Remove `Block` from `StageNode` and enforce StageInstance-only plans

* Update `framework/prompt_pipeline/__init__.py`:

  * `StageNode: TypeAlias = StageInstance`
  * remove the `isinstance(node, Block)` branch in `compile_stage_nodes(...)`.
  * tighten types and error messages around plan output.
* Update plan interface:

  * `SequencePromptPlan.stage_nodes(...) -> list[StageInstance]`
* Update all plans/plugins accordingly.

### 5) Fix block-returning plans/plugins

#### 5.1 `SimpleNoConceptsPromptPlan`

* Add a new stage module:

  * `stages/standard/initial_prompt_freeform.py` exporting `STAGE` with id `standard.initial_prompt_freeform`.
* Update plan to use `STANDARD_INITIAL_PROMPT_FREEFORM.instance()` rather than `make_chat_stage_block(...)`.

#### 5.2 Blackbox isolated idea cards

* Replace `impl/current/blackbox_idea_cards.py:build_blackbox_isolated_idea_card_specs(...) -> list[Block]` with:

  * `build_blackbox_isolated_idea_card_instances(...) -> list[StageInstance]`
* Add stage kind:

  * `stages/blackbox/idea_card_generate.py` with id `blackbox.idea_card_generate`.
* Builder derives `idea_id` from instance id suffix (`blackbox.idea_card_generate.<idea_id>`), preserves existing capture keys.

### 6) Improve Stage IO observability in compiled metadata

* Update `framework/prompt_pipeline/__init__.py:compile_stage_nodes(...)` to record per-instance:

  * declared requires/provides/captures from StageRef.io
  * discovered capture keys from `_collect_capture_keys(...)`
* Attach to `ctx.outputs["prompt_pipeline"]` (likely via compiled metadata).

### 7) Enforce boundaries with tests

* Extend `tests/test_import_boundaries.py`:

  * add `test_stages_source_does_not_import_impl()`.
  * (optional) add `test_framework_contains_no_prompt_templates()` as a simple substring heuristic guarding against reintroducing prompt templates in framework.

### 8) Fix misleading `source=` metadata

* Update `stages/blackbox_refine/seed_from_draft.py` `source=` to point to actual builder function/module.

### 9) Update docs

* Update `docs/where_things_live.md` to include `prompts/` and stage macro rule.
* Regenerate `docs/stages.md` if stage `source=` paths changed.

---

## Error handling + observability contract

### New failure points

* If a plan returns a non-StageInstance stage node:

  * raise `TypeError` with plan name and offending node type/value.
* If a stage instance id is malformed for derived-parameter stages (e.g. `blackbox.idea_card_generate` missing idea id suffix):

  * raise `ValueError` with the required instance id format.
* Capture key collisions discovered across stages:

  * raise `ValueError` listing colliding keys and stage instance IDs.

### Observability additions

* `outputs.prompt_pipeline.stage_io_effective` includes discovered capture keys per stage instance.
* For derived-parameter stages (idea cards), record derived parameters (idea_id) in `stage_configs_effective` or a small `stage_params_effective` structure, so transcript debugging doesn’t require reading parsing logic.

---

## Data/artifact changes

* Transcript JSON adds:

  * `outputs.prompt_pipeline.stage_io_effective` (new field).
* No CSV schema changes.
* Preserve existing capture key names for blackbox idea cards to avoid breaking downstream consumers.

---

## Testing requirements (pytest, offline)

1. **Boundary test**: stages do not import impl.
2. **Plan authoring test**: all registered plans return StageInstances only.
3. **Blackbox isolated idea cards integration**:

   * run plan that uses isolated idea cards and verify:

     * expected stage instance paths exist
     * expected capture keys exist (`blackbox.idea_card.<idea_id>.json`)
4. **Framework prompt policy removal**:

   * test that framework no longer contains embedded prompt templates (heuristic or direct content checks).
5. **Metadata correctness**:

   * compiled pipeline metadata includes `stage_io_effective` with discovered capture keys for `refine.tot_enclave_*` and for a blackbox iteration stage.

---

## Acceptance criteria

1. No files under `image_project/framework/` contain embedded multi-line prompt policy; all such content is imported from `image_project.prompts.*`.
2. No files under `image_project/stages/` import `image_project.impl.*` (enforced by tests).
3. `StageNode` supports StageInstances only; all plans/plugins comply.
4. `SimpleNoConceptsPromptPlan` no longer reuses `standard.initial_prompt` for different behavior; it uses a distinct stage kind.
5. Blackbox isolated idea card generation no longer returns raw Blocks from plan helpers; it uses StageInstances and a stage kind.
6. `outputs.prompt_pipeline.stage_io_effective` is present and includes discovered capture keys per stage instance.
7. `stages/blackbox_refine/seed_from_draft.py` has correct `source=` metadata.
8. All tests pass offline.

---

## Pitfalls to avoid

* Reintroducing “stringly stage composition” by encoding more logic into instance id parsing than necessary (use parsing only when unavoidable; keep formats explicit and tested).
* Allowing “inline block stages” to creep back in as an “easy experiment shortcut.”
* Moving too much non-prompt logic into `prompts/` (keep it prompt/parse only; stage logic stays in stages).
* Silent acceptance of unused stage config keys (must remain strict).
