# Extract a reusable `pipelinekit` kernel (stage system + compiler + config namespaces) and keep image-project conventions as injected policy

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `.todos/PLANS.md` (ExecPlan requirements) and `agents.md` (repo invariants: no silent fallbacks/defaults, heavy logging, offline tests, strict docstrings).

## Purpose / Big Picture

The current repo has a “stage = macro that returns a Block” authoring model that works well for prompt pipelines, but several pieces that *could* be generic infrastructure are still coupled to image-project conventions (e.g., suffix selector resolution, “primary draft step” targeting for overrides, and “capture stage defaults to last chat stage”). This makes it hard to reuse the stage system and compiler in other experiments without inheriting image-specific assumptions.

After this change, the reusable parts (stage reference/registry, strict stage-owned config namespaces, stage compilation + patching) live in a new top-level package (`pipelinekit`) that has **no dependency on `image_project.*`**. Image-project-specific conventions (selector syntax, override targeting, capture semantics) become explicit, injected policy objects implemented inside `image_project/`. The user-visible behavior of `image_project` should remain the same: running generation produces the same artifacts (prompt pipeline metadata, transcript structure, outputs like `image_prompt`) with no new silent defaults.

## Progress

- [x] (2026-01-18 19:27Z) Drafted initial ExecPlan from the concrete implementation notes.
- [x] (2026-01-18 19:27Z) Moved ExecPlan to `.plans/3 - in_progress/` and began implementation.
- [x] (2026-01-18 20:34Z) Inventoried existing generic-vs-project-specific boundaries in code and docs; recorded findings in this plan.
- [x] (2026-01-18 20:20Z) Implemented `pipelinekit` package (engine + authoring kit) and re-exported via existing `image_project.*` modules.
- [x] (2026-01-18 20:29Z) Refactored stage compilation to be policy-driven; implemented image-project policies to preserve behavior.
- [x] (2026-01-18 20:34Z) Fixed strict unknown-key handling so `prompt.stage_configs` is accepted in strict mode; added a regression test.
- [x] (2026-01-18 20:32Z) Added `pipelinekit` import-boundary tests (must not import `image_project.*`).
- [x] (2026-01-18 20:32Z) Updated docs to include the new `pipelinekit` kernel layer and clarified dependency direction.
- [x] (2026-01-18 20:34Z) Ran full offline test suite; `python -m pytest -q` is green.

## Surprises & Discoveries

- Observation: The "generic" stage compilation logic currently lives in `image_project/framework/prompt_pipeline/__init__.py` (`compile_stage_nodes()` + `resolve_stage_blocks()`), but it hardcodes image-project conventions (suffix stage-id normalization, override targets `draft` step with `meta.role == primary`, default capture stage inferred as "last chat-producing stage").
  Evidence: `image_project/framework/prompt_pipeline/__init__.py` functions `normalize_stage_id()` (local), `_find_primary_draft_steps()`, `_override_primary_draft_step()`, `_block_produces_assistant_output()`.

- Observation: The pipeline engine (steps/blocks/runner/step recording) is already relatively generic and isolated in `image_project/foundation/pipeline.py`, and the repo already enforces layer boundaries via `tests/test_import_boundaries.py`.
  Evidence: `tests/test_import_boundaries.py` forbids `image_project.foundation` importing `image_project.framework|impl|stages` and forbids `image_project.framework` importing `image_project.impl|stages`.

- Observation: `RunConfig.from_dict()` has an unknown-key schema (`prompt_schema`) that exempts `prompt.stages.overrides` and `prompt.extensions.*`, but (as of drafting) `prompt.stage_configs` parsing is validated later and may still be reported as “unknown” by the early schema pass in strict mode unless explicitly exempted/added.
  Evidence: `image_project/framework/config.py` has `prompt_schema` + later parsing of `prompt.stage_configs` under `raw_stage_configs = prompt_cfg.get("stage_configs")`.

- Observation: Moving the engine/stage kit into a separate top-level package (`pipelinekit`) required compatibility wrappers in `image_project/foundation/*` and `image_project/framework/*` to keep existing imports stable.
  Evidence: `image_project/foundation/pipeline.py` and `image_project/foundation/messages.py` now re-export from `pipelinekit/engine/*`; `image_project/framework/stage_types.py` and `image_project/framework/stage_registry.py` now re-export from `pipelinekit/*`.

- Observation: Strict unknown-key mode (`strict: true`) would have rejected `prompt.stage_configs` before this change because `prompt_schema` did not include it, even though `prompt.stage_configs` is validated later.
  Evidence: `image_project/framework/config.py` now includes `prompt_schema["stage_configs"] = ANY`; regression test in `tests/test_run_config_unknown_keys.py`.

## Decision Log

- Decision: Create a new top-level Python package named `pipelinekit` (not under `image_project/`) and enforce a strict one-way dependency: `image_project` may import `pipelinekit`, but `pipelinekit` must not import `image_project.*`.
  Rationale: This is the minimal packaging change that makes the “generic parts truly generic” and testable as such, while keeping the existing stage macro execution model intact.
  Date/Author: 2026-01-18 (Codex)

- Decision: Keep behavior compatibility by re-exporting legacy import paths (temporarily) from `image_project/foundation/*` and `image_project/framework/*` to the new `pipelinekit` symbols.
  Rationale: This reduces churn across the many stage modules and tests, and enables incremental migration with stable behavior checkpoints.
  Date/Author: 2026-01-18 (Codex)

- Decision: Implement stage compilation as a policy-driven compiler in `pipelinekit/compiler.py`, with image-project conventions supplied by `image_project/framework/prompt_pipeline/stage_policies.py`.
  Rationale: This is the dependency-direction “kernel + injected policy” refactor target; it removes image-project conventions from the generic compiler while preserving existing behavior via explicit policies.
  Date/Author: 2026-01-18 (Codex)

- Decision: Keep `StageRegistry.resolve()` suffix lookup behavior intact (it is now in `pipelinekit/stage_registry.py`) to preserve the current user-facing stage-id ergonomics.
  Rationale: Existing behavior and tests rely on suffix lookup (e.g. `registry.resolve("initial_prompt")`); stage *instance* selector resolution in compilation is now policy-driven, which is the primary coupling we needed to remove.
  Date/Author: 2026-01-18 (Codex)

- Decision: Fail loudly if stage config merging produces a non-mapping payload during compilation.
  Rationale: This removes a silent “defensive” fallback and aligns with the repo invariant “no silent fallbacks”; mis-shaped config should fail at compilation with clear attribution.
  Date/Author: 2026-01-18 (Codex)

- Decision: Leave the legacy compilation implementation in `image_project/framework/prompt_pipeline/__init__.py` temporarily but make it unreachable via an early return to the `pipelinekit` compiler.
  Rationale: This reduces risk during a large refactor by keeping a local reference implementation available for comparison, while ensuring production/test behavior is driven by the new kernel compiler.
  Date/Author: 2026-01-18 (Codex)

## Outcomes & Retrospective

Outcomes:

- Added a new generic kernel package `pipelinekit/` containing:
  - engine primitives (`pipelinekit/engine/pipeline.py`, `pipelinekit/engine/messages.py`)
  - stage authoring kit (`pipelinekit/stage_types.py`, `pipelinekit/stage_registry.py`, `pipelinekit/config_namespace.py`)
  - a policy-driven stage compiler (`pipelinekit/compiler.py`)
- Preserved existing `image_project.*` import paths via compatibility wrappers (re-exports) in `image_project/foundation/*` and `image_project/framework/*`.
- Implemented image-project conventions as explicit policies in `image_project/framework/prompt_pipeline/stage_policies.py` and wired them into `image_project/framework/prompt_pipeline/__init__.py`.
- Fixed strict unknown-key validation so `prompt.stage_configs` is accepted in strict mode and still validated explicitly.
- Updated `docs/where_things_live.md` to document `pipelinekit` as the lowest layer.
- Added boundary tests for `pipelinekit` import independence and verified the full offline test suite passes.

Gaps / follow-ups:

- Remove (or delete) the now-unreachable legacy compilation code in `image_project/framework/prompt_pipeline/__init__.py` once the migration is considered stable.

## Context and Orientation

### Current architecture (what exists today)

The prompt pipeline is executed as a tree of nodes:

- `ChatStep`: one LLM call.
- `ActionStep`: pure-Python glue.
- `Block`: a named container of nodes with merge semantics.
- `ChatRunner`: executes a node tree, enforces deterministic path naming, and records per-step records into `ctx.steps`.

Key files:

- Pipeline engine (canonical): `pipelinekit/engine/pipeline.py`, `pipelinekit/engine/messages.py`.
- Pipeline engine (compat re-exports): `image_project/foundation/pipeline.py`, `image_project/foundation/messages.py`.
- Run context: `image_project/framework/runtime.py` (`RunContext`).
- Stage system (canonical): `pipelinekit/stage_types.py` (`StageRef`, `StageInstance`, `StageIO`), `pipelinekit/stage_registry.py` (`StageRegistry`), `pipelinekit/config_namespace.py` (`ConfigNamespace`).
- Stage system (compat re-exports): `image_project/framework/stage_types.py`, `image_project/framework/stage_registry.py`, `image_project/framework/config_namespace.py`.
- Stage compilation + patching (canonical): `pipelinekit/compiler.py` (policy-driven generic), `image_project/framework/prompt_pipeline/stage_policies.py` (image-project conventions).
- Stage compilation + patching (compat wrappers): `image_project/framework/prompt_pipeline/__init__.py` (`compile_stage_nodes()`, `resolve_stage_blocks()`).
- App entrypoint that uses these: `image_project/app/generate.py` (`run_generation()`), which calls `compile_stage_nodes()` then `resolve_stage_blocks()` and runs the resulting pipeline.

The repo already documents intended layering in `docs/where_things_live.md` and pipeline semantics in `docs/pipeline.md`.

### What “generic” means for this refactor (explicit invariants)

This refactor introduces two explicit, enforced invariant sets:

Generic invariants (the new `pipelinekit` kernel):

1) The kernel must not import `image_project.*` (enforced by tests).
2) The kernel contains the engine primitives (Block/ChatStep/ActionStep + runner + recording), the stage authoring kit (StageRef/StageRegistry/ConfigNamespace), and the policy-driven stage compiler.
3) The kernel contains **no image-project-specific conventions** such as stage instance suffix selector rules, “primary draft step” targeting, or image-prompt capture semantics; those are injected via explicit policies.
3) Anything that would otherwise be a convention is expressed as an injected policy interface (selector resolution, override targeting, capture selection/attachment).
4) There are no silent fallbacks and no silent defaults: if a choice matters, it must be explicit in configuration/code and logged.

Project invariants (the existing `image_project` app):

1) Image-project owns conventions: selector syntax (suffix resolution rules), what “primary step” means (if kept), capture-key naming for the final prompt, and what “default capture stage” means.
2) Image-project owns stage libraries (`image_project/stages/*`) and plan authoring (`image_project/impl/*`).
3) Image-project remains responsible for artifact schemas (transcript JSON fields, CSV schema, manifests).

Deliverable for this phase: a short doc in `pipelinekit/docs.py` (or `pipelinekit/README.md`) that states these invariants, plus an automated import-boundary test that enforces “`pipelinekit` does not import `image_project.*`”.

## Plan of Work

Implement this as an incremental packaging + dependency-direction refactor, with tests passing at each milestone. The key risk is accidental behavior drift in stage override targeting and capture selection; mitigate by keeping the existing behavior implemented in `image_project` policies and writing behavior-equivalence tests before removing old code paths.

### Milestone 0: Define boundaries and tests first (no code motion yet)

Write down the “generic vs project” invariants (above) in a new `pipelinekit` doc module and add boundary tests that will fail if `pipelinekit` depends on `image_project`. This milestone exists to prevent the refactor from accidentally reintroducing coupling as the code moves.

In this milestone, also inventory which parts of `image_project/framework/prompt_pipeline/__init__.py` are truly generic and which are conventions, and list them explicitly in this plan (update `Surprises & Discoveries` with concrete references).

### Milestone 1: Create `pipelinekit/` and move (or copy + re-export) the kernel primitives

Create a top-level package `pipelinekit/` with two conceptual areas:

1) Pipeline engine (generic primitives): `MessageHandler`, `ChatStep`, `ActionStep`, `Block`, `ChatRunner`, `StepRecorder` (+ default recorder).
2) Pipeline authoring kit (generic): `ConfigNamespace`, `StageIO`, `StageRef`, `StageInstance`, `StageRegistry`.

Concrete file mapping (initial proposal; adjust as needed but keep imports stable):

- `pipelinekit/engine/messages.py` from `image_project/foundation/messages.py`
- `pipelinekit/engine/pipeline.py` from `image_project/foundation/pipeline.py`
- `pipelinekit/config_namespace.py` from `image_project/framework/config_namespace.py`
- `pipelinekit/stage_types.py` from `image_project/framework/stage_types.py` (updated to import `pipelinekit.engine.pipeline.Block` and `pipelinekit.config_namespace.ConfigNamespace`)
- `pipelinekit/stage_registry.py` from `image_project/framework/stage_registry.py` (updated so resolution behavior is policy-owned; see Milestone 2)

Maintain compatibility by turning the existing `image_project/*` modules into thin re-exports:

- `image_project/foundation/pipeline.py` re-exports from `pipelinekit.engine.pipeline`
- `image_project/foundation/messages.py` re-exports from `pipelinekit.engine.messages`
- `image_project/framework/config_namespace.py`, `image_project/framework/stage_types.py`, `image_project/framework/stage_registry.py` re-export from `pipelinekit`

This milestone should not change runtime behavior.

### Milestone 2: Make stage compilation + patching fully policy-driven

The stage compiler becomes generic when it stops inferring or hardcoding conventions. The current “conventions to extract into policy” are:

- selector resolution rules (suffix matching, ambiguity errors, etc.)
- override targeting rules (what step(s) inside a stage block are patchable)
- capture selection rules (how to choose a default capture stage, and what it means to “attach capture”)

Implement three small policy interfaces in `pipelinekit`:

1) `SelectorResolver`: resolves include/exclude selectors and override selectors to concrete stage instance ids (and, if needed, resolves kind selectors to stage kind ids).
2) `OverrideTargetPolicy`: given a stage block and an override payload, returns the concrete nodes/paths to patch; the compiler applies patches only to those targets.
3) `CapturePolicy`: selects/validates a capture stage and attaches capture behavior (e.g., setting a `capture_key` or installing a capture step).

Then refactor `compile_stage_nodes()` and `resolve_stage_blocks()` into a single generic compilation entrypoint inside `pipelinekit` (either a function or a `Compiler` class) that:

- validates stage-node uniqueness and IO (`StageIO.requires/provides/captures`)
- builds stage blocks using `StageRef.build(...)` with per-stage `ConfigNamespace` that enforces consumed keys
- discovers capture keys in blocks and errors on collisions
- normalizes and applies overrides via `OverrideTargetPolicy`
- chooses capture stage and attaches capture via `CapturePolicy`
- returns both the compiled `Block` tree and a metadata structure suitable for `ctx.outputs["prompt_pipeline"]`

Preserve existing image-project behavior by implementing policies in `image_project/`:

- `image_project/framework/prompt_pipeline/stage_policies.py` (or similar) implements:
  - suffix selector resolution identical to today’s `normalize_stage_id()` behavior
  - override targeting identical to `_find_primary_draft_steps()` (“`ChatStep.name == 'draft'` and `meta.role == 'primary'`”)
  - default capture stage selection identical to `resolve_stage_blocks()` (“last stage block that produces assistant output”) and capture attachment identical to `_with_capture_key(..., capture_key='image_prompt')`

Update `image_project/app/generate.py` to compile via the new pipelinekit compiler while keeping the existing “stage = macro” execution model.

### Milestone 3 (optional, explicitly gated): Plugin discovery without “import everything to register”

This milestone is optional and should only be implemented if there is a clear near-term need to distribute stages/plans outside this repo. If implemented, it must not introduce silent fallback behavior.

Define “entry points” in plain language here: an entry point is a packaging metadata hook that allows discovering callables from installed Python distributions without importing arbitrary modules by scanning a package directory.

Proposed approach:

- Add entry-point discovery support in `pipelinekit` using `importlib.metadata.entry_points()`, with explicit configuration selecting the discovery mode.
- Keep `image_project/stages/registry.py` as an explicit “static registrar” path for in-repo development.
- Require explicit selection (config or code) of which discovery mechanism to use; do not silently fall back from entry points to static imports.

### Milestone 4: Config layering with strictness preserved (fix the “stage_configs + strict unknown keys” class of bugs)

Separate config into two strata:

1) Core orchestration config: stable keys for paths, backend selection, seed policy, plan selection, and stage wiring.
2) Extension config: run-to-run experiment knobs that should not require central schema changes when adding a new stage or plugin.

Concrete changes inside `image_project/framework/config.py`:

- Ensure `prompt.stage_configs` is recognized by the unknown-key schema pass so strict mode does not reject it.
- Keep the *container validation* centralized (types/shapes of `defaults`/`instances`), but do not enumerate per-stage keys centrally.
- Ensure `prompt.extensions` (already exempted by tests) remains the supported place for plugin config blocks, and carry it through into runtime so plugins can consume it explicitly via `ConfigNamespace` (or a separate plugin namespace helper).

### Milestone 5 (optional): Apply the authoring kit to the “app pipeline” beyond prompts

This milestone is explicitly out-of-scope unless there is a concrete need to make the media/artifact phases stage-driven. If pursued later, treat it as a follow-on ExecPlan: “Define run pipeline as stages across prompt/media/artifacts”, with explicit decisions about output namespaces and conversation scopes.

### Milestone 6: Tests, migration guardrails, and documentation

Update tests to enforce the new boundaries and to prove behavior equivalence:

- Import boundary tests:
  - `pipelinekit` must not import `image_project.*`.
  - Pipeline engine module in `pipelinekit` must not import project code.
  - `image_project/foundation` remains free of `framework|impl|stages` imports (existing test should continue to pass after re-exports).
- Behavior tests:
  - An integration test that runs the prompt pipeline (offline fakes) and asserts:
    - artifacts exist (prompt output + transcript)
    - `outputs.prompt_pipeline` retains required metadata keys
    - transcript step paths remain stable (or any intentional change is documented in the test and in `docs/pipeline.md`).

Docs to update:

- `docs/where_things_live.md`: add `pipelinekit` as the lowest layer and explain that `image_project/foundation` and `image_project/framework` are now app-facing wrappers.
- `docs/pipeline.md`: clarify what is generic vs policy-driven, and document the override/capture policies used by image_project.

## Concrete Steps

All commands below run from the repo root.

Milestone 0 (boundaries + inventory):

    rg -n "compile_stage_nodes\\(|resolve_stage_blocks\\(|_override_primary_draft_step\\(|_block_produces_assistant_output\\(" image_project/framework/prompt_pipeline/__init__.py
    pdm python -m pytest -q tests/test_import_boundaries.py

Milestone 1 (introduce pipelinekit + re-exports):

    pdm python -m pytest -q

Milestone 2 (policy-driven compiler + image_project policies):

    pdm python -m pytest -q tests/test_config_namespace_and_stage_configs.py
    pdm python -m pytest -q tests/test_prompt_plan_modifiers.py
    pdm python -m pytest -q tests/test_stage_catalog.py
    pdm python -m pytest -q

## Validation and Acceptance

This refactor is accepted when:

1) `pipelinekit` can be imported in isolation and does not import `image_project.*` (enforced by a boundary test).
2) The existing `image_project` entrypoint still runs the prompt pipeline with no behavior regressions in:
   - stage selection (include/exclude)
   - override application semantics
   - default capture stage semantics when `prompt.output.capture_stage` is unset
3) All tests pass offline (`pdm test`), with no network access required.
4) No new silent defaults/fallbacks are introduced; any behavior that depends on convention is explicitly represented as an injected policy and logged when applied.

## Idempotence and Recovery

This refactor should be implemented as additive moves plus re-exports first, so that the work can be reverted by removing `pipelinekit/` and restoring original module bodies if needed.

If a milestone introduces widespread import failures, the safe recovery path is:

1) Revert the most recent code-motion patch.
2) Re-run `pdm python -m pytest -q tests/test_import_boundaries.py` to restore boundary sanity.
3) Re-apply the milestone with smaller steps, keeping re-exports in place until the end.

## Artifacts and Notes

When the refactor is complete, the following “map” should be true (document it in `docs/where_things_live.md`):

- Generic kernel:
  - `pipelinekit.engine.*` (node types + runner)
  - `pipelinekit.stage_types`, `pipelinekit.stage_registry`, `pipelinekit.config_namespace`
  - `pipelinekit.compiler` (policy-driven stage compilation)
- Image-project application:
  - `image_project/stages/*` (stage libraries)
  - `image_project/impl/*` (plans + experiment wiring)
  - `image_project/app/generate.py` (entrypoint + artifact writing)
  - `image_project/*` wrappers that re-export kernel symbols as needed during migration

## Interfaces and Dependencies

In `pipelinekit/compiler.py`, define policy interfaces (names may vary, but they must be explicit and documented):

    class SelectorResolver(Protocol):
        def resolve_stage_instance_ids(self, selectors: tuple[str, ...], *, available: tuple[str, ...], path: str) -> tuple[str, ...]: ...

    class OverrideTargetPolicy(Protocol):
        def targets(self, stage_block: Block, *, stage_instance_id: str, override: Any) -> tuple[tuple[str, ...], ...]: ...

    class CapturePolicy(Protocol):
        def choose_capture_stage(self, *, stage_blocks: tuple[Block, ...], configured: str | None) -> str: ...
        def attach_capture(self, stage_block: Block, *, capture_key: str) -> Block: ...

In `image_project/framework/prompt_pipeline/stage_policies.py` (new), implement the concrete policies that preserve current behavior and ensure they log their decisions at INFO/WARN with enough detail to debug selector normalization and override/capture behavior.

## Plan Revision Notes

- (2026-01-18 20:34Z) Updated Progress/Decisions/Outcomes to reflect completed implementation, recorded key decisions, and moved the ExecPlan to `.plans/5 - complete/` so future work can start from the final state. (Codex)
- (2026-01-19) Updated referenced file paths after framework organization (`image_project/framework/prompt_pipeline/*` and `image_project/framework/prompt_pipeline/stage_policies.py`). (Codex)
