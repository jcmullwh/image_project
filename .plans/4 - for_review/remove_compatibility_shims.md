# Remove compatibility shims and deprecated aliases (canonicalize imports, config keys, and artifact schemas)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `.todos/PLANS.md` (ExecPlan requirements) and `agents.md` (repo invariants: no silent fallbacks/defaults, heavy logging, offline tests, deterministic randomness when used, and the `.plans/*` lifecycle for plans).

This ExecPlan builds on the completed kernel extraction work in `.plans/5 - complete/pipelinekit_kernel_refactor.md`. The intent here is to retire the temporary migration scaffolding (re-export modules, deprecated aliases, and legacy parsing paths) that still “snake” through the repo.

## Purpose / Big Picture

After several refactors (most recently extracting the reusable `pipelinekit/` kernel), the repository still contains multiple “compatibility” layers whose only job is to keep old import paths, old config keys, old plan names, or old artifact schemas working. Even when these shims log warnings, they still hide decisions, split the codebase into parallel paths, and make it harder to reason about what is canonical.

After this change:

1) There is exactly one canonical import path for pipeline primitives (`pipelinekit.engine.*`) and stage authoring/compiler primitives (`pipelinekit.*`). Internal code and tests import those directly.
2) Deprecated modules that only re-export prompt policy are removed; prompt policy is imported from `image_project.prompts.*`.
3) Deprecated config keys and plan aliases are removed and fail loudly with a clear error message.
4) Legacy artifact parsing (CSV schema compatibility) is either removed entirely or quarantined behind an explicit “legacy/migration” tool so the default runtime path does not silently guess schemas.

The visible “proof” of correctness is that `pdm run test` is green offline, and that:

- `rg` finds no remaining compatibility shim markers in mainline code.
- Running `python -m image_project list-stages` / `list-plans` still works.
- Old compatibility paths now fail loudly (missing module / invalid config key / unknown plan) with actionable guidance in the error text and docs.

## Progress

- [x] (2026-01-18 22:18Z) Authored this ExecPlan and placed it in `.plans/2 - ready/`.
- [x] (2026-01-18 22:22Z) Moved this ExecPlan to `.plans/3 - in_progress/` and began implementation.
- [x] (2026-01-18 22:33Z) Inventoried remaining shims and confirmed baseline tests are green (`pdm run test`).
- [x] (2026-01-18 22:43Z) Canonicalized imports to `pipelinekit.*` and deleted import-path compatibility wrapper modules (and the stage-types re-export).
- [x] (2026-01-18 22:52Z) Canonicalized prompt-policy imports to `image_project.prompts.*` and deleted the deprecated prompt-policy shim modules.
- [x] (2026-01-18 22:52Z) Removed the deprecated entrypoint shim (`main.py`) and updated docs/scripts accordingly.
- [x] (2026-01-18 22:59Z) Removed deprecated config key support (`image.save_path`) everywhere (config validation, run-review helpers, docs) and added a regression test.
- [x] (2026-01-18 22:59Z) Removed legacy plan alias support (`blackbox_refine_legacy`) everywhere (plan registry, config validation, tests, docs).
- [x] (2026-01-18 23:51Z) Removed legacy scoring (`legacy_v0`) and legacy generations CSV schema support from mainline; added an explicit offline migration tool (`tools/migrate_generations_csv_legacy_to_v2.py`) and regression tests; confirmed tests are green (`pdm run test`).
- [x] (2026-01-19 00:16Z) Removed unreachable legacy compilation code in `image_project/framework/prompt_pipeline/__init__.py` (post-`return` dead implementations) and deleted the now-dead helper functions that were only referenced by that code.
- [x] (2026-01-19 00:16Z) Added guardrail tests to prevent compatibility shims from reappearing (module/file absence + no re-export marker strings).
- [x] (2026-01-19 00:16Z) Ran full offline test suite and updated developer-facing docs that still referenced removed shim paths.
- [x] (2026-01-19 00:17Z) Moved this ExecPlan to `.plans/4 - for_review/`.
- [ ] Move this ExecPlan to `.plans/5 - complete/` when accepted.

## Surprises & Discoveries

- Observation: The repo currently contains explicit import-path compatibility wrappers introduced during the `pipelinekit` extraction.
  Evidence: `image_project/foundation/pipeline.py`, `image_project/foundation/messages.py`, `image_project/framework/config_namespace.py`, `image_project/framework/stage_types.py`, `image_project/framework/stage_registry.py` (each is a “Compatibility re-export … canonical now in `pipelinekit.*`”).

- Observation: There are deprecated prompt-policy shim modules that only re-export prompt builders from `image_project.prompts.*`.
  Evidence: `image_project/impl/current/prompting.py`, `image_project/impl/current/blackbox_refine_prompts.py` (“DEPRECATED: compatibility shim … canonical … `image_project.prompts.*`”).

- Observation: Root `main.py` is an explicit deprecated entrypoint shim that prints a warning and then forwards to `image_project.app.generate`.
  Evidence: `main.py` and `README.md` (“Root `main.py` is a deprecated compatibility shim”).

- Observation: Config parsing still accepts the deprecated key `image.save_path` as an alias for `image.generation_path`, and other code also consults it.
  Evidence: `image_project/framework/config.py` (reads both, warns, then uses `save_path` as fallback), and `image_project/run_review/cli.py` (uses `generation_path or save_path`).

- Observation: There is a backwards-compatible plan alias `blackbox_refine_legacy` that exists only to preserve an old name.
  Evidence: `image_project/impl/current/plan_plugins/variations.py` (`BlackboxRefineLegacyPromptPlan`), `image_project/framework/config.py` (plan_requires_blackbox_refine includes `blackbox_refine_legacy`), `tests/test_profile_io.py`.

- Observation: Some "legacy" behaviors were still carried as first-class options in mainline code (not just docs), including the novelty scoring method `legacy_v0` and the generations CSV "legacy schema" reader.
  Evidence: Prior to Milestone 5, `image_project/framework/config.py` allowed `legacy_v0`, `image_project/framework/scoring.py` branched on it, and `image_project/framework/artifacts/index.py` schema-sniffed `generations.csv` and recorded `legacy_output` (all removed in Milestone 5).

- Observation: `image_project/framework/prompt_pipeline/__init__.py` now delegates compilation to `pipelinekit.compiler` but still contains a full legacy compilation implementation after an early `return`, making the file harder to trust and maintain.
  Evidence: `image_project/framework/prompt_pipeline/__init__.py` `compile_stage_nodes()` returns at ~line 226 and the old implementation continues afterward, unreachable.

- Observation: `image_project/stages/types.py` was effectively acting as a refactor-era re-export because it forwarded to `image_project/framework/stage_types.py`, which was a `pipelinekit` compatibility wrapper.
  Evidence: `image_project/stages/types.py` re-exported stage types from `image_project.framework.stage_types` (deleted during Milestone 1).

- Observation: Some developer-facing markdown files outside `docs/` still refer to the removed `image_project/impl/current/prompting.py` path.
  Evidence: `new_developer_experiment_preparation.md` and related files reference `image_project/impl/current/prompting.py`.

## Decision Log

- Decision: Treat “compatibility shims” as any code path whose primary purpose is to keep an old refactor-era interface working (old import paths, deprecated config keys, deprecated plan names, or legacy artifact schemas), and remove them from the default runtime path.
  Rationale: These shims create parallel semantics, hide decisions, and violate the repo’s “no silent fallbacks” spirit even when warnings are printed.
  Date/Author: 2026-01-18 (Codex)

- Decision: Prefer deletion over “soft deprecation” when retiring shims (i.e., do not leave stubs that silently redirect; if we keep something, it must be explicit and quarantined under a `legacy`/migration tool).
  Rationale: This repo is an experimental harness; ambiguity and hidden redirects are more damaging than breaking an old path.
  Date/Author: 2026-01-18 (Codex)

- Decision: Remove `legacy_v0` novelty scoring from mainline (config + implementation + docs/tests); treat the pre-removal revision history as the “legacy reference.”
  Rationale: The legacy method exists solely for backwards comparison and adds branching complexity to a core scoring path; in an experimentation harness, it is clearer to pin comparisons to a known revision instead of carrying legacy code indefinitely.
  Date/Author: 2026-01-18 (Codex)

- Decision: Remove "legacy schema" parsing from `index-artifacts` and instead provide an explicit offline migration tool to convert historical `generations.csv` into the v2 schema.
  Rationale: Schema sniffing is a form of implicit fallback; it hides data quality problems and makes index output ambiguous. A one-time migration keeps the default path strict while preserving historical data via an explicit step.
  Date/Author: 2026-01-18 (Codex)

- Decision: Make `image_project index-artifacts` surface warnings (stderr) and return a non-zero exit code when warnings are present.
  Rationale: The indexer is intentionally best-effort, but warnings must not be silent; failing the command makes schema problems and missing artifacts visible in automation.
  Date/Author: 2026-01-18 (Codex)

- Decision: Delete `image_project/stages/types.py` and import `StageRef`/`StageInstance`/`StageIO` directly from `pipelinekit.stage_types` everywhere.
  Rationale: The goal is a single canonical path for stage authoring primitives; keeping a second re-export path perpetuates shim sprawl and made it impossible to delete the `image_project.framework.stage_types` wrapper cleanly.
  Date/Author: 2026-01-18 (Codex)

## Outcomes & Retrospective

This work succeeded in collapsing the repo back to a single set of canonical APIs and schemas, with loud failures where legacy behavior previously existed. The biggest improvement is not “less code,” but less ambiguity: there is now exactly one place to look for pipeline primitives (`pipelinekit.*`), exactly one place to look for prompt policy text (`image_project/prompts/*`), and exactly one place to look for stage registrations (`image_project/stages/*` assembled by `image_project/stages/registry.py`).

The core goal of “no silent fallbacks” is materially better satisfied now. Legacy config keys and schema sniffing have been removed from runtime paths and replaced with either explicit errors or explicit migration tooling. Concretely:

- Import-path compatibility modules were deleted and all internal call sites updated to import from `pipelinekit.*` directly. There is no longer a “wrapper” layer that can drift or behave differently.
- Prompt-policy compatibility shims were deleted and all code now imports the canonical helpers under `image_project/prompts/*`.
- The deprecated config key `image.save_path` no longer behaves as an alias. It is rejected with a clear `ValueError` instructing the user to use `image.generation_path`.
- The plan alias `blackbox_refine_legacy` is gone; configs must use `blackbox_refine` (or another current plan).
- The novelty scoring method `legacy_v0` was removed from config + implementation + docs/tests; comparisons to the legacy behavior must be done by pinning a historical revision instead of carrying the branch in mainline.
- `index-artifacts` no longer schema-sniffs legacy `generations.csv` headers or emits a `legacy_output` field. It requires the v2 schema and points users to an explicit offline migration tool (`tools/migrate_generations_csv_legacy_to_v2.py`) when legacy headers are encountered.

To prevent shim creep from returning, new guardrail tests assert that known shim modules/files are absent and that “compatibility shim” marker strings do not appear in `image_project/` or `pipelinekit/`. This makes the refactor intent durable and reviewable.

From a “can we prove it still works?” standpoint, the offline test suite remains green (199 tests at the time of this write-up), which is the primary signal that behavior and invariants held through the deletions.

Tradeoffs and “breaks by design” worth calling out explicitly:

- Any external scripts still importing deleted modules (e.g., `image_project.foundation.pipeline`) will now fail immediately with `ModuleNotFoundError`. This is intentional: the repo should force canonical import paths.
- Any existing configs still using removed keys/plans/methods (`image.save_path`, `blackbox_refine_legacy`, `prompt.scoring.novelty.method: legacy_v0`) will now fail fast with actionable messages.
- `image_project index-artifacts` now returns a non-zero exit code when warnings exist, to avoid quiet partial indexes in automation. This may require updating any callers that previously assumed `0` on “best effort,” but it makes data problems visible.

The main “lesson learned” is that leaving dead implementations after early returns and leaving schema sniffing in “best-effort” utilities both create long-lived uncertainty. Removing those paths and adding explicit migration tooling is a cleaner pattern for an experimentation harness: strict mainline + explicit conversions when reading historical data.

## Context and Orientation

Key terms used in this plan:

- A “compatibility shim” is a module, config alias, or parsing path whose primary job is to preserve behavior after a refactor by supporting old names or old schemas.
- “Canonical” means the one path we want all internal code and docs to reference going forward.

Current canonical locations (post-kernel refactor):

- Pipeline engine primitives: `pipelinekit/engine/pipeline.py` and `pipelinekit/engine/messages.py`.
- Stage authoring + compilation primitives: `pipelinekit/stage_types.py`, `pipelinekit/stage_registry.py`, `pipelinekit/config_namespace.py`, `pipelinekit/compiler.py`.
- Prompt policy helpers: `image_project/prompts/*` (e.g., `image_project/prompts/blackbox.py`, `image_project/prompts/standard.py`).
- Image-project-specific stage selection/override/capture conventions: `image_project/framework/prompt_pipeline/stage_policies.py`.

Known compatibility shim locations to remove (mainline):

- Import-path wrappers:
  - `image_project/foundation/pipeline.py` and `image_project/foundation/messages.py`
  - `image_project/framework/config_namespace.py`, `image_project/framework/stage_types.py`, `image_project/framework/stage_registry.py`
- Prompt policy wrappers:
  - `image_project/impl/current/prompting.py`
  - `image_project/impl/current/blackbox_refine_prompts.py`
- Deprecated entrypoint:
  - `main.py`
- Deprecated config key and other aliases:
  - `image.save_path` (alias for `image.generation_path`) in `image_project/framework/config.py`, `image_project/run_review/cli.py`, and docs.
- Legacy plan alias:
  - `blackbox_refine_legacy` in `image_project/impl/current/plan_plugins/variations.py`, config validation, and tests.
- Legacy behaviors embedded in mainline:
  - novelty method `legacy_v0` (`image_project/framework/config.py`, `image_project/framework/scoring.py`, docs/tests)
  - generations CSV "legacy schema" parsing (`image_project/framework/artifacts/index.py`)
- Unreachable legacy code:
  - the old `compile_stage_nodes` implementation in `image_project/framework/prompt_pipeline/__init__.py`.

## Plan of Work

This work is intentionally a “migration cleanup” refactor. The safe way to do it is to make changes in small, verifiable increments: first update all internal imports and call sites to use canonical APIs while shims still exist, then delete the shims once nothing depends on them, and finally add tests that prevent them from reappearing.

### Milestone 0: Baseline + inventory (no behavior change)

1) Move this plan to `.plans/3 - in_progress/` when starting implementation.
2) Run the full test suite to establish a baseline.
3) Inventory compatibility shims by searching for:
   - “Compatibility re-export”
   - “DEPRECATED: compatibility shim”
   - “Deprecated entrypoint”
   - `save_path`, `blackbox_refine_legacy`, `legacy_v0`, and `legacy_output`
4) Record any additional compatibility layers found (for example, alias keys, schema sniffers, or re-export modules not already listed) in the `Surprises & Discoveries` section before editing code.

### Milestone 1: Remove import-path compatibility wrappers (pipeline engine + stage kit)

Goal: no `image_project.*` module exists solely to re-export `pipelinekit.*`.

1) Update internal imports across `image_project/`, `tools/`, and `tests/`:
   - Replace `from image_project.foundation.pipeline import ...` with `from pipelinekit.engine.pipeline import ...`.
   - Replace `from image_project.foundation.messages import MessageHandler` with `from pipelinekit.engine.messages import MessageHandler`.
   - Replace `from image_project.framework.config_namespace import ConfigNamespace` with `from pipelinekit.config_namespace import ConfigNamespace`.
   - Replace `from image_project.framework.stage_types import ...` with `from pipelinekit.stage_types import ...`.
   - Replace `from image_project.framework.stage_registry import StageRegistry` with `from pipelinekit.stage_registry import StageRegistry`.

2) Decide whether to keep `image_project/stages/types.py` as an image-project convenience re-export of `pipelinekit.stage_types`:
   - If the goal is “single canonical path everywhere,” delete it and update call sites to import from `pipelinekit.stage_types`.
   - If the goal is “remove only refactor-era compatibility shims,” update it to re-export from `pipelinekit.stage_types` and keep it as an intentional image-project API.
   Record the decision here and reflect it in docs (`docs/where_things_live.md`).

3) Once all internal imports are updated and tests are green, delete the wrapper modules:
   - `image_project/foundation/pipeline.py`
   - `image_project/foundation/messages.py`
   - `image_project/framework/config_namespace.py`
   - `image_project/framework/stage_types.py`
   - `image_project/framework/stage_registry.py`

4) Update docs that describe these wrappers:
   - `docs/where_things_live.md` (remove “compatibility wrappers” language if the wrappers are deleted).

### Milestone 2: Remove deprecated prompt-policy shim modules

Goal: prompt templates/builders are imported from `image_project.prompts.*` only.

1) Replace any imports of `image_project.impl.current.prompting` with imports from the specific `image_project.prompts.*` module that contains the referenced symbol.
2) Remove the shim modules:
   - `image_project/impl/current/prompting.py`
   - `image_project/impl/current/blackbox_refine_prompts.py`
3) Update docs that still point to the shim module as canonical:
   - `README.md` currently claims prompt helpers live in `image_project/impl/current/prompting.py`; update it to point to `image_project/prompts/*` instead.

### Milestone 3: Remove the deprecated root entrypoint (`main.py`)

Goal: the only documented entrypoint is `python -m image_project generate` (or `pdm run generate`).

1) Ensure nothing in-repo imports `main.py` or relies on `python main.py` (confirm with `rg`).
2) Delete `main.py`.
3) Update docs to remove mention of `main.py` as a compatibility shim (`README.md`).

### Milestone 4: Remove deprecated config key support (`image.save_path`)

Goal: `image.generation_path` is the only supported key; `image.save_path` fails loudly.

1) In `image_project/framework/config.py`:
   - Remove `save_path` from the unknown-key schema and from parsing logic.
   - If `image.save_path` is present in user config, raise `ValueError` with a clear, actionable message (do not warn+continue).
2) In `image_project/run_review/cli.py`:
   - Stop reading `image.save_path` as a fallback for `image.generation_path`.
   - If a pipeline config uses `save_path`, fail loudly (or require the user to update their config first).
3) Update docs (`README.md`) to remove the deprecated alias mention.
4) Add/adjust tests:
   - A unit test that `RunConfig.from_dict()` raises when `image.save_path` is present (both strict and non-strict modes if the code supports both).

### Milestone 5: Remove legacy plan alias support (`blackbox_refine_legacy`)

Goal: `blackbox_refine` is the only supported plan name; `blackbox_refine_legacy` fails loudly.

1) Delete `BlackboxRefineLegacyPromptPlan` from `image_project/impl/current/plan_plugins/variations.py`.
2) Remove `blackbox_refine_legacy` from config validation logic in `image_project/framework/config.py`.
3) Update tests that assert the alias exists (`tests/test_profile_io.py`) to use the canonical plan name, or remove the test if it is purely about legacy behavior.
4) Update any docs that mention the alias (if any).

### Milestone 6: Remove legacy behaviors embedded in mainline (scoring + artifact schemas)

Goal: eliminate refactor-era “legacy support” branches from mainline code paths; preserve historical data access via an explicit, offline migration tool rather than implicit fallbacks.

1) Remove novelty scoring method `legacy_v0`:
   - In `image_project/framework/config.py`, remove `legacy_v0` from the validated literal/validation logic.
   - In `image_project/framework/scoring.py`, remove `legacy_v0` branches and delete helper functions that exist only for that method.
   - Update `docs/scoring.md` to remove mention of `legacy_v0`.
   - Update or remove tests that depend on `legacy_v0` behavior (`tests/test_blackbox_scoring.py`).

2) Remove generations CSV "legacy schema" parsing from `image_project/framework/artifacts/index.py`:
   - Replace `_read_generations_csv_any()` with a strict v2-only reader (raise loudly if required headers are missing).
   - Remove `legacy_output` plumbing from record schemas and index outputs.
   - Add a one-off migration tool under `tools/` (offline) that converts legacy `generations.csv` into a v2 `generations_v2.csv` with the canonical headers used by the indexer.
   - Add offline unit tests for the migration tool and for the v2-only reader (use small fixture CSV strings written into a temp dir).

### Milestone 7: Remove unreachable legacy compilation code in `image_project/framework/prompt_pipeline/__init__.py`

Goal: the file contains only the canonical compilation path (the `pipelinekit.compiler` call) and its still-relevant helpers.

1) Delete the unreachable legacy implementation body below the early `return` in `compile_stage_nodes()`.
2) Remove any helper functions/types in `image_project/framework/prompt_pipeline/__init__.py` that were only used by the deleted code (verify with `rg` / IDE references).
3) Ensure the module still satisfies repo invariants (no silent fallbacks, clear errors with full paths).

### Milestone 8: Guardrails + final cleanup

1) Add guardrail tests that fail if compatibility shims reappear, for example:
   - A test that `importlib.util.find_spec("image_project.foundation.pipeline")` (and other removed modules) returns `None`.
   - A test that deprecated config keys (`image.save_path`) raise loudly with a clear error message.
2) Run `pdm run update-stages-docs` if doc generation depends on updated import paths.
3) Final repository sweep:
   - `rg` for “Compatibility re-export”, “compatibility shim”, “Deprecated entrypoint”, and for removed key/name strings.
4) Run full tests and validate CLI commands (`list-stages`, `list-plans`, `generate` in prompt-only mode).

## Concrete Steps

All commands below run from the repo root (`i:\\code\\image_project`).

Baseline:

    pdm run test

Inventory commands:

    rg -n "Compatibility re-export|compatibility shim|Deprecated entrypoint" -S .
    rg -n "\\bsave_path\\b|\\bblackbox_refine_legacy\\b|\\blegacy_v0\\b|\\blegacy_output\\b" -S image_project tests docs README.md tools
    rg -n "image_project\\.foundation\\.(pipeline|messages)" -S image_project tests tools
    rg -n "image_project\\.framework\\.(config_namespace|stage_types|stage_registry)" -S image_project tests tools
    rg -n "image_project\\.impl\\.current\\.prompting" -S image_project tests tools

Implementation cadence (repeat after each milestone):

    pdm run test

Doc regeneration (only if needed / after import-path changes):

    pdm run update-stages-docs

## Validation and Acceptance

This work is accepted when all of the following are true:

1) Offline tests are green: `pdm run test` succeeds.
2) No mainline compatibility wrappers remain:
   - `image_project/foundation/pipeline.py` and `image_project/foundation/messages.py` are deleted (or, if intentionally kept as part of a deliberate API, they are no longer described or used as “compatibility” and the decision is documented).
   - `image_project/framework/config_namespace.py`, `image_project/framework/stage_types.py`, `image_project/framework/stage_registry.py` are deleted.
   - `image_project/impl/current/prompting.py` and `image_project/impl/current/blackbox_refine_prompts.py` are deleted.
3) Deprecated interfaces fail loudly and clearly:
   - `image.save_path` in config raises a `ValueError` that tells the user to use `image.generation_path`.
   - `prompt.plan=blackbox_refine_legacy` is rejected as unknown.
4) Legacy schema/behavior support is removed from mainline:
   - Docs/tests no longer reference `legacy_v0`.
   - `index-artifacts` no longer schema-sniffs “legacy” CSV headers; it requires the v2 schema.
   - A migration tool exists for converting historical legacy CSVs into v2 explicitly.
5) `image_project/framework/prompt_pipeline/__init__.py` contains only the canonical compilation path; there is no unreachable legacy implementation left in-file.
6) Docs are updated (`README.md`, `docs/where_things_live.md`, and any other touched docs) so a novice can find the canonical paths without encountering deprecated guidance.

## Idempotence and Recovery

This refactor should be implemented as “edit-imports first, delete shims last” to keep the repo runnable at every step.

Safe recovery strategy if a milestone breaks many imports:

1) Revert only the most recent change set (use `git restore` on the touched files).
2) Re-run `pdm run test` to re-establish baseline.
3) Re-apply the milestone in smaller slices (e.g., update imports in `tests/` first, then `tools/`, then `image_project/`).

## Artifacts and Notes

At completion, the “canonical map” should read:

- Pipeline engine: `pipelinekit.engine.pipeline`, `pipelinekit.engine.messages`
- Stage authoring/compiler: `pipelinekit.stage_types`, `pipelinekit.stage_registry`, `pipelinekit.config_namespace`, `pipelinekit.compiler`
- Prompt policy: `image_project.prompts.*`
- Image project policies: `image_project/framework/prompt_pipeline/stage_policies.py`

Any legacy support that remains should be physically isolated (for example under `legacy/` or `tools/legacy_*`) and must require explicit invocation (no automatic schema guessing or key fallbacks in the default path).

## Interfaces and Dependencies

This plan should not add new third-party dependencies. Use:

- Existing stdlib modules (`csv`, `json`, `pathlib`, `importlib`) for any migration tooling or tests.
- Existing test stack (`pytest`) and the existing CLI entrypoints (`python -m image_project ...`).

If a migration tool is created (for legacy `generations.csv` → `generations_v2.csv`), define it as a small, offline script under `tools/` with:

- A single `main(argv: list[str] | None = None) -> int` entrypoint.
- Clear docstring explaining expected input columns and output schema.
- Loud failure on unexpected headers or ambiguous columns (no silent guessing).

## Plan Revision Notes

- (2026-01-18 22:18Z) Initial draft of the compatibility-shim removal plan, created in `.plans/2 - ready/`. (Codex)
- (2026-01-18 22:18Z) Made Milestone 6 fully prescriptive: remove `legacy_v0` and legacy CSV schema support from mainline and provide an explicit migration tool, to avoid leaving “options” unresolved. (Codex)
- (2026-01-18 22:22Z) Moved plan into `.plans/3 - in_progress/` to start implementation. (Codex)
- (2026-01-18 22:43Z) Completed Milestone 1: updated imports to `pipelinekit.*`, removed the compatibility wrapper modules, updated `docs/where_things_live.md`, and confirmed tests remain green. (Codex)
- (2026-01-18 22:52Z) Completed Milestone 2/3 pieces: deleted `image_project/impl/current/*prompting*.py` compatibility shims, removed root `main.py`, updated README/docs, and confirmed tests remain green. (Codex)
- (2026-01-18 23:51Z) Completed Milestone 5: removed `legacy_v0` scoring and legacy generations CSV parsing from mainline, added `tools/migrate_generations_csv_legacy_to_v2.py` + tests, and updated `image_project index-artifacts` to surface warnings via exit code. (Codex)
- (2026-01-19 00:16Z) Completed Milestone 6: removed unreachable compilation code in `image_project/framework/prompt_pipeline/__init__.py`, added guardrail tests for shim removal, updated legacy onboarding docs referencing removed paths, and confirmed tests remain green. (Codex)
- (2026-01-19) Updated referenced file paths after framework organization (`image_project/framework/artifacts/index.py` and `image_project/framework/prompt_pipeline/stage_policies.py`). (Codex)
