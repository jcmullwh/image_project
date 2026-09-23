## Refactor Phase II — Surprises + Decisions Log

This log captures notable surprises encountered and decisions made while implementing `.todos/work/refactor_phase_ii.md`.

### Surprises

- The spec file `.todos/work/refactor_phase_ii.md` is effectively a single-line “blob”, which made line-based reading awkward; used targeted searches (`rg`) and extracted requirements by section headings/keywords instead.
- PowerShell environment does not provide `head`; replaced with `| Select-Object -First ...` when inspecting `rg` output.
- Several repo entry points and tools still assumed the old `StageCatalog` lived in `image_project/impl/current/prompting.py`:
  - `python -m image_project list-stages` imported a removed `list_stages()` helper.
  - `tools/generate_stages_doc.py` AST-parsed `StageCatalog.register(...)` decorators and returned 0 rows after the refactor.
  - Fixing these surfaced additional tests that were previously “accidentally” coupled to StageCatalog.
- `tools/*.py` scripts run as files (`python tools/x.py`) do not automatically have repo root on `sys.path` (script directory is `tools/`), so direct `import image_project` failed until a root path shim was added in the relevant tools.
- Converting the ToT/enclave implementation to reusable pattern builders necessarily changed transcript paths (added `fanout/` and `reduce/` structural blocks), requiring updates to tests that asserted exact step paths.

### Decisions

- **Stage kind vs stage instance:** Adopted `StageRef` (kind) + `StageInstance` (instance id) as the code-facing API; config remains string ids resolved via `StageRegistry`.
- **Stage modularization:** Split stage implementations into `image_project/stages/<group>/<stage>.py` with `STAGE: StageRef` exports; grouped by domain (`preprompt`, `standard`, `blackbox`, `refine`, `postprompt`, `direct`, `ab`, `blackbox_refine`).
- **Registry boundary:** Implemented `StageRegistry` (`image_project/stages/registry.py`) as the string-id boundary (suffix resolution + suggestions + duplicate detection); removed StageCatalog/StageCatalog-based tooling and tests.
- **Stage-owned config namespaces:** Implemented `ConfigNamespace` (`image_project/framework/config_namespace.py`) with typed getters and consumed-key enforcement. Enforced “no silent fallbacks” by raising on unknown/unconsumed keys and by validating stage-config applicability.
- **Early validation:** Added `compile_stage_nodes(...)` to validate:
  - include/exclude selectors at the stage-node level (suffix-aware),
  - stage kind defaults (`prompt.stage_configs.defaults`) via `StageRegistry`,
  - instance configs (`prompt.stage_configs.instances`) against included stages only,
  - `StageIO.requires` before execution begins.
- **Reusable algorithmic patterns:** Added `image_project/framework/blocks/patterns.py` (`fanout_then_reduce`, `generate_then_select`, `iterate`) and refactored ToT enclave to use them (`image_project/framework/refinement.py`), so ToT is no longer the lone “special” builder.
- **Prompt helpers vs stage wiring:** Truncated `image_project/impl/current/prompting.py` to remain “prompt strings + helpers” only (removed stage wiring entirely). To avoid losing blackbox helper functionality, moved `build_blackbox_isolated_idea_card_specs(...)` into `image_project/impl/current/blackbox_idea_cards.py`.
- **Generated stage docs:** Rewrote `tools/generate_stages_doc.py` to use `StageRegistry.describe()` as the source of truth and regenerated `docs/stages.md` from that. Added IO contract fields to the generated markdown.
- **Config examples:** Added `prompt.stage_configs.{defaults,instances}` stubs to `config/config.yaml`, `config/config.full_example.yaml`, and `config/config_repo_specific.yaml` to document the new structure without enabling unknown-key failures.
- **Test coverage:** Added explicit Phase II unit coverage for `ConfigNamespace` strict typing/constraints and for stage config/IO validation failures (`tests/test_config_namespace_and_stage_configs.py`).

### Notable behavioral changes (intentional)

- Stage subsets selected via `prompt.stages.include` can now fail earlier due to missing `StageIO.requires` outputs (pre-run validation instead of runtime failure).
- ToT/enclave transcript paths now include explicit `fanout/` and `reduce/` blocks (nested algorithmic structure is visible in the transcript).

### Follow-ups (optional)

- Many stages currently accept no stage-owned config keys (they `assert_consumed()` immediately). As experiments introduce per-stage knobs, stage modules should adopt `ConfigNamespace.get_*` usage and document supported keys (and/or extend stage docs generation to include config schema notes).

