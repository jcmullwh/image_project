# Plan: Docs Refresh (repo-wide)

## Task

Update repository documentation to match the current codebase: CLI entrypoints, experiment runner + plugin system, stage catalog, config schema, and artifact conventions. Produce a concrete list of remaining documentation gaps.

## Goal

After this work:

- All documentation examples run against the current repo (paths, flags, stage ids, and config keys are accurate).
- Architecture/layering docs match current module boundaries (including first-class experiments + context injector plugins).
- Generated docs (`docs/stages.md`) reflect the live stage registry.
- A tracked list of documentation gaps exists for follow-up work.

## Context

This repo is an experimentation harness whose “source of truth” is the code plus the artifacts it writes. Several docs referenced older experiment runner scripts and older stage ids (notably `blackbox_refine.iter` / `init_state` / `finalize`) that no longer exist as public stage kinds. The canonical experiment runner is now centralized and experiments live as plugins, so documentation must reflect the new architecture and CLI surface.

## Constraints

- No silent fallbacks: docs must not describe behavior that does not exist, and must not rely on implicit/quiet defaults in examples without calling them out.
- Offline testability: examples should avoid requiring network access unless explicitly noted (runs can be `--dry-run` / `--mode prompt_only`).
- Determinism: where randomness is used (experiments), docs should mention seed recording and reproducibility expectations.

## Implementation plan

1. Inventory doc files (README + `docs/*.md` + key templates under `.todos/`).
2. Validate all CLI commands and config examples against current code:
   - stage ids: `python -m image_project list-stages`
   - plan ids: `python -m image_project list-plans`
   - experiments: `python -m image_project experiments list|describe|run ...`
3. Update docs:
   - Fix stale stage ids and config key paths.
   - Fix broken example paths (`example_images/`, `scripts/`).
   - Add missing CLI flags/documentation where present in code (e.g., `run_review --compare-experiment`, `experiments --resume`).
4. Regenerate generated docs:
   - Run `pdm run update-stages-docs` and commit the updated `docs/stages.md`.
5. Write a documentation gaps list and store it as a tracked todo.

## Progress

- [x] (2026-01-20) Inventory docs and identify drift.
- [x] (2026-01-20) Patch docs for correctness (README + `docs/*.md` + legacy notes).
- [x] (2026-01-20) Regenerate `docs/stages.md` from current stage registry.
- [x] (2026-01-20) Create “docs gaps” list under `.todos/`.

## Outcomes

- Updated core docs for current pipelinekit + first-class experiments (`README.md`, `docs/experiments.md`, `docs/experiment_log.md`, `docs/pipeline.md`, `docs/where_things_live.md`, `docs/run_review.md`, `docs/openai_1-5-guidance.md`).
- Regenerated stage catalog from the live stage registry (`docs/stages.md`).
- Updated internal spec template to match current pipeline architecture (`.todos/spec_template.md`).
- Captured follow-up doc gaps as a tracked todo (`.todos/2 - ready/2026-01-20_docs_gaps.md`).
