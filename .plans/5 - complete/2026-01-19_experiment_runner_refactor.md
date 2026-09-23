# Plan: First-Class Experiments (Runner + Plugins)

## Goal

Replace duplicated standalone experiment scripts in `tools/` with:

- A single canonical experiment runner (infra + artifact writing).
- An experiment registry + plugin discovery (similar to prompt plan plugins).
- Experiment definitions as plugins under `image_project/impl/current/experiment_plugins/`.
- CLI surface: `python -m image_project experiments ...`

## Non-goals (for this change)

- Rewriting stage logic or prompt policy content.
- Changing artifact indexer behavior or schema beyond making experiment plans consistent.
- Adding new online dependencies or non-offline tests.

## Constraints / invariants to preserve

- No silent fallbacks: missing/invalid config should fail clearly.
- Determinism: per-run seeds recorded; deterministic outputs given same seed + config.
- Transcript remains canonical: experiment metadata is present in transcripts and plan artifacts.
- Offline testability: tests must not require network or external binaries.

## Work items

1. Implement experiment plugin interface + registry
   - `image_project/impl/current/experiments.py`
   - `image_project/impl/current/experiment_plugins/__init__.py` with discovery
2. Implement canonical experiment runner
   - Load config + extract `experiment_runners`
   - Compute output dirs + apply standard overrides
   - Write `experiment_plan.json` + `experiment_plan_full.json`
   - Execute runs + write `experiment_results.json`
   - Optional: write `pairs.json` for A/B experiments
   - Update artifacts index once at end
3. Add CLI plumbing
   - `image_project experiments list|describe|run ...`
4. Port existing experiments into plugins
   - `3x3` (fix removed prompt blocks: migrate to `prompt.stage_configs.*`)
   - `profile_v5_3x3`
   - `ab_refinement_block`
   - `ab_scenespec_json_intermediary` (+ analysis hook)
5. Update docs + scripts
   - Update `docs/experiments.md` (remove legacy `prompt.scoring` examples)
   - Update `docs/experiment_log.md`, `docs/where_things_live.md`, `README.md`
   - Update `pyproject.toml` pdm scripts to call the new CLI runner
6. Validate
   - Run `pytest`
   - Run at least one `--dry-run` experiment via CLI to ensure `experiment_plan_full.json` is produced and config errors surface early

