# Experiment Log

This file summarizes experiments that have been planned and/or executed in this repo. Artifact folders live under `_artifacts/experiments/` and should be treated as the source of truth.

## Artifact conventions

- `experiment_plan.json`: planned variants, seeds, and config provenance (`config_meta`).
- `experiment_results.json`: per-run status + primary outputs (only present when the runner executed).
- `logs/runs_index.jsonl`: per-run metadata, resolved prompt pipeline, and artifact paths.

## Experiments (newest first)

### Planned (dry-run): 3x3 blackbox profile routing, no ToT

- **Artifacts:** `_artifacts/experiments/_dryrun_blackbox_swap/`
- **Experiment id:** `exp3x3_20251227_222203`
- **Runner:** `tools/run_experiment_3x3.py`
- **Goal:** Replace the old `standard` baseline with a materially different blackbox configuration and remove ToT refinement everywhere; isolate whether profile routing (`raw` vs `generator_hints`) changes blackbox outputs.
- **Distinct differences (variants):**
  - **A**: `prompt.plan=blackbox`, scoring on, `prompt.scoring.{judge,final}_profile_source=generator_hints`, `prompt.refinement.policy=none`
  - **B**: `prompt.plan=blackbox`, scoring on, `prompt.scoring.{judge,final}_profile_source=raw`, `prompt.refinement.policy=none`
  - **C**: `prompt.plan=simple_no_concepts`, scoring off, `prompt.refinement.policy=none` (concept selection is ignored by this plan)
- **Status:** dry-run only (no `experiment_results.json`).

### 2025-12-26/25: A/B refinement-block (prompt-only) smoke runs

- **Runners:** `tools/run_experiment_ab_refinement_block.py`
- **Goal:** Test whether adding an explicit "refinement block checklist" to the middle prompt improves the final strict one-line prompt formatting.
- **Distinct differences (variants):**
  - **A** (`no_refinement_block`): middle stage is `ab.scene_refine_no_block`
  - **B** (`with_refinement_block`): middle stage is `ab.scene_refine_with_block`
  - Everything else is held constant (same token generation, same draft stage, same final formatting stage, `prompt.refinement.policy=none`, `prompt.plan=custom`, `run.mode=prompt_only`, `context.enabled=false`, scoring off).
- **Runs executed (all 2/2 success):**
  - `_artifacts/experiments/_smoke_ab_run/` - `exp_ab_refinement_block_20251225_191749`
  - `_artifacts/experiments/_smoke_ab_run2/` - `exp_ab_refinement_block_20251225_191816`
  - `_artifacts/experiments/_smoke_ab_refinement_block_run/` - `exp_ab_refinement_block_20251226_075215`
- **Planned-only (no results file):**
  - `_artifacts/experiments/_smoke_ab/` - `exp_ab_refinement_block_20251225_191659` (runs_per_variant=2)
  - `_artifacts/experiments/_smoke_ab_refinement_block/` - `exp_ab_refinement_block_20251226_075116`

### A/B SceneSpec JSON intermediary (prompt-only)

- **Runner:** `tools/run_experiment_ab_scenespec_json_intermediary.py`
- **Goal:** Compare a prose-refinement path vs a SceneSpec JSON intermediary path for producing the final strict one-line prompt.
- **Distinct differences (variants):**
  - **A** (`prose_refine`): `ab.scene_refine_with_block` -> `ab.final_prompt_format` (captures `ab.final_prompt_format`)
  - **B** (`scenespec_json`): `ab.scene_spec_json` -> `ab.final_prompt_format_from_scenespec` (captures `ab.final_prompt_format_from_scenespec`)
  - Common: starts with `ab.random_token` -> `ab.scene_draft`, `prompt.plan=custom`, `prompt.refinement.policy=none`, scoring off, `run.mode=prompt_only`, `context.enabled=false`.
- **Status:** no artifacts currently found under `_artifacts/experiments/*_ab_scenespec_json_intermediary/` (they may have been deleted/cleaned).

### 2025-12-25: 3x3 (full image runs) - baseline vs blackbox vs simple_no_concepts

These runs predate the current `tools/run_experiment_3x3.py` variant definitions; use the per-run `runs_index.jsonl` + transcripts as the canonical record of what actually executed.

#### Run 1 - `exp3x3_20251225_103121`

- **Artifacts:** `_artifacts/experiments/20251225_103121_3x3/`
- **Runner:** `tools/run_experiment_3x3.py`
- **Goal:** Compare three prompt pipeline variants against the same fixed concepts and seeds to assess impact on final prompt quality and resulting images.
- **Distinct differences (variants):**
  - **A**: `prompt.plan=standard`, scoring off, `prompt.refinement.policy=tot`
  - **B**: `prompt.plan=blackbox`, scoring on (`num_ideas=8`), `prompt.refinement.policy=tot`
  - **C**: `prompt.plan=simple_no_concepts`, scoring off, `prompt.refinement.policy=none` (skips concept selection/filtering)
- **Shared concepts (fixed):** bioluminescent koi pond; secret romance; optimistic serenity; minimalist courtyard installation; three-quarter angle; stylized digital illustration; winter golden hour; neon complementary palette
- **Result:** 9/9 success (see `_artifacts/experiments/20251225_103121_3x3/experiment_results.json`).

#### Run 2 - `exp3x3_20251225_121028`

- **Artifacts:** `_artifacts/experiments/20251225_121028_3x3/`
- **Runner:** `tools/run_experiment_3x3.py`
- **Goal:** Same as Run 1, re-run with a different base seed while holding the variant definitions constant.
- **Distinct differences (variants):** same A/B/C as Run 1
- **Shared concepts (fixed):** bioluminescent koi pond; secret romance; optimistic serenity; minimalist courtyard installation; three-quarter angle; stylized digital illustration; winter golden hour; neon complementary palette
- **Result:** 9/9 success (see `_artifacts/experiments/20251225_121028_3x3/experiment_results.json`).
