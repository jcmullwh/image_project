# Experiment Artifacts and Logs

Experiments are executed via `python -m image_project experiments run <name>` and write a self-contained artifact directory under `_artifacts/experiments/` by default.

Note: `_artifacts/` is gitignored, so treat this file as a description of conventions rather than a catalog of checked-in runs.

## Where experiments write

`python -m image_project experiments run <name>` writes a single experiment directory:

- Default: `./_artifacts/experiments/<YYYYMMDD_HHMMSS>_<experiment_name>/`
- Override: `--output-root <dir>`
- Resume: `--resume --output-root <existing_experiment_dir>`

Inside `<output_root>/` (top-level):

- `experiment.log`: runner log (stdout mirror + debug details)
- `experiment_plan.json`: compact, portable plan summary (written for non-resume runs)
- `experiment_plan_full.json`: expanded plan with per-run `cfg_dict` + prompt pipeline compilation results (written even for `--dry-run`)
- `experiment_results.json`: per-run results (written when runs execute; merged on `--resume`)
- `pairs.json`: optional pairing manifest for A/B-style experiments (written when the plugin provides it)

Standard subdirectories (from runner-owned overrides):

- `logs/`: per-run operational logs, transcripts, run-review reports, and `runs_index.jsonl`
- `generated/`: images and `titles_manifest.csv` (only in `run.mode=full`)
- `upscaled/`: upscaled images (only when enabled)

## Plan vs results

- Use `experiment_plan_full.json` to validate whether each planned run's config parses and the prompt pipeline compiles; a run with errors has `config_error`.
- By default, `experiments run` aborts if any planned run has a config error; override with `--no-fail-on-config-error`.

## Indexing artifacts

To build a query-friendly index from `_artifacts/`:

```bash
python -m image_project index-artifacts
# or
pdm run index-artifacts
```

This writes (by default) under `_artifacts/index/`:

- `experiments_index.json`
- `experiment_registry.csv`
- `run_registry.csv`
- `image_registry.csv`

## Notes for experiment authors

- Determinism: experiments should set `prompt.random_seed` for reproducibility and should record a base seed under `experiment_plan.json` (`experiment_meta.base_seed` by convention).
- Paired experiments: implement `build_pairs_manifest(...)` so run-review can compare pairs by run index via `--compare-experiment <output_root>`.
