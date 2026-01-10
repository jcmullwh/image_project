# New Developer Experiment Preparation (A/B prompt refinement block)

Goal: help a new developer successfully set up and run an A/B experiment that tests whether adding a **refinement block** to the middle prompt improves the final image prompt output.

This file is intentionally a running log + checklist of “what you need to look at and do” while wiring the experiment into the repo, so we can identify documentation/architecture gaps.

---

## What we’re building

An A/B experiment runner that executes a **3‑prompt pipeline** for image prompt creation:

1) **Prompt 1 (draft)**: uses a **randomly generated token** as an input and generates a scene draft.
2) **Prompt 2 (refine)**: refines the draft using a single prompt.
   - **Variant A**: refinement prompt without the refinement block
   - **Variant B**: refinement prompt with an explicit refinement block (extra structured instructions)
3) **Prompt 3 (format)**: converts the refined draft into a final image prompt using a strict output format.

Execution should default to `run.mode: prompt_only` so new developers can run it without GPU/image dependencies.

---

## “Where do I even start?” pointers (repo landmarks)

- Prompt orchestration entrypoint: `image_project/app/generate.py`
- Prompt plans + stage modifiers overview: `docs/experiments.md`
- Stage catalog + prompt builders (source of truth for stages): `image_project/impl/current/prompting.py`
- Existing experiment harness pattern: `tools/run_experiment_3x3.py`
- PDM script registry (how commands are exposed): `pyproject.toml` `[tool.pdm.scripts]`

---

## Notes discovered while starting (gaps to fix / document)

- `config/config.yaml` currently points to `M:/My Drive/...` for `prompt.categories_path` and `prompt.profile_path`, which a new developer likely won’t have.
- `config/config.full_example.yaml` references sample CSVs under `image_project/impl/current/data/sample/`, but those files are missing in the repo right now.
  - A new developer can’t run *any* prompt pipeline without valid CSVs, because `generate.py` always loads both files.

---

## Implementation log (chronological)

### 2025-12-26

- Inspected experiment harness: `tools/run_experiment_3x3.py` (builds per-variant config overlays, writes `experiment_plan.json`, runs `run_generation`, writes `experiment_results.json`).
- Read prompt experimentation docs: `docs/experiments.md` (custom plans via `prompt.plan: custom` + `prompt.stages.sequence`).
- Confirmed prompt pipeline + seed wiring: `image_project/app/generate.py` (seed -> `random.Random(seed)` -> `PlanInputs.rng` + `RunContext.rng`).
- Confirmed sample-data mismatch: `config/config.full_example.yaml` points to sample CSVs that do not exist under `image_project/impl/current/data/sample/`.
- Added A/B experiment stages to the Stage Catalog in `image_project/impl/current/prompting.py`:
  - `ab.random_token` (action; captures `ab_random_token`)
  - `ab.scene_draft` (captures `ab_scene_draft`)
  - `ab.scene_refine_no_block` vs `ab.scene_refine_with_block` (both capture `ab_scene_refined`)
  - `ab.final_prompt_format` (default capture -> `image_prompt`)
- Added repo sample CSVs so a new developer can run prompt-only out of the box:
  - `image_project/impl/current/data/sample/category_list_v1.csv`
  - `image_project/impl/current/data/sample/user_profile_v4.csv`
- Updated default config to be cross-platform and runnable without personal paths: `config/config.yaml`
  - `run.mode: prompt_only`
  - `_artifacts/` output paths
  - sample data CSV paths
  - `rclone.enabled: false`, `upscale.enabled: false`, `context.enabled: false`
- Added an A/B experiment runner: `tools/run_experiment_ab_refinement_block.py`
  - Registered as `pdm run experiment-ab-refinement-block` in `pyproject.toml`
  - Documented in `docs/experiments.md`
- Verified the runner works:
  - Important: use `pdm run ...` (system `python` may not have dependencies like `pandas` installed).
  - Dry-run (no AI calls): `pdm run experiment-ab-refinement-block --dry-run --runs 2 --seed 123 --output-root ./_artifacts/experiments/_smoke_ab`
  - Prompt-only execution (AI calls): `pdm run experiment-ab-refinement-block --runs 1 --seed 123 --output-root ./_artifacts/experiments/_smoke_ab_run2`
  - Tests: `pdm run test` (all passed)

Next actions:
- Run the new runner in `--dry-run` and prompt-only modes to verify artifacts + stage wiring.
- Add “how to compare A vs B” instructions (run-review, paths to transcripts/final prompts).

---

## How a new developer runs the A/B experiment (happy path)

1) Ensure dependencies are installed (PDM):

- `pdm install`

2) Ensure you have an OpenAI key available to the runtime:

- Set `OPENAI_API_KEY` in your environment (required for non-`--dry-run`).

3) Run the experiment:

- Dry run (config + stage wiring only):
  - `pdm run experiment-ab-refinement-block --dry-run --runs 5`
- Actual prompt-only run (writes transcripts + final prompts; no images):
  - `pdm run experiment-ab-refinement-block --runs 5`
- Use your own data instead of repo samples:
  - `pdm run experiment-ab-refinement-block --data config --runs 5`

Artifacts land under:

- `./_artifacts/experiments/<timestamp>_ab_refinement_block/`
  - `experiment_plan.json`
  - `experiment_results.json`
  - `logs/<generation_id>_transcript.json`
  - `logs/<generation_id>_final_prompt.txt`
  - `logs/runs_index.jsonl`

## How to compare A vs B

- The experiment runner prints generation ids like `A3_<uuid>` and `B3_<uuid>`.
- Compare a run pair with run-review:
  - `pdm run run-review --compare <A_generation_id> <B_generation_id> --logs-dir <output_root>/logs`
