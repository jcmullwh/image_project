# New Developer Experiment Preparation - SceneSpec JSON Intermediary (A/B)

## Update (2026-01)

This developer diary predates the `pipelinekit` refactor. References to `image_project/impl/current/prompting.py` are historical; the stage catalog now lives under `image_project/stages/*` and is assembled by `image_project/stages/registry.py`.

## Counters (running)
- files_opened: 19
- searches: 10
- commands_run: 37
- errors_hit: 0
- files_changed: 4

## Change Surface Area (required vs optional)
- `new developer experiment preparation - SceneSpec JSON Intermediary.md` (required): developer diary + experiment spec per task request.
- `image_project/stages/ab/*` (required): adds SceneSpec JSON intermediary stages and a "format-from-SceneSpec" final stage for the new A/B experiment.
- `tools/run_experiment_ab_scenespec_json_intermediary.py` (required): new A/B experiment runner + automatic checks output to `experiment_results.json`.
- `pyproject.toml` (required): wires the new runner into `pdm run experiment-ab-scenespec-json-intermediary`.

## Developer Diary (append-only)

### Step 1
- Intent: Start the required append-only developer diary so I can log every subsequent action.
- Action taken: Created `new developer experiment preparation - SceneSpec JSON Intermediary.md`.
- Expected result: A place to append structured log entries and counters while I onboard.
- Observed result: File created with initial counters + diary scaffolding.
- Confusion / “wait what does this do?”: The diary is “append-only” but also asks for “running counters at the top”; I’m assuming “don’t rewrite earlier *diary entries*”, but I will update the counter block in-place.
- Decision / next step: Open the README first and try to follow the repo’s happy path for running/adding experiments.

### Step 2
- Intent: Find the primary docs entry point (README) and get oriented without searching the whole repo yet.
- Action taken: Ran `Get-ChildItem -Force` in repo root.
- Expected result: See a `README.md` or similar “start here” doc.
- Observed result: Found `README.md` (plus `tools/`, `image_project/`, `tests/`, `_artifacts/`, and a couple similarly-named experiment-prep markdown files).
- Confusion / “wait what does this do?”: None yet.
- Decision / next step: Open `README.md` and follow whatever it says for running experiments.

### Step 3
- Intent: Understand the repo’s “official” workflow for experiments before touching code.
- Action taken: Opened `README.md` via `Get-Content README.md -TotalCount 200`.
- Expected result: Find canonical entry points + how experiments are supposed to be configured.
- Observed result: README points to a step-driven prompt pipeline (`ChatStep`/`Block`), docs at `docs/experiments.md`/`docs/stages.md`, built-in plans in `image_project/impl/current/plans.py`, and mentions an existing A/B experiment: `pdm run experiment-ab-refinement-block`.
- Confusion / “wait what does this do?”: A bit unclear whether experiments are “plans” vs standalone scripts under `tools/` (both are mentioned).
- Decision / next step: Read `docs/experiments.md` next and look for the existing A/B refinement-block experiment implementation.

### Step 4
- Intent: Learn the supported “experiment knobs” (plans, stage overrides, experiment labels) before I implement anything.
- Action taken: Opened `docs/experiments.md` via `Get-Content docs/experiments.md -TotalCount 200`.
- Expected result: Clear guidance on how A/B experiments are intended to be done (config-only vs code changes).
- Observed result: Docs describe `prompt.plan`, stage include/exclude/overrides, refinement policy, and an optional `experiment: {id, variant, notes, tags}` label that flows into transcripts and run-review reports.
- Confusion / “wait what does this do?”: Still not seeing where the `pdm run experiment-ab-refinement-block` command maps to code (yet).
- Decision / next step: Look for the actual experiment runner entrypoints (likely under `tools/` and/or `pyproject.toml` scripts) without doing a full-text repo search yet.

### Step 5
- Intent: Find the concrete experiment runner scripts referenced by README.
- Action taken: Ran `Get-ChildItem tools -Force`.
- Expected result: See scripts named similarly to the README commands.
- Observed result: Found `tools/run_experiment_3x3.py` and `tools/run_experiment_ab_refinement_block.py`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open `tools/run_experiment_ab_refinement_block.py` first to see how the existing A/B is implemented, then mirror it for “SceneSpec JSON Intermediary”.

### Step 6
- Intent: Understand how the existing A/B refinement-block experiment is structured (so I can copy the pattern).
- Action taken: Opened `tools/run_experiment_ab_refinement_block.py` via `Get-Content tools/run_experiment_ab_refinement_block.py -TotalCount 240`.
- Expected result: See variant definitions + stage ids + how it writes artifacts.
- Observed result: The runner builds a “custom” plan with stage ids like `ab.random_token`, `ab.scene_draft`, and variant-specific refine stages, and writes `experiment_plan.json` / `experiment_results.json` under an `_artifacts/experiments/<timestamp>_ab_refinement_block/` folder.
- Confusion / “wait what does this do?”: I don’t yet know where stage ids like `ab.scene_draft` are defined (stage catalog doc didn’t mention them).
- Decision / next step: Keep reading the rest of the script to find CLI args and how it validates configs before running.

### Step 7
- Intent: Find the full CLI surface + validation behavior (especially `--dry-run` flow).
- Action taken: Opened more of `tools/run_experiment_ab_refinement_block.py` via `Get-Content tools/run_experiment_ab_refinement_block.py -TotalCount 520`.
- Expected result: See `argparse` config, validation loop, and `run_generation(...)` call site.
- Observed result: Script validates each planned run via `RunConfig.from_dict(...)`, writes a plan JSON, then runs `run_generation(...)` for each planned run, writing results JSON at the end.
- Confusion / “wait what does this do?”: My terminal output truncated the middle of the file, so I couldn’t see some details inline (I’ll need to re-open specific ranges or search within the file later).
- Decision / next step: Read `docs/stages.md` to see how/where stages are defined, then track down the `ab.*` stage definitions.

### Step 8
- Intent: Find the “single source of truth” for stage ids and prompt builders.
- Action taken: Opened `docs/stages.md` via `Get-Content docs/stages.md -TotalCount 220`.
- Expected result: A list including the `ab.*` stages used by the refinement-block experiment.
- Observed result: The doc only lists `preprompt.*`, `standard.*`, `blackbox.*`, and `refine.*` stages; no mention of `ab.*`.
- Confusion / “wait what does this do?”: Slight mismatch: README says stage catalog lives in `image_project/impl/current/prompting.py`, but the doc’s list seems incomplete/out-of-date relative to `tools/run_experiment_ab_refinement_block.py`.
- Decision / next step: Double-check the bottom of `docs/stages.md` (in case `ab.*` is listed later), then open `image_project/impl/current/prompting.py`.

### Step 9
- Intent: Confirm whether `docs/stages.md` has any additional sections beyond what I first saw.
- Action taken: Opened `docs/stages.md` tail via `Get-Content docs/stages.md -Tail 120`.
- Expected result: Maybe see additional stage groups (including `ab.*`).
- Observed result: No additional groups; doc repeats the same content and ends at `refine.image_prompt_refine`.
- Confusion / “wait what does this do?”: None beyond “docs seem stale vs code”.
- Decision / next step: Open `image_project/impl/current/prompting.py` and locate the `ab.*` stage wiring.

### Step 10
- Intent: Locate where stage ids are wired (especially the `ab.*` stages referenced by the existing experiment runner).
- Action taken: Opened `image_project/impl/current/prompting.py` via `Get-Content image_project/impl/current/prompting.py -TotalCount 220`.
- Expected result: Quickly find a “stage catalog” mapping stage ids (like `ab.scene_draft`) to prompt builders.
- Observed result: The top of the file contains prompt construction helpers like `generate_first_prompt()` and a large `generate_image_prompt()` template; I didn’t immediately see a stage catalog structure in the first ~200 lines.
- Confusion / “wait what does this do?”: The docs say this file is the stage catalog source of truth, but what I saw looks like prompt templates rather than stage wiring. Maybe the catalog is further down, or the docs are slightly out of date.
- Decision / next step: Use a targeted search within `image_project/impl/current/prompting.py` for `ab.` / `stage` to find the relevant section rather than scrolling manually.

### Step 11
- Intent: Find where the `ab.*` stages are defined without guessing filenames.
- Action taken: Searched `image_project/impl/current/prompting.py` for `ab\.` via `rg "ab\\." image_project/impl/current/prompting.py`.
- Expected result: Line hits showing where `StageCatalog.register("ab....")` appears.
- Observed result: Found entries for `ab.random_token`, `ab.scene_draft`, `ab.scene_refine_no_block`, `ab.scene_refine_with_block`, and `ab.final_prompt_format`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Jump to the `ab.random_token` section by line number.

### Step 12
- Intent: Get precise line numbers so I can view the surrounding code in manageable chunks.
- Action taken: Ran `rg -n "ab\\.random_token" image_project/impl/current/prompting.py`.
- Expected result: A concrete line number for the `ab.random_token` stage registration.
- Observed result: Found references around line ~1139+.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open the file around that line range to inspect the A/B stage implementations.

### Step 13
- Intent: Inspect the existing A/B stage prompts to copy the pattern for my new A/B experiment.
- Action taken: Opened a mid-file slice via `Get-Content image_project/impl/current/prompting.py | Select-Object -Skip 1100 -First 170`.
- Expected result: See `@StageCatalog.register(...)` blocks for `ab.random_token` and `ab.scene_draft`.
- Observed result: Confirmed `ab.random_token` is an action stage that writes `ab_random_token`, and `ab.scene_draft` is a chat stage that requires the token and outputs `ab_scene_draft`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Scroll further to see the refine stages and the final formatting stage.

### Step 14
- Intent: Understand the exact “one-line final prompt” format enforced by the current A/B runner.
- Action taken: Opened the next slice via `Get-Content image_project/impl/current/prompting.py | Select-Object -Skip 1270 -First 220`.
- Expected result: See how the refined scene is converted into the strict one-line prompt.
- Observed result: `ab.final_prompt_format` outputs one line in the form `SUBJECT=... | SETTING=... | ... | TEXT_IN_SCENE="{token}" | AR=16:9`, and it insists on exact token inclusion.
- Confusion / “wait what does this do?”: This experiment’s “final prompt format” is already close to what I need; my new work is mainly inserting an intermediate SceneSpec JSON path for variant B and adding validators/checks.
- Decision / next step: Open `tools/run_experiment_3x3.py` (active tab) to see if there’s already parsing/automatic checking logic I can reuse for validation and logging.

### Step 15
- Intent: See if the 3x3 runner already has any automatic “prompt quality” checks or parsing utilities I can reuse.
- Action taken: Opened `tools/run_experiment_3x3.py` via `Get-Content tools/run_experiment_3x3.py -TotalCount 260`.
- Expected result: Find any prompt parsing / validation routines.
- Observed result: The runner mostly mirrors the A/B structure (build plan via config overlays, validate with `RunConfig.from_dict`, run `run_generation`, write plan/results JSON). It also has helper code for concept sampling, but nothing about validating final prompt formats.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Check the bottom of the file for any post-processing/metrics before concluding there’s no validation.

### Step 16
- Intent: Confirm whether `tools/run_experiment_3x3.py` does any additional result parsing or checks after `run_generation`.
- Action taken: Opened `tools/run_experiment_3x3.py` tail via `Get-Content tools/run_experiment_3x3.py -Tail 200`.
- Expected result: Maybe see final prompt extraction, regex checks, or CSV augmentation.
- Observed result: No extra checks; it records a couple outputs (title + prompt pipeline) and errors, but doesn’t validate prompt structure.
- Confusion / “wait what does this do?”: So far it looks like experiment runners are intentionally thin and rely on run_review for human evaluation; I’ll need to add lightweight automatic checks myself if I want them.
- Decision / next step: Find the “run-review compare” flow mentioned in the task request, and decide whether to implement checks in (a) the runner script output JSON, or (b) a pipeline action stage that writes check results into the transcript.

### Step 17
- Intent: Locate the “run-review compare” implementation so the new experiment plugs into the existing human evaluation flow.
- Action taken: Listed app code via `Get-ChildItem image_project/app -Force`.
- Expected result: Find a `run_review` entrypoint or module in `image_project/app`.
- Observed result: Only `generate.py` and `__init__.py` exist in `image_project/app`; no run review code there.
- Confusion / “wait what does this do?”: I expected run review to be an “app” command because README mentions it alongside generate/transcripts.
- Decision / next step: List the package root (`image_project/`) to look for a `run_review/` module.

### Step 18
- Intent: Find where run-review lives in the package.
- Action taken: Ran `Get-ChildItem image_project -Force`.
- Expected result: See a `run_review` folder or similarly named module.
- Observed result: Found `image_project/run_review/` alongside `cli.py` and `__main__.py`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: List `image_project/run_review/` and open its CLI to understand `--compare` behavior.

### Step 19
- Intent: Discover the run-review CLI surface (especially compare flow) and expected input artifacts.
- Action taken: Ran `Get-ChildItem image_project/run_review -Force`.
- Expected result: Find `cli.py` and a `compare` module.
- Observed result: Found `cli.py`, `compare.py`, and report/render helpers (parse transcript/oplog, render HTML, etc.).
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open `image_project/run_review/cli.py` and `image_project/run_review/compare.py` to see how A/B comparisons are driven and what metadata it expects.

### Step 20
- Intent: Understand how run_review discovers runs and how compare works (so the new experiment’s artifacts will slot in).
- Action taken: Opened `image_project/run_review/cli.py` via `Get-Content image_project/run_review/cli.py -TotalCount 260`.
- Expected result: See CLI flags/subcommands for generating reports and doing comparisons.
- Observed result: The file sets up run discovery from `image.log_path`, loads transcripts/oplogs, builds reports, and has render helpers like `render_compare_html`. The output I saw was truncated before the compare flags/subcommands.
- Confusion / “wait what does this do?”: The CLI output got truncated again, so I can’t see how “compare” is invoked without doing a targeted search within the file.
- Decision / next step: Search within `image_project/run_review/cli.py` for `compare`/`--compare` and then open the relevant section.

### Step 21
- Intent: Find exactly how `run_review` does comparisons (flags + expected inputs).
- Action taken: Searched `image_project/run_review/cli.py` for `compare` via `rg -n "compare" image_project/run_review/cli.py`.
- Expected result: See where `--compare` is parsed and what files it reads.
- Observed result: Found `--compare` argument (takes two generation ids) and code that writes `<id1>_vs_<id2>_run_compare.html/json`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open the section around the `--compare` handling to confirm what it diffs (transcript steps, metadata, etc.).

### Step 22
- Intent: Confirm the run_review compare flow so I can describe the human evaluation “happy path” for this experiment.
- Action taken: Opened `image_project/run_review/cli.py` around `--compare` via `Get-Content image_project/run_review/cli.py | Select-Object -Skip 250 -First 140`.
- Expected result: Compare flow should load two runs by generation id and output an HTML report.
- Observed result: `--compare <base_id> <other_id>` resolves both runs’ transcript/oplog under `--logs-dir` (defaulted from `config/config.yaml`), builds reports, diffs them, and writes both HTML and JSON diff artifacts.
- Confusion / “wait what does this do?”: It compares *two arbitrary generation_ids*; there’s no built-in “pair by experiment run_index”, so the experiment runner should make it easy to identify A/B pairs (e.g., `A1_...` vs `B1_...` IDs like the existing A/B script does).
- Decision / next step: Implement the new experiment runner in `tools/` following the existing A/B runner pattern (generation ids prefixed with A1/B1) and add new `ab.*` (or new namespace) stages for SceneSpec JSON + validation.

### Step 23
- Intent: Figure out how stage outputs are captured/stored so I can attach validators after the final prompt stage.
- Action taken: Searched for `output_key` usage via `rg -n "output_key" image_project/framework image_project/impl/current/prompting.py | Select-Object -First 40`.
- Expected result: Find the core stage spec types and pipeline capture behavior.
- Observed result: Located `image_project/framework/prompting.py` which defines `StageSpec` / `ActionStageSpec`, and saw logic referencing `capture_key="image_prompt"` and “capture stage output_key conflict”.
- Confusion / “wait what does this do?”: I wasn’t sure whether the final prompt is accessible as `ctx.outputs["image_prompt"]` during later stages or only after the run ends.
- Decision / next step: Open `image_project/framework/prompting.py` around the pipeline-building code to confirm capture timing and whether a post-check action stage can read the final prompt.

### Step 24
- Intent: Understand StageSpec/ActionStageSpec semantics (especially output capture).
- Action taken: Opened `image_project/framework/prompting.py` via `Get-Content image_project/framework/prompting.py -TotalCount 220`.
- Expected result: See how `output_key` is validated and how capture keys might work.
- Observed result: `StageSpec` and `ActionStageSpec` both support an optional `output_key`. There’s a `build_pipeline_block(...)` function later that likely controls capture behavior.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Jump to the `build_pipeline_block(...)` section where capture keys are assigned.

### Step 25
- Intent: Confirm whether a stage sequence like `[final_prompt_stage, validate_stage]` can have the validate stage read `ctx.outputs["image_prompt"]`.
- Action taken: Opened `image_project/framework/prompting.py` mid-file via `Get-Content image_project/framework/prompting.py | Select-Object -Skip 280 -First 180`.
- Expected result: See capture key assignment rules.
- Observed result: If a stage is the chosen `capture_stage`, its output is captured under `capture_key` (default `image_prompt`), and subsequent stages can access it via `ctx.outputs["image_prompt"]`. Non-capture stages only write outputs if they set `output_key`.
- Confusion / “wait what does this do?”: Important gotcha: a capture stage cannot set `output_key` to a different key (it errors), so any “also store under another name” needs a separate follow-on stage.
- Decision / next step: Implement the SceneSpec JSON path as a non-capture stage (`output_key="ab_scene_spec_json"`) and keep the final one-line stage as the capture stage, then add a post-validation action stage that reads `ctx.outputs["image_prompt"]` and writes check results to `ctx.outputs["ab_prompt_checks"]`.

### Step 26
- Intent: See whether run_review reports have a natural place to surface “automatic check” outputs (so I don’t invent a parallel reporting format unnecessarily).
- Action taken: Opened `image_project/run_review/report_model.py` via `Get-Content image_project/run_review/report_model.py -TotalCount 240`.
- Expected result: Find fields for “prompt validation results” or a generic “extras/outputs” bucket.
- Observed result: Run metadata captures high-level things like `experiment`, `prompt_pipeline`, and `title_generation`, but there’s no explicit slot for arbitrary pipeline outputs like “ab_prompt_checks”. Steps do include prompt/response per stage though.
- Confusion / “wait what does this do?”: Not clear whether run_review renders *all* step responses (including action-stage responses) in a way that’s easy to scan.
- Decision / next step: For minimal implementation, record automatic check results in the experiment runner’s `experiment_results.json`. Optionally later, add a pipeline validation stage if we want the checks to show up directly in run_review HTML.

### Step 27
- Intent: Avoid reinventing a “cliche phrase list” if the repo already has one.
- Action taken: Searched for “clich” across code/docs via `rg -n "clich" -S image_project tools docs`.
- Expected result: Find a canonical list of clichés to flag.
- Observed result: Only found scattered mentions of “avoid cliches” in prompt text; no reusable list.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Implement a small, explicit cliché phrase list inside the new experiment runner for the requested automatic check.

### Step 28
- Intent: Figure out how existing experiment runners are wired into `pdm run ...` commands so I can add the new one.
- Action taken: Opened `pyproject.toml` via `Get-Content pyproject.toml`.
- Expected result: See an entry under `[tool.pdm.scripts]` for the A/B refinement-block and 3x3 experiments.
- Observed result: Confirmed `experiment-3x3` and `experiment-ab-refinement-block` are defined as `python tools/<script>.py`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Add a new script entry (e.g., `experiment-ab-scenespec-json-intermediary`) and create a corresponding `tools/run_experiment_ab_scenespec_json_intermediary.py` runner modeled after the existing A/B script.

### Step 29
- Intent: Copy the existing A/B experiment runner pattern as closely as possible to minimize code surface area.
- Action taken: Searched `tools/run_experiment_ab_refinement_block.py` for argparse wiring via `rg -n "parser\\.add_argument" tools/run_experiment_ab_refinement_block.py | Select-Object -First 40`.
- Expected result: Locate where CLI flags are defined so I can keep the new runner’s flags consistent.
- Observed result: Found the argument block starting around line ~248.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open the argument parsing section to mirror it.

### Step 30
- Intent: See the exact CLI flags and defaults of the existing A/B runner.
- Action taken: Opened `tools/run_experiment_ab_refinement_block.py` around its `main()` via `Get-Content tools/run_experiment_ab_refinement_block.py | Select-Object -Skip 220 -First 150`.
- Expected result: A clean set of flags like `--runs`, `--mode`, `--seed`, `--dry-run`, etc.
- Observed result: The runner supports `--runs`, `--mode (prompt_only|full)`, `--data (sample|config)`, and writes a plan JSON before running.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open the results-writing section to see what it records (so I can extend it with automatic checks).

### Step 31
- Intent: Understand the current results JSON schema so the new experiment output feels consistent.
- Action taken: Opened the end of `tools/run_experiment_ab_refinement_block.py` via `Get-Content tools/run_experiment_ab_refinement_block.py | Select-Object -Skip 340 -First 140`.
- Expected result: A per-run results list that captures generation id, prompt, image path, and errors.
- Observed result: On success it stores `final_prompt` (from `ctx.outputs["image_prompt"]`) and `prompt_pipeline`; on failure it stores exception type/message.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Add new `ab.scene_spec_json` + `ab.final_prompt_format_from_scenespec` stages to the stage catalog and then implement a new runner that computes/checks the requested automatic metrics.

### Step 32
- Intent: Add the SceneSpec JSON intermediary path into the existing stage catalog so experiments can reference it by stage id.
- Action taken: Edited `image_project/impl/current/prompting.py` (added stages `ab.scene_spec_json` and `ab.final_prompt_format_from_scenespec`).
- Expected result: New stage ids resolve during config validation and can be used in a custom stage sequence.
- Observed result: Patch applied cleanly (no immediate errors yet).
- Confusion / “wait what does this do?”: I’m slightly worried about “stage id sprawl” because `docs/stages.md` doesn’t list `ab.*` stages at all, so discoverability relies on `list-stages` output or reading `prompting.py`.
- Decision / next step: Create `tools/run_experiment_ab_scenespec_json_intermediary.py` modeled after the existing A/B runner, then wire it into `[tool.pdm.scripts]` and run it in `--dry-run` mode to validate end-to-end.

### Step 33
- Intent: Implement the “SceneSpec JSON Intermediary” A/B experiment runner with the same ergonomics as existing runners.
- Action taken: Added `tools/run_experiment_ab_scenespec_json_intermediary.py`.
- Expected result: A script that (1) builds an A/B plan (A=prose refine, B=SceneSpec JSON), (2) runs the plan, (3) writes plan/results JSON, and (4) computes the requested automatic checks.
- Observed result: File added (modeled closely on `tools/run_experiment_ab_refinement_block.py`) and includes basic automatic check parsing for final prompt format + token correctness + cliché hits + per-field word counts, plus JSON parsing for the SceneSpec intermediary in variant B.
- Confusion / “wait what does this do?”: I’m not sure whether “cliché phrase hits” should be checked against the *final prompt*, the *draft/refined prose*, the *SceneSpec JSON*, or all three. I implemented it against the final prompt only (minimal/consistent).
- Decision / next step: Wire the runner into `pdm run ...` in `pyproject.toml` and run `--dry-run` to validate stage ids + config.

### Step 34
- Intent: Make the new experiment runnable via the repo’s existing “happy path” (`pdm run ...`) instead of requiring a raw `python tools/...` invocation.
- Action taken: Edited `pyproject.toml` to add `experiment-ab-scenespec-json-intermediary = "python tools/run_experiment_ab_scenespec_json_intermediary.py"`.
- Expected result: `pdm run experiment-ab-scenespec-json-intermediary --dry-run` works the same way as other experiment scripts.
- Observed result: Patch applied cleanly.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Run the new experiment in `--dry-run` mode to validate configs and ensure the new stage ids resolve (`ab.scene_spec_json`, `ab.final_prompt_format_from_scenespec`).

### Step 35
- Intent: Validate the new experiment runner’s end-to-end wiring (CLI, config overlays, artifact paths) without calling any AI.
- Action taken: Ran `pdm run experiment-ab-scenespec-json-intermediary --dry-run --runs 1`.
- Expected result: A printed plan showing A1/B1 stage sequences and a written `experiment_plan.json`.
- Observed result: Success; it printed A1/B1 with stage sequences including `ab.scene_spec_json` and `ab.final_prompt_format_from_scenespec`, and wrote `_artifacts/experiments/<timestamp>_ab_scenespec_json_intermediary/experiment_plan.json`. It also emitted this warning:
  - `RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work`
- Confusion / “wait what does this do?”: The warning is surprising for a prompt-only dry run; it suggests an import side-effect (pydub/ffmpeg detection) happens even when we aren’t processing audio/video.
- Decision / next step: Confirm the new stage ids are actually registered in the stage catalog via `pdm run list-stages`.

### Step 36
- Intent: Ensure the new `ab.*` stage ids are discoverable and actually registered (not just referenced in config).
- Action taken: Ran `pdm run list-stages | rg "ab\\.scene_spec_json|ab\\.final_prompt_format_from_scenespec"`.
- Expected result: Both stage ids appear in the printed catalog output.
- Observed result: Both stage ids were listed with their docstrings.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Write up the actual experiment spec in this markdown file (prompts, schema, checks, and how to run + how to do human compare in run_review), then run `pytest` for a quick sanity check.

### Step 37
- Intent: Make sure the repo still passes tests after adding new stages + a new tools runner.
- Action taken: Ran `pdm run test`.
- Expected result: Tests pass; no unexpected failures from the new stage ids or script wiring.
- Observed result: `140 passed` with a single warning:
  - `RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work`
- Confusion / “wait what does this do?”: The ffmpeg warning shows up in tests too; I’m not touching upscaling/audio, so it feels like a noisy global import side effect.
- Decision / next step: Append the experiment spec + postmortem into this file, then capture `git diff --stat` for the “Change Surface Area” section.

### Step 38
- Intent: Write down the experiment design and how to run/review it (so the next person doesn’t have to reverse-engineer stage ids from code).
- Action taken: Appended an “Experiment Spec” + “Postmortem” section to `new developer experiment preparation - SceneSpec JSON Intermediary.md`.
- Expected result: A single place that explains variants A/B, prompts/stages, automatic checks, and the run_review compare workflow.
- Observed result: Sections added below.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Run `git diff --stat` and paste it at the end per protocol.

## Experiment Spec — “SceneSpec JSON Intermediary” (A/B)

### Hypothesis
Forcing a structured intermediate representation (SceneSpec JSON) will increase final-prompt correctness and specificity for a strict one-line prompt format, without harming coherence.

### Controlled variable
Both variants share a deterministic per-run random token (derived from `prompt.random_seed`) that must appear as visible text in-scene and must match `TEXT_IN_SCENE` exactly.

### Variants (implemented)
Common prefix stages (both A and B):
- `ab.random_token` (action): derives token `RV-<seed>-<roll>` and writes `ctx.outputs["ab_random_token"]`.
- `ab.scene_draft` (chat): generates 4–6 sentences from the token and requires the token appear verbatim as visible text. Writes `ctx.outputs["ab_scene_draft"]`.

Variant A (prose refine, like current refinement-block approach):
- `ab.scene_refine_with_block` (chat): refines the scene draft using an explicit checklist and writes `ctx.outputs["ab_scene_refined"]`.
- `ab.final_prompt_format` (chat, capture stage): converts refined prose into a strict one-line prompt captured as `ctx.outputs["image_prompt"]`.

Variant B (structured spec):
- `ab.scene_spec_json` (chat): converts the draft into strict SceneSpec JSON and writes `ctx.outputs["ab_scene_spec_json"]`.
- `ab.final_prompt_format_from_scenespec` (chat, capture stage): converts SceneSpec JSON into the strict one-line prompt captured as `ctx.outputs["image_prompt"]`.

### Prompt 1 (draft)
Implemented by stage `ab.scene_draft`:
- Input: `Required token: RV-...`
- Output: 4–6 sentence scene description (no headings/bullets)
- Constraint: token must appear verbatim as visible in-scene text

### Prompt 2 (variant step)
Variant A: stage `ab.scene_refine_with_block` (prose refinement checklist)

Variant B: stage `ab.scene_spec_json` (SceneSpec JSON intermediary)

SceneSpec JSON schema (required keys):
```json
{
  "subject": "string (non-generic, non-empty)",
  "setting": "string (non-empty)",
  "action": "string (non-empty)",
  "composition": "string (non-empty)",
  "camera": "string (non-empty)",
  "lighting": "string (non-empty)",
  "color": "string (non-empty)",
  "style": "string (non-empty)",
  "text_in_scene": "MUST exactly equal the token",
  "must_keep": ["at least 3 non-empty strings"],
  "avoid": ["at least 3 non-empty strings"]
}
```

### Prompt 3 (format)
Variant A: stage `ab.final_prompt_format` generates the final strict one-liner from refined prose.

Variant B: stage `ab.final_prompt_format_from_scenespec` generates the final strict one-liner from SceneSpec JSON.

Final one-line format (required, exact separators):
```
SUBJECT=<...> | SETTING=<...> | ACTION=<...> | COMPOSITION=<...> | CAMERA=<...> | LIGHTING=<...> | COLOR=<...> | STYLE=<...> | TEXT_IN_SCENE="RV-..." | AR=16:9
```

### Automatic checks (implemented in the runner)
Computed per-run and stored in `<output_root>/experiment_results.json` under `auto_checks`.

Checks against final one-line prompt (A and B):
- Format violations: multiline output, wrong segment count, missing keys, wrong key order, missing `=` separators, placeholder `<...>`, missing quotes around `TEXT_IN_SCENE`, `AR!=16:9`.
- Token correctness: `TEXT_IN_SCENE` extracted value must exactly equal the expected token.
- Per-field specificity: minimum word counts per field (defaults in runner):
  - `SUBJECT>=5`, `SETTING>=6`, `ACTION>=5`, `COMPOSITION>=5`, `CAMERA>=3`, `LIGHTING>=3`, `COLOR>=3`, `STYLE>=3`
- Cliché phrase hits: substring hits against a small hardcoded list (currently checked against the final one-line prompt).

Checks against SceneSpec JSON (B only):
- JSON parse validity and “object shape” checks.
- Required keys present; type checks for strings vs list fields.
- Empty-field checks (including `must_keep`/`avoid` having <3 items).
- Subject “genericness” heuristic (flags subjects like “a person”, “someone”, etc.).
- `text_in_scene` must exactly equal the expected token.

Cross-step token presence (A and B):
- Whether the token appears in `ab_scene_draft`, `ab_scene_refined`, and raw `ab_scene_spec_json` text (helpful when diagnosing failures).

### Human check (existing run_review compare flow)
Goal: rate A/B pairs for “coherence + vividness + shootability”.

Workflow:
1. Run the experiment (prompt-only recommended for speed).
2. Use the printed plan (or `experiment_plan.json`) to pair `A1_...` with `B1_...`, `A2_...` with `B2_...`, etc.
3. For each pair, run `run_review --compare` and review the HTML diff.

### How to run (commands)
Dry-run (validates config only; no model calls):
- `pdm run experiment-ab-scenespec-json-intermediary --dry-run --runs 3`

Prompt-only run (produces transcripts + results JSON):
- `pdm run experiment-ab-scenespec-json-intermediary --runs 5 --mode prompt_only`

Full run (generates images; slower):
- `pdm run experiment-ab-scenespec-json-intermediary --runs 5 --mode full`

Compare A/B for a given run index (example):
- `pdm run run-review --logs-dir <output_root>/logs --compare <A1_generation_id> <B1_generation_id> --output-dir <output_root>/review`

## Postmortem (onboarding friction)

### Top 10 friction points (with evidence)
1. `docs/stages.md` appears stale/incomplete: it doesn’t mention any `ab.*` stages even though experiment runners use them (Steps 8–9, 13–14).
2. Stage catalog “source of truth” is technically correct (`image_project/impl/current/prompting.py`), but the stage registry is far down in a large file mixed with other prompt helpers (Step 10, 13–14).
3. Terminal output truncation made it hard to read medium-sized files end-to-end (Step 7, Step 20).
4. Not obvious where “run-review compare flow” lives; I had to browse `image_project/` to find `run_review/` (Steps 17–19).
5. `run_review` compares *two arbitrary generation ids*; there’s no built-in “pair A/B by run_index” helper, so the experiment runner must enforce pairable ids (Step 22).
6. It’s unclear where automatic checks “should live” (runner JSON vs pipeline stage outputs vs run_review issues model) without reading quite a bit of code (Steps 23–26).
7. The `--dry-run` pattern validates config shape via `RunConfig.from_dict`, but doesn’t inherently validate that stage ids exist (I had to separately run `list-stages`) (Steps 35–36).
8. Unexpected ffmpeg/pydub runtime warning shows up during prompt-only dry-run and during tests (Steps 35, 37).
9. Repo has multiple experiment runner patterns (plans vs tools scripts) and similarly named markdown prep files; it’s easy to pick the wrong entry point (Steps 3–6, 28).
10. No canonical “cliché phrase list” exists even though prompts talk about clichés, so implementing “cliché hits” required inventing a list (Step 27, Step 33).

### What a new dev needed to learn that wasn’t obvious
- Custom experiments are typically implemented as `tools/run_experiment_*.py` scripts that build config overlays and invoke `run_generation(...)`.
- Stage ids referenced by experiments are registered via `@StageCatalog.register(...)` inside `image_project/impl/current/prompting.py` (not documented in `docs/stages.md`).
- Capture stage outputs are stored under `ctx.outputs["image_prompt"]`; non-capture stages must set `output_key` to persist outputs (Step 25).
- `run_review` is a separate CLI (`pdm run run-review`) and supports `--compare <id1> <id2>` for diffing two runs (Steps 20–22).

### What docs/path would have prevented each friction point
- Update `docs/stages.md` to include the `ab.*` group (or auto-generate it from `list-stages`).
- Add a short “Where are stages defined?” section to `docs/experiments.md` with pointers to `StageCatalog.register` and how `prompt.plan: custom` resolves stage ids.
- Add a “How to run A/B comparisons” snippet that explicitly shows `A1_...` vs `B1_...` naming + `run_review --compare` usage.
- Document where to put automatic checks (runner vs pipeline vs run_review), with a recommended minimal approach and an “ideal” approach.
- Add a note about the ffmpeg/pydub warning and how to silence it (or defer that import to only when needed).

### Step 39
- Intent: Capture the required `git diff --stat` output for the “Change Surface Area” protocol section.
- Action taken: Ran `git diff --stat`.
- Expected result: A concise summary of changed files and insertions/deletions.
- Observed result:
  ```text
  warning: in the working copy of 'image_project/impl/current/prompting.py', LF will be replaced by CRLF the next time Git touches it
  warning: in the working copy of 'pyproject.toml', LF will be replaced by CRLF the next time Git touches it
   image_project/impl/current/prompting.py | 93 +++++++++++++++++++++++++++++++++
   pyproject.toml                          |  1 +
   2 files changed, 94 insertions(+)
  ```
- Confusion / “wait what does this do?”: `git diff --stat` doesn’t include untracked new files, so it doesn’t list `tools/run_experiment_ab_scenespec_json_intermediary.py` or this new diary file unless they’re added to git.
- Decision / next step: Task requirements satisfied; hand off.

### Step 40
- Intent: Capture exact line numbers for the new stage ids so I can reference them cleanly in the final handoff message.
- Action taken: Ran `rg -n "ab\\.scene_spec_json" image_project/impl/current/prompting.py`.
- Expected result: Line hits showing where the new stage id is registered.
- Observed result: Found matches at lines `1282` and `1331`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Do the same for `ab.final_prompt_format_from_scenespec`.

### Step 41
- Intent: Capture exact line numbers for the new “format-from-SceneSpec” stage id.
- Action taken: Ran `rg -n "ab\\.final_prompt_format_from_scenespec" image_project/impl/current/prompting.py`.
- Expected result: Line hits showing where the new stage id is registered.
- Observed result: Found matches at lines `1374` and `1403`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Capture the `pyproject.toml` script entry line for the new `pdm run ...` command.

### Step 42
- Intent: Capture exact line number of the new `pdm` script entry for the final handoff.
- Action taken: Ran `rg -n "experiment-ab-scenespec-json-intermediary" pyproject.toml`.
- Expected result: The new script line appears under `[tool.pdm.scripts]`.
- Observed result: Found the entry at line `39`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Hand off to user.
