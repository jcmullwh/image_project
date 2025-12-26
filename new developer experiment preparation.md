# New Developer Experiment Preparation (A/B prompt refinement block)

Goal: help a new developer successfully set up and run an A/B experiment that tests whether adding a **refinement block** to the middle prompt improves the final image prompt output.

---

## Running Counters

- files_opened: 15
- searches: 5
- commands_run: 35
- errors_hit: 1
- files_changed: 11

## Change Surface Area (running)

- `README.md` — required; primary happy-path doc now mentions the A/B runner command.
- `config/config.yaml` — required; default `run.mode: prompt_only` + repo-relative sample CSV paths so a new dev can run without personal paths.
- `docs/experiments.md` — required; documents prompt plans + includes A/B runner section (and now a co-located compare snippet).
- `image_project/impl/current/prompting.py` — required; Stage Catalog entries for `ab.*` pipeline stages.
- `pyproject.toml` — required; registers `pdm run experiment-ab-refinement-block`.
- `tools/run_experiment_ab_refinement_block.py` — required; A/B harness runner.
- `image_project/impl/current/data/sample/category_list_v1.csv` — required; sample categories so prompt-only can run out of the box.
- `image_project/impl/current/data/sample/user_profile_v4.csv` — required; sample profile so prompt-only can run out of the box.
- `new developer experiment preparation.md` — required; append-only onboarding diary + checklist per task instructions.
- `tools/run_experiment_3x3.py` — optional (related); referenced by docs/README as an existing harness pattern.
- `new_developer_experiment_preparation.md` — optional (diagnostic); older diary file that also exists in the repo root (potential naming confusion).

---

## Developer Diary (append-only)

### Step 1

- Intent: Start an append-only onboarding diary before touching the repo.
- Action taken: Created `new developer experiment preparation.md`.
- Expected result: A place to log every file open/search/command/edit with minimal friction.
- Observed result: File created with counters + diary scaffolding.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open the README and try the repo’s “happy path” first.

### Step 2

- Intent: Find the repo’s primary docs entrypoint (README) without assuming structure.
- Action taken: Ran `Get-ChildItem` in repo root.
- Expected result: See `README.md` (or similar) and any obvious “getting started” docs.
- Observed result: Found `README.md`, plus `docs/`, `tools/`, and an existing `new_developer_experiment_preparation.md` (underscored) file.
- Confusion / “wait what does this do?”: Slight confusion why there’s both `agents.md` (empty) and “AGENTS.md instructions” in the prompt; not sure which is authoritative for repo behavior.
- Decision / next step: Open `README.md` and attempt the documented setup/run flow.

### Step 3

- Intent: Follow the “happy path” documentation and see if the A/B experiment already exists.
- Action taken: Opened `README.md` via `Get-Content README.md -TotalCount 200`.
- Expected result: A clear quickstart and pointers for prompt pipelines and experiment runners.
- Observed result: README explicitly lists `pdm run experiment-ab-refinement-block` as a supported command, and claims default `run.mode: prompt_only` + artifacts under `./_artifacts/`.
- Confusion / “wait what does this do?”: The README implies the A/B experiment already exists; not sure if it’s fully wired (stages + plan + runner) or just documented.
- Decision / next step: Try running the experiment in `--dry-run` mode as the lowest-risk validation.

### Step 4

- Intent: Validate the A/B experiment runner wiring without making AI calls.
- Action taken: Ran `pdm run experiment-ab-refinement-block --dry-run --runs 1 --seed 123 --output-root ./_artifacts/experiments/_smoke_ab_refinement_block`.
- Expected result: It should write an `experiment_plan.json` and `experiment_results.json` under the output root and exit quickly.
- Observed result: The command did not finish before the tool timeout. Verbatim output:
  - `command timed out after 13042 milliseconds`
  - `C:\\Users\\jason\\projects\\code\\image_project\\.venv\\Lib\\site-packages\\pydub\\utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work`
  - `  warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)`
- Confusion / “wait what does this do?”: Not sure why a prompt-only dry-run path imports `pydub`/ffmpeg; also unclear if the run was actually still progressing when the tool timed out.
- Decision / next step: Re-run the exact command with a longer timeout to see the real failure/success mode.

### Step 5

- Intent: Confirm whether the runner completes successfully when not cut off by the tool timeout.
- Action taken: Re-ran `pdm run experiment-ab-refinement-block --dry-run --runs 1 --seed 123 --output-root ./_artifacts/experiments/_smoke_ab_refinement_block` with a longer timeout.
- Expected result: The command completes and writes the experiment plan/results artifacts.
- Observed result: Command succeeded and printed two variants with stage sequences:
  - A1 stages: `['ab.random_token', 'ab.scene_draft', 'ab.scene_refine_no_block', 'ab.final_prompt_format']`
  - B1 stages: `['ab.random_token', 'ab.scene_draft', 'ab.scene_refine_with_block', 'ab.final_prompt_format']`
  - It wrote `./_artifacts/experiments/_smoke_ab_refinement_block/experiment_plan.json`.
- Confusion / “wait what does this do?”: Still seeing the `pydub` ffmpeg warning even on dry-run; not sure if it’s safe to ignore for prompt-only usage.
- Decision / next step: Inspect the output folder contents to see what dry-run writes (plan only vs plan+results).

### Step 6

- Intent: Verify what artifacts are produced in `--dry-run`.
- Action taken: Ran `Get-ChildItem .\\_artifacts\\experiments\\_smoke_ab_refinement_block -Recurse`.
- Expected result: At least `experiment_plan.json`; possibly a placeholder results file.
- Observed result: Only `experiment_plan.json` exists in the folder.
- Confusion / “wait what does this do?”: Slight uncertainty whether dry-run should also write `experiment_results.json` (helpful to validate parsing), but plan-only is still useful.
- Decision / next step: Open `experiment_plan.json` to see if it captures the A/B variant config clearly (ids, stages, captures).

### Step 7

- Intent: Confirm the plan file is informative enough for comparing A vs B.
- Action taken: Opened `._artifacts/experiments/_smoke_ab_refinement_block/experiment_plan.json` via `Get-Content`.
- Expected result: Experiment metadata, run-mode, variant labeling, and per-run ids/seeds/tokens.
- Observed result: Plan includes `experiment_id`, `base_seed`, config path (`config/config.yaml`), `run_mode: prompt_only`, `data_mode: sample`, and `planned_runs` with matching `random_token` across A/B.
- Confusion / “wait what does this do?”: None — the plan format is clearer than I expected.
- Decision / next step: Run a non-dry-run prompt-only experiment (`--runs 1`) to confirm it writes transcripts + final prompts without needing any image/GPU setup.

### Step 8

- Intent: Execute the actual A/B experiment end-to-end in prompt-only mode and confirm artifacts are written for both variants.
- Action taken: Ran `pdm run experiment-ab-refinement-block --runs 1 --seed 123 --output-root ./_artifacts/experiments/_smoke_ab_refinement_block_run`.
- Expected result: Two runs (A and B) with transcripts + final prompt text files under `<output_root>/logs/`, plus `experiment_plan.json` and `experiment_results.json` at the root.
- Observed result: Success for both variants. Logs show:
  - Config loaded from `config/config.yaml` and data loaded from `image_project/impl/current/data/sample/*.csv`.
  - `run.mode=prompt_only; skipping media pipeline (image/upscale/upload/csv)`.
  - Wrote `<generation_id>_final_prompt.txt`, `<generation_id>_transcript.json`, `<generation_id>_oplog.log`, and appended `runs_index.jsonl`.
  - Wrote `experiment_plan.json` and `experiment_results.json` to the output root.
- Confusion / “wait what does this do?”: Still seeing `pydub`’s “couldn’t find ffmpeg” warning even though no audio/media is involved; it’s noisy and makes me worry there’s a hidden dependency.
- Decision / next step: Inspect `experiment_results.json` + one pair of final prompts to see how A vs B is represented and how to compare them.

### Step 9

- Intent: Confirm artifact filenames/locations so comparison instructions can be concrete.
- Action taken: Ran `Get-ChildItem .\\_artifacts\\experiments\\_smoke_ab_refinement_block_run -Recurse`.
- Expected result: `experiment_plan.json`, `experiment_results.json`, and a `logs/` folder with per-generation artifacts.
- Observed result: Output root contains `experiment_plan.json`, `experiment_results.json`, and `logs/` with:
  - `<generation_id>_final_prompt.txt`
  - `<generation_id>_transcript.json`
  - `<generation_id>_oplog.log`
  - `runs_index.jsonl`
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open `experiment_results.json` to see how it summarizes the A/B pair.

### Step 10

- Intent: See if the results artifact is sufficient to compare A vs B without digging into transcripts.
- Action taken: Opened `._artifacts/experiments/_smoke_ab_refinement_block_run/experiment_results.json` via `Get-Content`.
- Expected result: For each run: `{variant, generation_id, status, final_prompt}` and enough metadata to link back to logs.
- Observed result: `experiment_results.json` includes two entries (A and B) with:
  - Matching `random_token` across variants.
  - `final_prompt` inline (already in the strict `KEY=... | KEY=...` format).
  - `outputs.prompt_pipeline.resolved_stages` showing which refine stage was used.
- Confusion / “wait what does this do?”: The final prompts contain some odd mojibake like `workerƒ?Ts` instead of an apostrophe; not sure if that’s a model output artifact or a transcription/encoding issue.
- Decision / next step: Open the two `*_final_prompt.txt` files directly to see if they match the JSON and to sanity-check the strict formatting.

### Step 11

- Intent: Sanity-check the strict final prompt format for Variant A (no refinement block).
- Action taken: Opened `._artifacts/experiments/_smoke_ab_refinement_block_run/logs/A1_20251226_075215_e97b597f-02d8-4e01-a0bb-62735c316669_final_prompt.txt` via `Get-Content`.
- Expected result: A single line of `KEY=value | KEY=value ...` with the random token embedded.
- Observed result: Format is correct and includes `TEXT_IN_SCENE="RV-123-154907"`, but contains odd characters like `workerƒ?Ts`.
- Confusion / “wait what does this do?”: Not sure if that mojibake is a model output artifact or indicates some encoding issue in the prompt/context being sent.
- Decision / next step: Compare with Variant B’s final prompt to see if the refinement block reduces these artifacts.

### Step 12

- Intent: Sanity-check the strict final prompt format for Variant B (with refinement block).
- Action taken: Opened `._artifacts/experiments/_smoke_ab_refinement_block_run/logs/B1_20251226_075215_1bb3ea33-7b25-4cf7-96ff-344e4cedce50_final_prompt.txt` via `Get-Content`.
- Expected result: Same strict format; token embedded; ideally higher clarity/consistency.
- Observed result: Format is correct, includes `TEXT_IN_SCENE="RV-123-154907"`, and does not show the mojibake seen in Variant A.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Now that the runner works, open the runner + stage catalog code to confirm the experiment matches the required 3-prompt pipeline and see where the refinement block lives.

### Step 13

- Intent: Find documentation describing how the experiment is configured (plan/stage wiring) and how to compare outputs.
- Action taken: Opened `docs/experiments.md` via `Get-Content .\\docs\\experiments.md -TotalCount 220`.
- Expected result: An explicit section for the A/B refinement-block experiment and a “how to compare” snippet.
- Observed result: The doc explains `run.mode: prompt_only`, `prompt.plan: custom`, stage include/exclude/overrides, and where prompt pipeline metadata is recorded; did not yet see the specific A/B refinement-block section in the first 220 lines.
- Confusion / “wait what does this do?”: None, but it’s a lot of config surface area for a first run.
- Decision / next step: Search within `docs/experiments.md` for “refinement-block” / “experiment-ab-refinement-block” to jump to the relevant section.

### Step 14

- Intent: Jump directly to the A/B refinement-block experiment instructions.
- Action taken: Searched `docs/experiments.md` for `refinement-block` via `Select-String`.
- Expected result: A section containing the exact command(s) to run the experiment and any notes.
- Observed result: Matches found around lines ~296–297: `pdm run experiment-ab-refinement-block --dry-run` and `pdm run experiment-ab-refinement-block --runs 5`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open that part of the doc to see what it says about outputs and comparison.

### Step 15

- Intent: Read the A/B refinement block runner docs (stages, run commands, data mode).
- Action taken: Opened the relevant section of `docs/experiments.md` via `Get-Content .\\docs\\experiments.md | Select-Object -Skip 270 -First 80`.
- Expected result: A description matching the required 3-prompt pipeline and how to execute it.
- Observed result: The doc confirms:
  - It’s a “3-prompt custom pipeline” with stages `ab.random_token` -> `ab.scene_draft` -> `ab.scene_refine_*` -> `ab.final_prompt_format`.
  - Commands: `pdm run experiment-ab-refinement-block --dry-run` and `--runs 5`.
  - Data modes: `--data sample` (default) vs `--data config`.
- Confusion / “wait what does this do?”: None — this section maps directly to the task requirements.
- Decision / next step: Find (or add) “how to compare A vs B” instructions (ideally using the existing run-review tooling).

### Step 16

- Intent: Check whether “how to compare runs” is already documented somewhere before editing docs.
- Action taken: Searched `docs/experiments.md` for `run-review` via `Select-String`.
- Expected result: A snippet showing how to compare two generation ids.
- Observed result: Found a `run-review --compare <runA> <runB> --logs-dir <log_dir>` command at ~line 227, but it’s not co-located with the A/B refinement-block runner section.
- Confusion / “wait what does this do?”: Mild — as a new dev I wouldn’t necessarily know to look earlier in the same doc for comparison instructions.
- Decision / next step: Open the experiment runner implementation to confirm what identifiers/log paths it emits, then decide whether to add a short “Compare A vs B” snippet under the A/B runner section.

### Step 17

- Intent: Verify the experiment runner matches the required 3-prompt pipeline and how it constructs A vs B variants.
- Action taken: Opened `tools/run_experiment_ab_refinement_block.py` via `Get-Content .\\tools\\run_experiment_ab_refinement_block.py -TotalCount 260`.
- Expected result: A small harness that builds per-variant config overlays with stage sequences and runs `run_generation()`.
- Observed result: The runner:
  - Builds a custom stage sequence: `ab.random_token`, `ab.scene_draft`, `ab.scene_refine_{no_block|with_block}`, `ab.final_prompt_format`.
  - Generates a deterministic per-seed token `RV-<seed>-<roll>` and keeps it constant across A/B for the same run index.
  - Forces `run.mode` to `prompt_only` by default and disables context/upscale/upload.
  - Writes artifacts under `<output_root>/logs` and stamps `experiment.{id,variant,notes,tags}` into the run config.
- Confusion / “wait what does this do?”: Mild: the runner sets `prompt.generations_path` and `titles_manifest_path` even though `run.mode=prompt_only` skips CSV/manifest writes (so these paths look “required” but aren’t used in this mode).
- Decision / next step: Read the rest of the runner to see how `--dry-run`, `--data sample|config`, and results writing are implemented.

### Step 18

- Intent: Confirm the runner’s CLI args and artifact-writing behavior (plan + results).
- Action taken: Continued reading `tools/run_experiment_ab_refinement_block.py` via `Get-Content ... | Select-Object -Skip 260 -First 260`.
- Expected result: `--dry-run` exits after writing `experiment_plan.json`; non-dry-run writes both plan + results and runs A/B pairs.
- Observed result: Matches expectation:
  - Writes `experiment_plan.json` always (includes `validation_errors` and `planned_runs`).
  - Validates each merged config via `RunConfig.from_dict(...)`.
  - `--dry-run` returns after printing plan.
  - Non-dry-run loops through planned runs, calls `run_generation(...)`, then writes `experiment_results.json` with `final_prompt` and `outputs.prompt_pipeline`.
- Confusion / “wait what does this do?”: None — the runner is straightforward and readable.
- Decision / next step: Verify the stage catalog contains these `ab.*` stages and locate the actual prompt templates, especially the refinement block content for Variant B.

### Step 19

- Intent: Locate the Stage Catalog and the `ab.*` stage definitions.
- Action taken: Opened `image_project/impl/current/prompting.py` via `Get-Content .\\image_project\\impl\\current\\prompting.py -TotalCount 160`.
- Expected result: A centralized catalog mapping stable stage ids (strings) to prompt builders and capture behavior.
- Observed result: The top of the file contains prompt-building utilities (CSV loading, concept selection, prompt text generators) but not yet the stage catalog in the first 160 lines.
- Confusion / “wait what does this do?”: The file mixes a lot of older prompt generator functions; unclear where the “stage catalog” boundary is.
- Decision / next step: Search within `image_project/impl/current/prompting.py` for `ab.scene_draft` / `ab.final_prompt_format` to jump to the A/B stage definitions.

### Step 20

- Intent: Jump directly to the A/B experiment stage definitions inside the Stage Catalog.
- Action taken: Searched `image_project/impl/current/prompting.py` for `ab\\.scene_` via `Select-String`.
- Expected result: Line hits near where the `ab.*` stage catalog entries are defined.
- Observed result: Found matches around lines ~1158–1274 for `ab.scene_draft`, `ab.scene_refine_no_block`, and `ab.scene_refine_with_block`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Open the file around those line numbers to review the actual prompt templates and confirm Variant B contains the explicit refinement block.

### Step 21

- Intent: Confirm the required 3-prompt pipeline exists as catalog stages and that Variant B contains an explicit refinement block.
- Action taken: Opened `image_project/impl/current/prompting.py` around the `ab.*` definitions via `Get-Content ... | Select-Object -Skip 1120 -First 280`.
- Expected result: Four stages: random token action, scene draft, two refine variants (with/without block), and final strict-format stage.
- Observed result: Exactly matches:
  - `ab.random_token` (action; writes `ab_random_token`)
  - `ab.scene_draft` (writes `ab_scene_draft`)
  - `ab.scene_refine_no_block` vs `ab.scene_refine_with_block` (both write `ab_scene_refined`)
  - `ab.final_prompt_format` (default capture; writes `image_prompt`)
  - Variant B’s prompt includes a “Refinement block (apply silently; do not output this block)” checklist.
- Confusion / “wait what does this do?”: None — this is the clearest part of the pipeline so far.
- Decision / next step: Check config defaults + sample data paths so a truly fresh clone can run `pdm run experiment-ab-refinement-block` without personal paths.

### Step 22

- Intent: Verify the default config is runnable for a new developer (no personal drive paths, prompt-only default).
- Action taken: Opened `config/config.yaml` via `Get-Content .\\config\\config.yaml`.
- Expected result: `run.mode: prompt_only`, artifacts under `./_artifacts/`, and repo-relative sample CSV paths.
- Observed result: Config matches: `run.mode: prompt_only`, `_artifacts/` paths, `rclone.enabled: false`, `upscale.enabled: false`, `context.enabled: false`, and sample CSV paths under `./image_project/impl/current/data/sample/`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Confirm the referenced sample CSV files actually exist in the repo (since missing sample data is a common onboarding blocker).

### Step 23

- Intent: Confirm sample data exists at the paths used by default config and by the A/B runner `--data sample`.
- Action taken: Ran `Get-ChildItem .\\image_project\\impl\\current\\data\\sample`.
- Expected result: `category_list_v1.csv` and `user_profile_v4.csv` present.
- Observed result: Both files exist (`category_list_v1.csv`, `user_profile_v4.csv`).
- Confusion / “wait what does this do?”: None.
- Decision / next step: Check `pyproject.toml` to see how the experiment runner is registered (so a new dev learns the command surface).

### Step 24

- Intent: Confirm the A/B experiment runner is exposed as a simple `pdm run ...` command.
- Action taken: Opened `pyproject.toml` via `Get-Content .\\pyproject.toml`.
- Expected result: `[tool.pdm.scripts]` contains an entry like `experiment-ab-refinement-block = "python tools/..."`.
- Observed result: Found `experiment-ab-refinement-block = "python tools/run_experiment_ab_refinement_block.py"`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Decide whether to make any repo changes at all; the experiment appears already implemented + documented + runnable. The only missing piece I noticed is that the A/B runner docs don’t include an explicit “compare A vs B” snippet right next to the run commands.

### Step 25

- Intent: Check if the A/B refinement-block section continues with comparison instructions below the “Data” bullets.
- Action taken: Ran `Get-Content .\\docs\\experiments.md | Select-Object -Skip 330 -First 80`.
- Expected result: Either a “Compare A vs B” snippet or some reference to `run-review`.
- Observed result: No output (likely the file ends before line 330), so the A/B section appears to end right after the Data bullets.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Add a minimal “Compare A vs B” snippet directly under the A/B runner section in `docs/experiments.md`.

### Step 26

- Intent: Reduce onboarding friction by making “compare A vs B” discoverable right where the A/B runner is documented.
- Action taken: Edited `docs/experiments.md` to add a “Compare A vs B” section under “A/B Refinement Block Experiment Runner”.
- Expected result: A new developer can run the A/B harness and immediately see how to compare a run pair using `run-review`.
- Observed result: Added a small snippet: `pdm run run-review --compare <A_generation_id> <B_generation_id> --logs-dir <output_root>/logs`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Sanity-check the updated docs section renders as expected and then finish the diary with a Postmortem + `git diff --stat` output.

### Step 27

- Intent: Confirm the docs edit landed in the right place and reads cleanly.
- Action taken: Opened the tail of `docs/experiments.md` via `Get-Content .\\docs\\experiments.md | Select-Object -Skip 270 -First 120`.
- Expected result: “Compare A vs B” appears under the A/B runner section.
- Observed result: The snippet is present and correctly references `pdm run run-review --compare ... --logs-dir <output_root>/logs`.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Gather `git diff --stat` output and write the Postmortem section (top 10 friction points + suggested doc/path fixes).

### Step 28

- Intent: Make the new diary file show up in `git diff` output (untracked files don’t appear in `git diff`).
- Action taken: Ran `git add -N "new developer experiment preparation.md"`.
- Expected result: Git marks an “intent to add” so `git diff --stat` includes the new file without fully staging it.
- Observed result: No terminal output; file now shows as `A` in `git status --porcelain`.
- Confusion / “wait what does this do?”: Slight — `-N` is not a common flag; I had to remember that it’s “intent-to-add”.
- Decision / next step: Run `git diff --stat` to capture the final change surface area.

### Step 29

- Intent: Capture a concise summary of all working-copy changes for the diary (required output).
- Action taken: Ran `git diff --stat`.
- Expected result: A short list of modified/added files related to the A/B experiment + diary.
- Observed result: `git diff --stat` printed multiple “LF will be replaced by CRLF” warnings and showed changes in several files I didn’t directly edit in this session (README/config/prompting/pyproject). I’m pasting the exact `git diff --stat` output at the end of this file (per instructions).
- Confusion / “wait what does this do?”: Why is the working tree already dirty in unrelated-looking files (and why is Git warning about line endings) when I only edited docs + the new diary?
- Decision / next step: Run `git status --porcelain` to see exactly what’s modified vs untracked.

### Step 30

- Intent: Identify which changes are actually in the working tree (tracked vs untracked) so I can understand what a new dev would get on clone.
- Action taken: Ran `git status --porcelain`.
- Expected result: Only the files I touched (docs + new diary).
- Observed result: Many modified/untracked files. Verbatim output:
  - ` M README.md`
  - ` M config/config.yaml`
  - ` M docs/experiments.md`
  - ` M image_project/impl/current/prompting.py`
  - ` A "new developer experiment preparation.md"`
  - ` M pyproject.toml`
  - `?? image_project/impl/current/data/`
  - `?? new_developer_experiment_preparation.md`
  - `?? tools/run_experiment_3x3.py`
  - `?? tools/run_experiment_ab_refinement_block.py`
- Confusion / “wait what does this do?”: Big one — if the A/B runner + sample data are untracked (`??`), a fresh clone won’t have them, so the “happy path” docs would be lying. Also unclear why my environment has an untracked `new_developer_experiment_preparation.md` with almost the same name as the file I was asked to create.
- Decision / next step: Enumerate what’s inside `image_project/impl/current/data/` (to see exactly which sample files would be missing on clone), then proceed with the Postmortem using this as evidence of onboarding friction.

### Step 31

- Intent: Confirm what untracked prompt data exists under `image_project/impl/current/data/`.
- Action taken: Ran `Get-ChildItem .\\image_project\\impl\\current\\data -Recurse | Select-Object FullName`.
- Expected result: A `sample/` folder with the two referenced CSVs; anything else should be clearly ignorable.
- Observed result: `data/` contains `local/` and `sample/`, with two CSVs:
  - `...\\data\\sample\\category_list_v1.csv`
  - `...\\data\\sample\\user_profile_v4.csv`
- Confusion / “wait what does this do?”: Not sure whether `data/local/` is intended to be committed or ignored; as a new dev I’d assume “local” should be gitignored.
- Decision / next step: Write the Postmortem (top 10 friction points + what docs would have prevented each), then paste the `git diff --stat` output at the end.

### Step 32

- Intent: Make the required-but-currently-untracked experiment files appear in `git diff --stat` so the “Change Surface Area” is not misleading.
- Action taken: Ran `git add -N tools/run_experiment_3x3.py tools/run_experiment_ab_refinement_block.py image_project/impl/current/data/sample/category_list_v1.csv image_project/impl/current/data/sample/user_profile_v4.csv new_developer_experiment_preparation.md`.
- Expected result: Those files move from `??` to `A` in `git status` (intent-to-add), and `git diff --stat` includes them.
- Observed result: Command produced no output; proceeded to re-run `git diff --stat`.
- Confusion / “wait what does this do?”: Same as Step 28 — `-N` is a power-user git flag; not sure a new dev would think to use it.
- Decision / next step: Re-run `git diff --stat` and capture the final, comprehensive output.

### Step 33

- Intent: Capture a complete `git diff --stat` that includes the new experiment runner + sample CSVs.
- Action taken: Ran `git diff --stat`.
- Expected result: Should list 11 files (matching `git status`) and include the new tools + CSVs.
- Observed result: Success — it now shows 11 files changed, but also emits many “LF will be replaced by CRLF” warnings. I’m pasting this exact output at the end of this file.
- Confusion / “wait what does this do?”: The repeated CRLF warnings are noisy; not sure if the repo expects `core.autocrlf` to be set a certain way.
- Decision / next step: Run `git status --porcelain` one more time to confirm the `??` entries are gone (so the diff stat truly reflects what would need committing).

### Step 34

- Intent: Confirm the repo is no longer showing critical experiment files as untracked.
- Action taken: Ran `git status --porcelain`.
- Expected result: New experiment files show as added (`A`), not untracked (`??`).
- Observed result: The previously-untracked files now show as `A`, including the two sample CSVs and both tool runners.
- Confusion / “wait what does this do?”: None.
- Decision / next step: Finish with the Postmortem section and paste the Step 33 `git diff --stat` output verbatim at the end.

### Step 35

- Intent: Check whether the primary docs mention the required OpenAI credential for non-`--dry-run` prompt-only runs.
- Action taken: Searched `README.md` for `OPENAI_API_KEY|API key|api key` via `Select-String`.
- Expected result: README tells a new developer to set `OPENAI_API_KEY` before running `pdm run generate` or the experiment runner.
- Observed result: No matches — README does not mention the API key requirement.
- Confusion / “wait what does this do?”: As a new dev, I’d likely hit an auth error on first real run and have to guess what env var to set.
- Decision / next step: Call this out in the Postmortem as a docs gap (and suggest adding a 1-line note in README’s Quickstart).

### Step 36

- Intent: See if the API key requirement is documented anywhere else in the repo (so README could link to it instead of duplicating text).
- Action taken: Searched the repo for `OPENAI_API_KEY` via `rg -n "OPENAI_API_KEY"`.
- Expected result: At least one doc file (README/docs) mentions it.
- Observed result: The only hit outside this diary is `new_developer_experiment_preparation.md` (the underscored file), not the README or `docs/*`.
- Confusion / “wait what does this do?”: Confusing that the only “real” API key instruction lives in an untracked, similarly-named onboarding diary file.
- Decision / next step: In the Postmortem, recommend adding a 1-line `OPENAI_API_KEY` note to `README.md` Quickstart (or link to a dedicated setup doc).

### Step 37

- Intent: Confirm the repo test suite still passes after the experiment-related changes (basic sanity for a new dev).
- Action taken: Ran `pdm run test`.
- Expected result: Tests pass; any failures would be evidence of onboarding friction or broken integration.
- Observed result: `140 passed` with `1 warning` (the same `pydub` ffmpeg warning seen during experiment runs).
- Confusion / “wait what does this do?”: None.
- Decision / next step: Write the Postmortem and include the `pydub` warning noise as a top friction point (it appears even when not using media features).

### Step 38

- Intent: Close out the onboarding diary with the required Postmortem + change summary.
- Action taken: Appended the `Postmortem` section and pasted the final `git diff --stat` output.
- Expected result: The diary ends with actionable friction points + a concrete change surface area summary.
- Observed result: Postmortem and `git diff --stat` are present at the end of the file.
- Confusion / “wait what does this do?”: None.
- Decision / next step: None (task complete).

---

## Postmortem

### Top 10 friction points (with evidence)

1) **Command timeouts hide real outcomes**
   - Evidence: Step 4 timed out (`command timed out after 13042 milliseconds`) and looked like a failure until re-run (Step 5).

2) **Noisy `pydub`/ffmpeg warning even in prompt-only + tests**
   - Evidence: Step 4/5/8/37 all print `RuntimeWarning: Couldn't find ffmpeg or avconv...`.
   - Impact: Makes me think ffmpeg is required even when I’m only doing prompt work.

3) **API key requirement is not in the primary docs**
   - Evidence: Step 35 found no `OPENAI_API_KEY` mention in `README.md`; Step 36 shows the only repo mention is in `new_developer_experiment_preparation.md` (underscored).
   - Impact: A new dev will likely hit an auth error on first real run and have to guess the env var name.

4) **A/B “compare outputs” instructions were not discoverable where needed**
   - Evidence: Step 16 found `run-review --compare` earlier in `docs/experiments.md`, not next to the A/B runner section; required search (Step 14–16).
   - Fix applied: Added “Compare A vs B” snippet under the A/B runner section (Step 26–27).

5) **Stage Catalog is buried in a large mixed-purpose module**
   - Evidence: Step 19 opened `image_project/impl/current/prompting.py` and didn’t see the Stage Catalog; had to search by stage id (Step 20) to find `ab.*` definitions (Step 21).
   - Impact: New devs can’t quickly tell which functions are legacy vs actively wired stages.

6) **Repo “dirty state” confusion (many modified/untracked files)**
   - Evidence: Step 30 `git status --porcelain` showed many modified files plus untracked tool runners + sample data.
   - Impact: Hard to know what’s actually “in the repo” vs local workspace artifacts.

7) **Critical runnable components initially appeared untracked**
   - Evidence: Step 30 showed `?? tools/run_experiment_ab_refinement_block.py` and `?? image_project/impl/current/data/`.
   - Impact: If this reflects the real git state, a fresh clone won’t include the runner or sample CSVs, blocking onboarding.

8) **Evaluation artifacts show quality/encoding issues in Variant A output**
   - Evidence: Step 10/11 final prompt contains `workerƒ?Ts` (mojibake) while Variant B did not (Step 12).
   - Impact: Adds noise to the A/B comparison — not sure if it’s model randomness, prompt phrasing, or encoding.

9) **Too many config concepts to absorb before first successful run**
   - Evidence: Step 13 `docs/experiments.md` introduces many knobs (plans, stages include/exclude/overrides, refinement policies, context injection) before a new dev sees the simplest A/B runner.
   - Impact: New devs may overfit to config complexity instead of running the experiment quickly.

10) **Two similarly-named onboarding diary files**
   - Evidence: Step 2 and Step 30 show both `new developer experiment preparation.md` (requested) and `new_developer_experiment_preparation.md` (existing), which is easy to confuse.
   - Impact: Not clear which is canonical; also the underscored file was the only place the API key was documented (Step 36).

### What a new dev needed to learn that wasn’t obvious

- `--dry-run` is the safe starting point; it validates config + stage wiring without AI calls.
- The A/B runner is a `prompt.plan: custom` pipeline with stable `ab.*` stage ids; Variant A/B differ only by the refine stage.
- Comparison is easiest via generation ids + `run-review --compare`, using the experiment output’s `logs/` directory.
- The “random token” is deterministic per seed and is intended to be *held constant across A/B* to make comparisons fair.

### What docs/path would have prevented each friction point

- (1) Add “expected runtimes” (or progress prints) for dry-run vs real run in `README.md` and/or `docs/experiments.md`.
- (2) Document why `pydub` is imported and how to silence/avoid ffmpeg warnings in prompt-only mode (or defer the import so prompt-only users never see it).
- (3) Add a 1-line `OPENAI_API_KEY` requirement to `README.md` Quickstart, and link to a dedicated “Setup” doc.
- (4) Keep A/B comparison instructions co-located with the A/B runner section (done in Step 26).
- (5) Add a TOC or a “Stage Catalog starts here” marker in `image_project/impl/current/prompting.py`, or split the catalog into a smaller module.
- (6) Ensure the repo is clean on clone (no required files living as untracked). Add a “verify clean repo” note in onboarding docs.
- (7) Ensure sample CSVs + tool runners are committed (or provide a generation script that creates them deterministically).
- (8) Add a note in the experiment doc about expected variance/encoding artifacts and how many runs are needed before judging the refinement block’s impact.
- (9) Provide a “minimal path” doc section: “run 1 dry-run + run 1 prompt-only + compare” with no extra config details.
- (10) Consolidate to a single canonical onboarding diary filename (or move the underscored version into `docs/` and link it).

---

## `git diff --stat` (Step 33)

```text
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'config/config.yaml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/experiments.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'image_project/impl/current/prompting.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'pyproject.toml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'image_project/impl/current/data/sample/category_list_v1.csv', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'image_project/impl/current/data/sample/user_profile_v4.csv', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'new developer experiment preparation.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tools/run_experiment_3x3.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tools/run_experiment_ab_refinement_block.py', LF will be replaced by CRLF the next time Git touches it
 README.md                                          |   2 +
 config/config.yaml                                 |  30 +-
 docs/experiments.md                                |  51 +-
 .../impl/current/data/sample/category_list_v1.csv  |   9 +
 .../impl/current/data/sample/user_profile_v4.csv   |   4 +
 image_project/impl/current/prompting.py            | 178 +++++++
 new developer experiment preparation.md            | 355 ++++++++++++++
 new_developer_experiment_preparation.md            | 109 +++++
 pyproject.toml                                     |   2 +
 tools/run_experiment_3x3.py                        | 535 +++++++++++++++++++++
 tools/run_experiment_ab_refinement_block.py        | 445 +++++++++++++++++
 11 files changed, 1706 insertions(+), 14 deletions(-)
```
