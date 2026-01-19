# Prompt Plans + Stage Modifiers

The prompt pipeline is selected via `prompt.plan` and then modified via a single, explicit set of stage selectors/overrides. This enables fast experimentation (baseline, blackbox, refine-only, no-ToT, partial stage runs) without editing `image_project/app/generate.py`.

Stage wiring (stage composition, merge/capture behavior, provenance, IO contracts, stage-local config parsing) lives in `image_project/stages/*/*.py` as `STAGE: StageRef` exports. Prompt text helpers live in `image_project/prompts/*`.

## Run Mode (Prompt-Only)

Control whether the run executes the full media pipeline or only the prompt pipeline:

- `run.mode: full|prompt_only` (default `full`)

When `run.mode: prompt_only`:

- Runs the prompt pipeline as usual (produces `ctx.outputs["image_prompt"]`).
- Writes transcript + oplog as usual.
- Skips image generation / upscaling / manifest writes / generations CSV / rclone upload.
- Writes a small text artifact: `<log_dir>/<generation_id>_final_prompt.txt`.

Unknown `run.mode` values fail fast with `ValueError("Unknown run.mode: ...")`.

## Experiment Labels (Optional)

You can label runs for lightweight A/B testing and quick discovery without changing code:

```yaml
experiment:
  id: "holiday-theme"
  variant: "A"
  notes: "baseline plan + no-ToT"
  tags: ["ab", "prompt_only"]
```

When set, `experiment` is recorded in:

- transcript JSON (`<log_dir>/<generation_id>_transcript.json`)
- `run_review` HTML/JSON reports (and `--compare`)
- the run index JSONL (`<log_dir>/runs_index.jsonl`)

## Plans

Set `prompt.plan` to one of:

- `auto` (default): preserves the old behavior (`prompt.scoring.enabled: true` -> `blackbox`, otherwise `standard`)
- `standard`: the multi-stage pipeline (final stage produces `image_prompt`)
- `blackbox`: idea cards -> judge scoring -> selection -> final prompt stage (scoring sub-steps are regular stages)
- `blackbox_refine`: runs the blackbox idea-card pipeline to get a seed prompt, then iteratively refines it via the blackbox refinement loop
- `blackbox_refine_only`: iteratively refines a user-provided draft prompt via the blackbox refinement loop
- `refine_only`: refine a provided draft prompt into the final image prompt
- `baseline`: one-stage pipeline that captures `initial_prompt`
- `simple`: two-stage pipeline: `standard.initial_prompt` -> `standard.image_prompt_creation`
- `simple_no_concepts`: skips concept selection/filtering and uses generator-safe profile abstraction
- `direct`: one-stage final prompt from selected concepts + profile
- `profile_only`: `standard`, but forces context injection off (even if `context.enabled: true`)
- `profile_only_simple`: `simple`, but forces context injection off
- `custom`: run an explicit stage sequence from `prompt.stages.sequence`

Unknown plan values fail fast with `ValueError("Unknown prompt plan: ...")`.

## Refinement Stages

Refinement is represented as explicit stages in the plan flow (e.g. `refine.tot_enclave`). To disable ToT refinement for plans that include it, add it to `prompt.stages.exclude`.

## Stage Modifiers

Stage ids are stable strings defined by `StageRef.id` (canonical ids are namespaced, e.g. `standard.initial_prompt`, `blackbox.idea_cards_generate`). The config boundary registry is `image_project.stages.registry.get_stage_registry()` (also see `docs/stages.md`).

For convenience, `prompt.stages.include/exclude/overrides` and `prompt.output.capture_stage` also accept an unambiguous suffix (e.g. `initial_prompt` will resolve to `standard.initial_prompt` when running the standard plan). If a suffix is ambiguous, resolution fails fast.

- `prompt.stages.include: list[str]` (optional): if set, run only these stages (plan order preserved)
- `prompt.stages.exclude: list[str]` (optional): skip these stages
- `prompt.stages.overrides.<stage_id>`:
  - `temperature: float`
  - `params: dict` (must not contain `temperature`)
  - overrides apply only to the stage's primary `ChatStep(name=\"draft\", meta.role=\"primary\")`

Unknown stage ids or unknown override keys fail fast.

## Stage Configs (Stage-Owned)

Stage-specific config is validated by the stage that consumes it (typed getters + unknown-key enforcement). It is separate from `prompt.stages.overrides` (which only changes the primary draft step's `temperature`/`params`).

```yaml
prompt:
  stage_configs:
    defaults:
      # stage-kind defaults applied to all instances of this kind
      # (stage-specific keys; unknown keys are errors)
      standard.initial_prompt: {}
    instances:
      # per-instance overrides (instance id is usually the same as kind id)
      # (stage-specific keys; unknown keys are errors)
      standard.initial_prompt: {}
```

Notes:

- `defaults` keys are **stage kinds** (resolved via StageRegistry; suffixes allowed if unambiguous).
- `instances` keys are **stage instance ids** (after `prompt.stages.include/exclude` filtering; suffixes allowed if unambiguous).
- Unknown stage kind ids, unknown stage instance ids, and unknown config keys fail fast.
- Merge semantics: dicts are deep-merged, lists are replaced, scalars override.

## Concept Selection + Filters

Concept selection and filtering are configured under `prompt.concepts.*` and run as regular pipeline action stages:

- `preprompt.select_concepts`: chooses `ctx.selected_concepts` (random/fixed/file)
- `preprompt.filter_concepts`: applies ordered filters and records `concept_filter_log` in the transcript

Config keys:

- `prompt.concepts.selection.strategy: random|fixed|file` (default `random`)
- `prompt.concepts.selection.fixed: list[str]` (required when `strategy: fixed`)
- `prompt.concepts.selection.file_path: str` (required when `strategy: file`; one concept per non-empty line)
- `prompt.concepts.filters.enabled: bool` (default `true`)
- `prompt.concepts.filters.order: list[str]` (default `["dislike_rewrite"]`)
- `prompt.concepts.filters.dislike_rewrite.temperature: float` (default `0.25`)

## Output Capture

Choose which stage's final output becomes `ctx.outputs["image_prompt"]`:

- `prompt.output.capture_stage: <stage_id> | null`

If unset, the capture stage defaults to the last chat-producing stage in the resolved list.

## Refine-Only Inputs

For `prompt.plan: refine_only`, provide exactly one of:

- `prompt.refine_only.draft: "<text>"`
- `prompt.refine_only.draft_path: "./draft_prompt.txt"`

Missing draft input fails fast.

`prompt.plan: blackbox_refine_only` uses the same draft input keys (as the seed prompt).

## Custom Plan Stages

For `prompt.plan: custom`, provide:

- `prompt.stages.sequence: list[str]`

Stage ids must be stage kind ids from StageRegistry (or unambiguous suffixes). Unknown or ambiguous ids fail fast and list available ids.

## Context Injection

Context injectors are configured under `context.*` (default off via `context.enabled: false`).

Control where the generated context guidance text is injected:

- `context.injection_location: system|prompt|both` (default `system`)

Some plans can override context injection behavior regardless of config:

- `profile_only` / `profile_only_simple`: force context injection off.

## Observability

Each run records `outputs.prompt_pipeline` in the transcript JSON, including:

- requested plan name (e.g. `auto`)
- resolved plan name
- include/exclude lists
- resolved stage ids
- `stage_instances` (ordered list of `{instance, kind}`)
- `stage_io` (per-stage-kind IO contract summary)
- optional `stage_configs` summary (keys only; no prompt text)
- capture stage
- applied stage overrides
- explicit refinement stages (e.g. `refine.tot_enclave`, when present)
- effective context injection mode
- context injection location (`context.injection_location`)
- blackbox profile routing (`blackbox_profile_sources`, when blackbox stages are present)

Step paths in the transcript and oplog include stage ids (e.g. `pipeline/standard.section_2_choice/draft`, `pipeline/refine.tot_enclave/tot_enclave/fanout/hemingway`, `pipeline/refine.tot_enclave/tot_enclave/reduce/consensus/select/final_consensus`).

Each transcript step includes `type` (`chat` or `action`) and `meta` (when available), including:

- `meta.stage_id`
- `meta.source` (prompt/template identifier such as `prompts.idea_cards_judge_prompt`)
- `meta.doc` (short human-facing description, when defined by the stage)
- `meta.stage_kind` / `meta.stage_instance` (for stages built from StageRefs)

The transcript also includes `final_image_prompt` (top-level) when `ctx.outputs["image_prompt"]` is present.

Each run also appends a small discovery record to `<log_dir>/runs_index.jsonl` (one JSON object per line), including:

- generation_id, created_at, seed, status, run_mode
- optional experiment label
- `prompt_pipeline` summary (requested/resolved plan, resolved stages, capture stage)
- artifact paths (transcript/oplog/final_prompt/image/upscaled_image)

Operational logs include the same provenance fields on each step start/end line.

Discoverability helpers:

```bash
python -m image_project list-stages
python -m image_project list-plans
pdm run list-stages
pdm run list-plans
```

## Example Configs

Baseline: capture the first stage output (no ToT)

```yaml
prompt:
  plan: standard
  stages:
    include: ["select_concepts", "filter_concepts", "initial_prompt"]
  output:
    capture_stage: "initial_prompt"
```

Blackbox scoring pipeline

```yaml
prompt:
  plan: blackbox
  scoring:
    enabled: true
    num_ideas: 8
    # Profile text routing for A/B experiments:
    # - raw: ctx.outputs["preferences_guidance"]
    # - generator_hints: ctx.outputs["generator_profile_hints"] (requires it to be present + non-empty)
    # - generator_hints_plus_dislikes: generator_hints + ctx.outputs["dislikes"] list (no likes)
    judge_profile_source: raw  # raw|generator_hints|generator_hints_plus_dislikes
    final_profile_source: raw  # raw|generator_hints
  stages:
    overrides:
      idea_cards_judge_score:
        temperature: 0.0
```

Prompt-only A/B example (raw vs generator hints)

```yaml
run:
  mode: prompt_only
prompt:
  plan: blackbox
  scoring:
    enabled: true
    judge_profile_source: raw
    final_profile_source: generator_hints
```

Run twice with different configs (or config overrides) and compare with:

```bash
python -m image_project run-review --compare <runA> <runB> --logs-dir <log_dir>
```

Refine-only

```yaml
prompt:
  plan: refine_only
  refine_only:
    draft_path: "./draft_prompt.txt"
```

Profile-only (disable context injection even if enabled in config)

```yaml
prompt:
  plan: profile_only
context:
  enabled: true
```

Custom stage sequence

```yaml
prompt:
  plan: custom
  stages:
    sequence:
      - standard.initial_prompt
      - standard.image_prompt_creation
  output:
    capture_stage: standard.image_prompt_creation
```

## 3x3 Experiment Runner

The repo includes a small experiment harness that runs **3 distinct variants × 3 runs each** and labels every run via `experiment.{id,variant,tags}`.

```bash
pdm run experiment-3x3 --dry-run
pdm run experiment-3x3 --output-root ./_artifacts/experiments/my_3x3
```

Note: non-`--dry-run` execution forces `run.mode=full` so each run produces an image.
Concepts are randomly sampled per run index and reused across sets (A1/B1/C1 share concepts, etc).

Variants:

- **A**: `blackbox` + `prompt.scoring.enabled=true` + `prompt.scoring.{judge,final}_profile_source=generator_hints`
- **B**: `blackbox` + `prompt.scoring.enabled=true` + `prompt.scoring.{judge,final}_profile_source=raw`
- **C**: `simple_no_concepts`

## Profile v5 3x3 Runner

Compares v5 profile formats and a one-shot prompt path:

- **A**: `blackbox_refine` + v5 `like/dislike` profile
- **B**: `blackbox_refine` + v5 `love/like/dislike/hate` profile
- **C**: `direct` + v5 `love/like/dislike/hate` profile
- Random concepts: 2 injected per run index (shared across sets)
- Post-processing: profile nudge + OpenAI (GPT Image 1.5) prompt formatting
- Concept filters: enabled (see `preprompt.filter_concepts` / `prompt.concepts.filters.*`)
- Final prompt formatting: `postprompt.openai_format` (GPT Image 1.5 prompt text) after the blackbox refinement loop

Run:

```bash
pdm run experiment-profile-v5-3x3 --dry-run
pdm run experiment-profile-v5-3x3 --profile-like-dislike-path "M:/My Drive/image_project/data/user_profile_v5_like_dislike.csv" --profile-love-like-dislike-hate-path "M:/My Drive/image_project/data/user_profile_v5_love_like_dislike_hate.csv"
```

If you omit the `--profile-*` flags, the runner reads the profile paths from your config:
`experiment_runners.profile_v5_3x3.profile_like_dislike_path` and
`experiment_runners.profile_v5_3x3.profile_love_like_dislike_hate_path` (optional:
`experiment_runners.profile_v5_3x3.generator_profile_hints_path`). It fails fast if required paths are missing.

## A/B Refinement Block Experiment Runner

The repo also includes an A/B harness that runs a **3-prompt custom pipeline** and tests whether adding an explicit **refinement block** to the *middle* prompt improves the final formatted image prompt.

It runs pairs `A1/B1`, `A2/B2`, ... and labels each run via `experiment.{id,variant,tags}`.

Stages (custom plan):

- `ab.random_token` (action) -> `ab.scene_draft` -> `ab.scene_refine_*` -> `ab.final_prompt_format`

Run:

```bash
pdm run experiment-ab-refinement-block --dry-run
pdm run experiment-ab-refinement-block --runs 5
```

Data:

- Default: `--data config` uses the `prompt.categories_path` / `prompt.profile_path` from your loaded config.
- Use `--data sample` to use repo sample CSVs under `image_project/impl/current/data/sample/`.

Compare A vs B:

- The runner writes a pairing manifest at `<output_root>/pairs.json` so you don't need to manually match ids.
- Compare all run pairs with run-review:

```bash
pdm run run-review --compare-experiment <output_root> --all --output-dir <review_dir>
```

Compare a single pair (by run index):

```bash
pdm run run-review --compare-experiment <output_root> --pair 1 --output-dir <review_dir>
```
