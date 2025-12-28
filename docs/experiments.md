# Prompt Plans + Stage Modifiers

The prompt pipeline is selected via `prompt.plan` and then modified via a single, explicit set of stage selectors/overrides. This enables fast experimentation (baseline, blackbox, refine-only, no-ToT, partial stage runs) without editing `image_project/app/generate.py`.

Stage wiring (prompt builder, temperatures, merge/capture behavior, default refinement policy, provenance) is defined once in `image_project/impl/current/prompting.py`.

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

## Refinement Policy

Set `prompt.refinement.policy`:

- `tot`: Tree-of-Thought enclave refinement (default)
- `none`: no enclave; each stage is a single draft step

## Stage Modifiers

Stage ids are stable strings defined in the Stage Catalog (`image_project/impl/current/prompting.py`). Canonical ids are namespaced (e.g. `standard.initial_prompt`, `blackbox.idea_cards_generate`).

For convenience, `prompt.stages.include/exclude/overrides` and `prompt.output.capture_stage` also accept an unambiguous suffix (e.g. `initial_prompt` will resolve to `standard.initial_prompt` when running the standard plan). If a suffix is ambiguous, resolution fails fast.

- `prompt.stages.include: list[str]` (optional): if set, run only these stages (plan order preserved)
- `prompt.stages.exclude: list[str]` (optional): skip these stages
- `prompt.stages.overrides.<stage_id>`:
  - `temperature: float`
  - `params: dict` (must not contain `temperature`)
  - `refinement_policy: tot|none`

Unknown stage ids or unknown override keys fail fast.

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

If unset, the plan's default capture behavior applies (typically the final stage).

## Refine-Only Inputs

For `prompt.plan: refine_only`, provide exactly one of:

- `prompt.refine_only.draft: "<text>"`
- `prompt.refine_only.draft_path: "./draft_prompt.txt"`

Missing draft input fails fast.

`prompt.plan: blackbox_refine_only` uses the same draft input keys (as the seed prompt).

## Custom Plan Stages

For `prompt.plan: custom`, provide:

- `prompt.stages.sequence: list[str]`

Stage ids must be catalog ids (or unambiguous suffixes). Unknown or ambiguous ids fail fast and list available ids.

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
- global refinement policy
- include/exclude lists
- resolved stage ids
- capture stage
- applied stage overrides
- effective context injection mode
- context injection location (`context.injection_location`)
- blackbox profile routing (`blackbox_profile_sources`, when blackbox stages are present)

Step paths in the transcript and oplog include stage ids (e.g. `pipeline/standard.section_2_choice/draft`, `pipeline/standard.initial_prompt/tot_enclave/...`).

Each transcript step includes `type` (`chat` or `action`) and `meta` (when available), including:

- `meta.stage_id`
- `meta.source` (prompt/template identifier such as `prompts.idea_cards_judge_prompt`)
- `meta.doc` (short human-facing description, when defined by the catalog)

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
  refinement:
    policy: none
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
  refinement:
    policy: tot
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
  refinement:
    policy: tot
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
  refinement:
    policy: none
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

- **A**: `blackbox` + `prompt.scoring.enabled=true` + `prompt.scoring.{judge,final}_profile_source=generator_hints` + `refinement.policy=none`
- **B**: `blackbox` + `prompt.scoring.enabled=true` + `prompt.scoring.{judge,final}_profile_source=raw` + `refinement.policy=none`
- **C**: `simple_no_concepts` + `refinement.policy=none`

## Profile v5 3x3 Runner

Compares v5 profile formats and a one-shot prompt path:

- **A**: `blackbox_refine` + `refinement.policy=none` + v5 `like/dislike` profile
- **B**: `blackbox_refine` + `refinement.policy=none` + v5 `love/like/dislike/hate` profile
- **C**: `direct` + `refinement.policy=none` + v5 `love/like/dislike/hate` profile

Run:

```bash
pdm run experiment-profile-v5-3x3 --dry-run
pdm run experiment-profile-v5-3x3 --profile-like-dislike-path "M:/My Drive/image_project/data/user_profile_v5_like_dislike.csv" --profile-love-like-dislike-hate-path "M:/My Drive/image_project/data/user_profile_v5_love_like_dislike_hate.csv"
```

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

- Default: `--data sample` uses repo sample CSVs under `image_project/impl/current/data/sample/`.
- Use `--data config` to use the `prompt.categories_path` / `prompt.profile_path` from your loaded config.

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
