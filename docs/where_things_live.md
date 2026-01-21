# Where Things Live (Layering + Boundaries)

This repo is intentionally layered so experiments stay easy to iterate on, failures are debuggable, and prompt policy does not leak into generic infrastructure.

## Layers

### `pipelinekit/` (kernel primitives)

- Generic pipeline engine and stage authoring kit: `ChatStep`, `ActionStep`, `Block`, `ChatRunner`,
  step recording, strict `ConfigNamespace`, `StageRef`/`StageRegistry`, and the policy-driven compiler.
- Must not import `image_project.*` (enforced by tests).

### `image_project/foundation/` (repo helpers)

- Repo helpers: config IO helpers, logging helpers, and other low-level utilities.
- Must not import `image_project.framework`, `image_project.stages`, or `image_project.impl`.

### `image_project/prompts/` (prompt policy + parsing helpers)

- Canonical home for prompt templates, persona catalogs, and prompt/parse helpers.
- Must be pure: no imports from `image_project.stages`, `image_project.app`, or `image_project.impl`.

### `image_project/framework/` (structure + orchestration utilities)

- Structural utilities used across many stages (e.g. transcript/artifact helpers, run context, plan input contracts).
- May import `pipelinekit` (pipeline primitives), `image_project.foundation`, and `image_project.prompts` (for helper LLM calls like titles/concept filters).
- Must not import `image_project.stages` or `image_project.impl`.
- Must not embed multi-line prompt policy; import it from `image_project.prompts.*` instead.

Common locations inside framework:

- Artifact writing/indexing: `image_project/framework/artifacts/*`
- Prompt-pipeline authoring helpers + compilation policies: `image_project/framework/prompt_pipeline/*`

### `image_project/stages/` (stage catalog + stage implementations)

- Stage definitions (`StageRef`) that compile to `Block/ChatStep/ActionStep`.
- May import `pipelinekit` + `image_project.framework` + `image_project.foundation` + `image_project.prompts`.
- Must not import `image_project.impl`.

### `image_project/impl/` (experiment/plans glue)

- Prompt plans, plan plugins, and experiment-specific wiring.
- May import `image_project.stages` + `image_project.framework` + `image_project.foundation` + `image_project.prompts`.
- Must not be a prompt-policy dumping ground: prompt templates belong in `image_project/prompts/`.
- Experiments are registered plugins under:
  - `image_project/impl/current/experiments.py` (registry + interface)
  - `image_project/impl/current/experiment_plugins/*` (experiment definitions)

### `image_project/app/` and `tools/` (entrypoints + maintenance)

- CLI entrypoints (`run_generation()`), docs generation tools, experiment runners.
- Can import the layers above; should not contain prompt policy.
- Canonical experiment runner: `image_project/app/experiment_runner.py` (plugins supply plan-building only).

## Dependency direction

Allowed direction (top depends on bottom):

- `pipelinekit` -> (stdlib / third-party)
- `image_project/foundation` -> (stdlib / third-party)
- `image_project/prompts` -> (stdlib / third-party) (+ `image_project/foundation` helpers when needed)
- `image_project/framework` -> `pipelinekit` + `image_project/foundation` (+ `image_project/prompts` allowed)
- `image_project/stages` -> `pipelinekit` + `image_project/framework` + `image_project/foundation` (+ `image_project/prompts` allowed)
- `image_project/impl` -> `image_project/stages` + `image_project/framework` + `image_project/foundation` (+ `image_project/prompts` allowed)
- `image_project/app` + `tools` -> `pipelinekit` + `image_project/impl` + `image_project/stages` + `image_project/framework` + `image_project/foundation` (+ `image_project/prompts` allowed)

Disallowed direction:

- `pipelinekit` -> `image_project.*`
- `image_project/framework` -> `image_project/stages` or `image_project/impl`
- `image_project/stages` -> `image_project/impl`
- `image_project/prompts` -> `image_project/stages` / `image_project/app` / `image_project/impl`
- `image_project/foundation` -> anything above it (except `pipelinekit`)

## Examples

### ToT enclave (refinement)

- Structural pattern: `pipelinekit/engine/patterns.py`
- Stage + prompt policy: `image_project/stages/refine/tot_enclave.py` and `image_project/stages/refine/tot_enclave_prompts.py`

Example config override (run fewer critics):

```yaml
prompt:
  stage_configs:
    instances:
      refine.tot_enclave:
        critics: ["hemingway", "representative"]
        max_critics: 2
        reduce_style: "best_of"
```

### Blackbox refine loop

- Loop stage implementation (init + N iterations + finalize): `image_project/stages/blackbox_refine/loop.py`
- Prompt plans that use it: `image_project/impl/current/plan_plugins/blackbox_refine.py`

Example config override (apply to all iterations via kind defaults):

```yaml
prompt:
  stage_configs:
    defaults:
      blackbox_refine.loop:
        iterations: 6
        algorithm: hillclimb  # hillclimb|beam
```

## Observability

Stages that consume stage config keys record the small, consumed-only effective values under:

- `outputs.prompt_pipeline.stage_configs_effective[<stage_instance_id>]`

Compiled pipeline metadata also records per-stage effective IO (declared + discovered capture keys):

- `outputs.prompt_pipeline.stage_io_effective[<stage_instance_id>]`

Unknown stage config keys fail loudly (no silent ignore).

## Stage macro rule

Plans author **StageInstances only**:

- A stage is a macro (`StageRef.instance()` + stage configs + IO) that compiles to a stage block.
- Plans must not return raw `Block` nodes inline.
