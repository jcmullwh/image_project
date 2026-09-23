PROJECT_SPEC — Enforce clean layering, move prompt policy out of framework, and make stage-owned config real

Task

Complete the next step of the refactor by:

1. Unblurring boundaries between foundation/, framework/, stages/, and prompt/impl code by moving prompt policy (persona catalogs, stage-specific prompt text, domain logic) out of framework/ and into stages/ (and/or a prompts module).


2. Ensuring framework/ contains only generic structural patterns/utilities (e.g., fanout/reduce, iterate, generate/select) and does not import implementation prompt modules.


3. Making stage-owned config namespaces a real extension surface by implementing stage config consumption for a small set of meaningful stages (at least refine.tot_enclave, plus one other high-value stage such as blackbox refinement loop or a judge panel stage).


4. Adding a clear, repo-local document: “This is where everything should live” and enforcing it with import-boundary tests.



This is explicitly a “clarity and delineation” step. No new “pipeline language” or meta-abstractions should be introduced.


---

Goal

Developer-visible outcomes

A reader can answer “where should this live?” without debate:

structural patterns → framework/

prompt policy + personas → stages//prompts/

generic primitives → foundation/


Adding a new experimental capability (e.g., OCR-based judge, toolcall-based verifier) can be done primarily by:

adding a new stage module + stage config keys,

without editing a monolithic central config validator.


Stage modules become more meaningful and less “boilerplate shells” over a giant prompt file.


Run-visible outcomes

Transcript shows:

explicit stage boundaries,

ToT/judges/blackbox implemented using shared structural patterns,

stage instance ↔ stage kind mapping,

stage config values that were actually consumed (for the stages that support it).




---

Context

Current observed issues (from the implemented snapshot)

framework/ still contains prompt policy:

ToT enclave prompt content/persona catalog currently lives in framework/refinement.py (prompt text + personas).


Some “framework” modules import prompt policy from implementation modules:

framework/blackbox_refine_loop.py imports from impl/current/*prompts* (directional boundary violation).


Stage-owned config machinery exists (ConfigNamespace, prompt.stage_configs.defaults/instances), but built-in stages mostly do not consume it, making it feel like overhead rather than an extension surface.

Prompt content remains centralized (e.g., impl/current/prompting.py), so stages can still feel like wrappers rather than standalone compositions.



---

Constraints

No silent fallbacks:

framework must not “reach into impl” and silently couple layers,

unknown stage config keys must fail loudly.


Offline tests only.

Determinism preserved; any stage config affecting randomness must be recorded.

Transcript is canonical; ensure paths remain stable and debug-friendly.

No new abstraction-on-abstraction:

keep execution model as Block/ChatStep/ActionStep,

keep orchestration in canonical run path.




---

Non-goals

Not implementing post-image analysis loops (OCR pipeline phases) in this change.

Not redesigning experimentation runners (but boundary rules must make unification easier later).

Not rewriting all prompt content organization (this change focuses on moving policy out of framework and reducing the worst centralization).



---

Functional requirements

FR1 — Framework must contain structure, not prompt policy

1. image_project/framework/** must not contain:

persona catalogs,

critic/judge prompt strings,

stage-specific “what should the model do” instructions.



2. framework/ may contain:

compositional block builders (“patterns”),

utilities used by stages and orchestration,

stage compilation/resolution logic (if it is generic across stages).



3. framework/ must not import:

image_project.impl.* (including prompt modules),

image_project.stages.*.




FR2 — ToT enclave prompt policy becomes stage-level implementation

1. The ToT enclave “enclave artists/personas” and prompt text move into:

image_project/stages/refine/tot_enclave.py and/or

image_project/stages/refine/tot_enclave_prompts.py (or prompts/refine.py).



2. The ToT stage must be implemented in terms of generic patterns:

use framework/blocks/patterns.py:fanout_then_reduce (or equivalent),

no ToT-specific structural builder remains in framework/ that embeds prompt policy.



3. Observability:

stage meta must preserve stage_kind and stage_instance,

transcript step paths remain unique and reflect the nested fanout/reduce structure.




FR3 — Blackbox refinement loop code must live with its stage(s)

1. Any code that imports blackbox-specific prompt policy must live in stages/blackbox_refine/** (or equivalent), not in framework/.


2. framework/ may provide:

generic iteration patterns (iterate),

generic generate/select patterns,

but not blackbox-specific prompts or selection directives.



3. The blackbox stage(s) should continue to build explicit per-iteration stage blocks with stable naming.



FR4 — Stage-owned config is a supported extension surface (not dead code)

Implement stage config consumption for at least these stages:

FR4.1 refine.tot_enclave stage config

Supported keys (in the stage’s ConfigNamespace):

critics: list of critic IDs (strings) to run (validated against known critic set)

max_critics: optional int clamp (min 1)

reduce_style: optional enum string (e.g., "consensus" | "best_of") if supported

capture_prefix: optional string (defaults to stage instance ID)


Behavior:

If critics is provided, fanout uses exactly those critics in that order.

If invalid critic ID is provided, raise with a clear error listing allowed IDs.

The stage must consume these keys via typed getters.

Unknown keys must cause failure via assert_consumed().


Observability:

record the resolved critic list under outputs.prompt_pipeline.stage_configs_effective[<instance>] (or similar).

ensure capture keys reflect the instance ID (or configured prefix) to avoid collisions.


FR4.2 One additional stage with meaningful config

Choose one of:

Blackbox refinement loop stage (recommended), or

A judge-panel stage if it exists and is intended to be reused.


Minimum required supported keys if using blackbox loop:

iterations: int (min 1, max reasonable bound) controlling how many iteration stage blocks are built

candidates_per_iter: int (min 1)

judges: list of judge IDs (validated)


Constraints:

This must not reintroduce “n=6 hides structure” at the plan boundary:

the plan may still expand iterations into explicit stage instances/blocks;

iterations controls internal generation only if that aligns with current structure, OR

it controls expansion only when using the blackbox loop stage’s own stage expansion (must remain transparent in transcript stage list).



(If you prefer to keep iteration count explicit in plans, then implement config on blackbox for candidates_per_iter and judges, not iterations.)

FR5 — Repository “where everything should live” document

1. Add docs/where_things_live.md with:

layer responsibilities (foundation/framework/stages/prompts/app/tools)

dependency direction rules

examples for ToT and blackbox



2. Link it from README or an existing docs index.



FR6 — Enforce boundaries with tests

1. Add import-boundary tests:

foundation must not import framework/stages/impl

framework must not import stages/impl



2. Tests must be offline and run quickly.




---

Proposed conversation flow

A) refine.tot_enclave as stage-level policy using generic fanout/reduce

(Shown in the “flow-first” style, matching what appears in code.)

Block(name="refine.tot_enclave_01", merge="last_response", meta={"stage_kind":"refine.tot_enclave"}, nodes=[
  Block(name="tot_enclave", merge="all_messages", nodes=[
    # Fanout critics (isolated)
    ChatStep(name="critic.hemingway", merge="none",
             capture_key="refine.tot_enclave_01.critic.hemingway", prompt=...),
    ChatStep(name="critic.octavia", merge="none",
             capture_key="refine.tot_enclave_01.critic.octavia", prompt=...),
    # Reduce/synthesize
    ChatStep(name="final_consensus", merge="last_response", prompt=...)
  ])
])

B) Blackbox loop stage living with blackbox stages (no framework prompt imports)

The stage composes generic patterns but owns its prompt policy:

Block(name="blackbox_refine.iter_01", merge="none", meta={"stage_kind":"blackbox_refine.iter"}, nodes=[
  Block(name="beam_01", merge="none", nodes=[
    ChatStep(name="cand_A", merge="none", prompt=...),
    ChatStep(name="cand_B", merge="none", prompt=...),
  ]),
  Block(name="judge", merge="none", nodes=[
    ChatStep(name="judge.j1", merge="none", prompt=...),
    ChatStep(name="judge.j2", merge="none", prompt=...),
  ]),
  ActionStep(name="select", fn=select_next_state)
])

No code under framework/ should import blackbox prompt modules.


---

Implementation plan

1) Move ToT prompt policy out of framework

Files to change/add (indicative):

Remove or slim down:

image_project/framework/refinement.py


Add:

image_project/stages/refine/tot_enclave.py

image_project/stages/refine/tot_enclave_prompts.py (or image_project/prompts/refine.py)


Update registry imports so the stage remains discoverable.


Design rules:

framework/blocks/patterns.py remains the only place for structural fanout/reduce logic.

ToT persona list and prompt templates are stage-level (or prompts module) and imported by the stage.


2) Move blackbox loop implementation out of framework if it imports prompt policy

Move:

image_project/framework/blackbox_refine_loop.py

→ image_project/stages/blackbox_refine/loop.py (or similar)


Update blackbox stage modules to import from that new location.

Ensure no framework/* imports impl/current/*prompts*.


If there are truly generic parts of the loop:

extract only those into framework/blocks/patterns.py (structure only, no prompt text).


3) Implement stage-owned config consumption for refine.tot_enclave

In stages/refine/tot_enclave.py:

read keys via ConfigNamespace typed getters:

critics = cfg.get_list_str("critics", default=<default list>)

max_critics = cfg.get_int("max_critics", default=len(critics), min=1)

capture_prefix = cfg.get_str("capture_prefix", default=instance_id)


validate critic IDs against the available critic definitions

build fanout steps accordingly

call cfg.assert_consumed() at the end



4) Implement stage-owned config consumption for one additional stage

Preferred: blackbox refinement iteration stage (or its loop builder) consumes at least:

judges: list[str]

candidates_per_iter: int


Implementation:

stage reads config, validates, builds nested blocks accordingly

record resolved values in prompt_pipeline summary (see observability section)


5) Add/extend prompt pipeline summary fields (observability)

In the pipeline summary (likely ctx.outputs["prompt_pipeline"]):

add:

stage_configs_effective: { <instance_id>: {<consumed_key>: <value>, ...} }


record only the small consumed values (avoid dumping huge dicts).


6) Add the “where things live” document

Add docs/where_things_live.md (content similar to the delineation you requested).

Link from README or docs index.


7) Add import boundary tests

Extend existing boundary test file or add:

tests/test_import_boundaries.py


Validate:

importing image_project.framework.* does not pull in image_project.impl.* or image_project.stages.*

foundation restrictions remain intact



Implementation approach:

use modulefinder or import graph inspection already used for foundation boundary tests (keep consistent).



---

Error handling + observability contract

New failure points

1. refine.tot_enclave stage config contains invalid critic ID:

raise ValueError with allowed IDs listed.



2. blackbox stage config contains invalid judge ID:

raise similarly.



3. Unknown stage config keys:

ConfigNamespace.assert_consumed() must raise with:

stage instance id

unknown keys

known/consumed keys




4. Import boundary violations:

tests must fail clearly indicating offending import edge.




Observability requirements

Transcript must continue to show:

stage boundary blocks (instance IDs)

nested algorithmic structure


Prompt pipeline summary must include:

stage instance ↔ kind mapping (already expected)

stage_configs_effective for the stages that consume config keys




---

Data/artifact changes

Transcript JSON:

add outputs.prompt_pipeline.stage_configs_effective (small, consumed-only)


No CSV or manifest schema changes.


Backward compatibility:

Additional fields only; existing consumers should ignore unknown keys.



---

Testing requirements (pytest, offline)

Unit tests

1. Framework boundary



importing image_project.framework must not import image_project.impl or image_project.stages.


2. ToT stage config consumption



providing prompt.stage_configs.instances.refine.tot_enclave_01.critics=[...]:

results in exactly those critic steps being present in transcript paths

stage_configs_effective records the critic list


typo key triggers failure via unknown-key enforcement.


3. Blackbox stage config consumption



set judges list / candidates per iter and assert:

expected step count or expected nested paths exist

config summary recorded


invalid judge ID fails loudly.


Integration test

Run run_generation() with fakes and assert:

transcript exists

ToT stage exists and is built without importing framework refinement prompt policy

framework import boundary test passes



---

Documentation updates

Add docs/where_things_live.md

Update:

docs/pipeline.md (or equivalent) to state:

framework contains structural patterns only

ToT personas/prompt policy are stage-level

stage configs are consumed by stages, with unknown-key enforcement



Provide examples:

overriding ToT critics list

overriding blackbox judges list




---

Acceptance criteria

1. image_project/framework/ contains no ToT persona catalogs or prompt strings; ToT prompt policy is stage-level.


2. image_project/framework/ does not import image_project.impl.* or image_project.stages.* (enforced by tests).


3. refine.tot_enclave stage consumes stage config keys via ConfigNamespace and rejects unknown keys.


4. One additional stage (preferably blackbox) consumes stage configs in the same way.


5. docs/where_things_live.md exists and is linked from README/docs index.


6. Transcript and prompt pipeline summary record consumed stage config values in a small, clear form.


7. All tests pass offline.




---

Pitfalls to avoid

“Fixing” boundary blur by moving everything into framework (recreates the original problem).

Reintroducing hidden behavior: ToT must remain an explicit stage.

Dumping full stage config dicts to transcript (keep only consumed keys/values).

Allowing framework to depend on prompt policy via indirect imports (tests must catch).

Using config coercions (bool("false")) anywhere in stage config getters.