## Task: Threaded enclave opinions via step-level merge modes

### Goal

Represent the enclave as **independent “threads”**:

* One response per artist (Hemingway/Munch/da Vinci/Representative/Chameleon)
* Threads are **independent**: they do **not** see each other’s outputs
* Still executed sequentially (no real parallelism), but with controlled context so the behavior matches “parallel threads”
* A final **consensus** step combines those independent opinions into a single revised output
* Full transcript retains all thread outputs for analysis/debugging

This should be done **without** awkward nesting like “wrap every artist step in a `Block(merge="none")`”. The result should feel as simple and readable as your current Steps/Blocks structure.

### Current state (baseline)

`main.py` currently models the enclave as two steps inside one block:

* `enclave_opinion` produces one response containing all artists
* `enclave_consensus` produces the merged result 

---

## Design decision

### Key idea

Add **merge behavior** to `ChatStep` (just like `Block.merge`) so a step can run and be logged/captured **without** writing into the conversation history.

That unlocks thread semantics cleanly:

* Each artist step runs against the same “baseline” conversation state
* Each artist step uses `merge="none"` so it doesn’t affect later artist steps
* Each artist step writes its output to `ctx.outputs[...]` via `capture_key`
* A consensus step reads those captured outputs and writes the revised response into the conversation (default merge)

### Why this is better than reintroducing “main vs copy”

Reintroducing separate read/write contexts adds a second axis that is easy to misunderstand (“which conversation am I in?”). Step-level merge keeps the core rule simple:

* **Everything executes on a working copy**
* **Only merge controls what the next step sees**
* “Threads” = “steps that don’t merge”

---

## Non-goals

* Do not implement true parallel execution.
* Do not implement multi-agent enclaves / separate processes.
* Do not change prompt content outside enclave steps except where required to support thread prompts.
* Do not add new selection DSL (`where=lambda ...` filters) to pipeline definitions.
* Do not add external dependencies.

---

## Functional requirements

### FR1 — `ChatStep` supports merge modes

Add `merge: MergeMode = "all_messages"` to `ChatStep`.

* Allowed values: `"all_messages" | "last_response" | "none"` (same as `Block.merge`)
* Default is `"all_messages"` to preserve current behavior everywhere
* Validate in `ChatStep.__post_init__`:

  * invalid values raise `ValueError` (fail-fast)

**Meaning:**

* `all_messages`: parent working conversation receives the step’s user prompt + assistant response
* `last_response`: parent receives only the assistant response
* `none`: parent receives nothing (but the step still runs and is recorded)

### FR2 — Runner respects `ChatStep.merge` everywhere

Update pipeline execution so the merge mode comes from the node itself:

* Root node merge behavior:

  * if root is `ChatStep`, use `step.merge` (not forced to `all_messages`)
  * if root is `Block`, use `block.merge` (existing)

* Child merge behavior inside blocks:

  * use `child.merge` for both `ChatStep` and `Block`
  * remove the special case that forces steps to behave as `all_messages`

**Invariant:** parent conversation is still mutated only by `_apply_merge(...)`.

### FR3 — Threaded enclave structure in `main.py`

Replace the single `enclave_opinion` step with **N thread steps**, one per artist:

* Each thread step:

  * reads the same baseline conversation (the stage draft output is the last assistant message)
  * has `merge="none"` so it does not change the conversation for other threads
  * has a `capture_key` so its output is accessible to consensus

* Final consensus step:

  * reads all captured thread outputs from `ctx.outputs`
  * produces the revised response (the refined output for the stage)
  * must **merge** (default `all_messages`) so the stage has an assistant output to commit

### FR4 — Consensus prompt fails loudly if thread outputs are missing

If any expected thread output is missing/blank at consensus time, raise `ValueError` with the missing key.

No silent “continue with fewer artists”.

### FR5 — Thread prompts must explicitly enforce independence

Each artist prompt must say (in plain language) that:

* the artist is a single voice
* they do not see other artists
* they should critique/refine the last assistant response
* return a structured response (bullets / edits)

This is to prevent models from collapsing into “five voices in one answer”.

### FR6 — Transcript contains one record per artist

After a run, the transcript should show paths like:

* `pipeline/<stage>/tot_enclave/hemingway`
* `pipeline/<stage>/tot_enclave/munch`
* …
* `pipeline/<stage>/tot_enclave/consensus`

(Exact stage path depends on your wrapper blocks; the key is “each artist is its own step record”.)

---

## Implementation instructions

### Step 0 — Locate the pipeline runner and node types

Modify `pipeline.py` (where `ChatStep`, `Block`, `ChatRunner`, `MergeMode` live).

Do **not** change transcript format beyond what’s already recorded; you’re only adding step-level merge behavior.

### Step 1 — Add `merge` to `ChatStep`

In `pipeline.py`:

* Add `merge: MergeMode = "all_messages"` to `ChatStep`
* Validate in `__post_init__`:

  * if `merge not in ALLOWED_MERGE_MODES`: raise `ValueError("Invalid step merge mode: ...")`

Be careful not to break call sites:

* add the field with a default so existing `ChatStep(...)` usage still works.

### Step 2 — Respect step merge in `ChatRunner`

Update:

1. `ChatRunner.run(...)`:

   * currently chooses `all_messages` for root steps
   * change to use `node.merge` for steps too

2. `ChatRunner.run_step(...)`:

   * currently applies `merge="all_messages"` unconditionally
   * change to apply `step.merge`

3. `ChatRunner._execute_block(...)`:

   * currently computes merge mode as:

     * blocks use `child.merge`
     * steps forced to `"all_messages"`
   * change to: `merge_mode = child.merge` for both

**Do not change** how steps are executed:

* Step execution should still run on a working copy and return produced delta.
* Only merge behavior changes what gets appended to the parent working conversation.

### Step 3 — Implement threaded enclave in `main.py`

Replace the existing two-step enclave with:

#### 3.1 Define a single source of truth for artists

Add something like:

* `ENCLAVE_ARTISTS = [(key, label, persona), ...]`

Keep the `key` simple and path-safe: `hemingway`, `munch`, `da_vinci`, `representative`, `chameleon`.

#### 3.2 Add a per-artist prompt builder

Implement:

* `enclave_thread_prompt(label: str, persona: str) -> str`

Output should force:

* one voice only
* structured “issues” + “edits”
* no meta commentary

#### 3.3 Build a per-stage `tot_enclave` block

Make `tot_enclave` a function `make_tot_enclave_block(stage_name: str) -> Block` so capture keys are stage-scoped.

Block structure:

* `ChatStep(name=<artist_key>, merge="none", capture_key=f"enclave.{stage_name}.{artist_key}", prompt=...)` repeated for all artists
* `ChatStep(name="consensus", prompt=..., merge="all_messages")`

Consensus prompt factory should:

* read each `ctx.outputs[f"enclave.{stage_name}.{artist_key}"]`
* raise if any are missing/blank
* provide those notes to the model
* instruct: “Return ONLY the revised response.”

**Important pitfall:** when creating prompt callables in a loop, bind loop variables with default args or helper functions to avoid the “late binding closure” bug.

### Step 4 — Wire it into `refined_stage(...)`

Instead of using a single global `tot_enclave_block`, create it per stage:

* inside `refined_stage`, compute `enclave = make_tot_enclave_block(stage_name)`
* use `nodes=[draft_step, enclave]`

Everything else stays unchanged.

---

## Testing requirements (pytest, offline)

### Unit tests for step-level merge

Add tests in `tests/test_step_block_structure.py` (or a new test file) using a FakeTextAI that records `messages` passed into each call.

#### T1 — `ChatStep(merge="none")` does not affect subsequent steps

Create a block with two steps:

* `s1` with `merge="none"`
* `s2` with `merge="none"`

Assert:

* both model calls see the same base conversation length (system + current user prompt only)
* `ctx.messages` remains unchanged if the parent block also merges none (or remains consistent if merged elsewhere)
* transcript records both steps

#### T2 — `ChatStep(merge="last_response")` appends only assistant response

Build a root block that merges all, containing one step with merge last_response.

Assert parent conversation includes:

* system
* assistant “response”
  and **not** the user prompt message for that step.

#### T3 — invalid merge mode raises

Construct `ChatStep(merge="bad")` and assert `ValueError`.

### Integration test for threaded enclave semantics

Add a focused test (can be small) that simulates:

* a “draft” step that produces a draft assistant response (merged normally)
* two “artist” steps with `merge="none"` (captured outputs)
* a consensus step that reads captures and returns a refined output

Assertions:

* the second artist model call does **not** include the first artist’s output in its input messages
* consensus prompt includes both captured outputs
* final merged output equals consensus output

This test should verify independence by inspecting the `messages` passed to FakeTextAI, not by matching prompt strings.

---

## Documentation updates

Update `docs/pipeline.md` (or equivalent) to include:

1. `ChatStep.merge` modes and meaning (same as block)
2. Explanation of “thread” behavior:

   * “thread steps run and are recorded, but do not write into conversation”
   * “consensus reads captured outputs”
3. One short example showing:

   * a stage draft
   * multiple merge-none thread steps
   * consensus

Keep it written in plain language.

---

## Acceptance criteria

* Each enclave artist produces its own transcript step entry per stage.
* Artist steps do not see each other’s outputs (verified by unit/integration tests).
* Consensus step combines the captured notes and produces the refined stage output.
* No wrapper “Block per artist” workaround is needed.
* All tests pass offline under pytest.
* No silent fallbacks: missing thread outputs cause a clear error.

---

## Notes / pitfalls to avoid

* **Closure capture bug:** don’t write `lambda: artist_prompt(artist_label)` inside a loop without binding defaults.
* **Accidental merge:** if an artist step is not set to `merge="none"`, you lose independence.
* **Consensus formatting:** if consensus returns “analysis + answer”, that becomes your stage output (because stage commits the last response). Make the instruction “return only the revised response” explicit.
* **Key collisions:** use stage-scoped capture keys or ensure consensus reads only the current stage’s captures, not leftovers.

If you want this to be extra experiment-friendly later, you can put the artist list + temperatures into config, but that’s explicitly out of scope for the minimal version above.
