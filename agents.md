# PROJECT_SPEC_TEMPLATE.md

Project-specific spec template and guidance for this repo (the “image_project” pipeline).

Use this when writing implementation specs or working on or with the repo. It is tuned to the actual architecture and recurring failure modes in this codebase.

---

## What this repo is

A **living experimentation harness** that:

- builds multi-step LLM prompt pipelines (Step/Block) to produce an image prompt,
- writes structured artifacts (JSON transcript + CSV records + titles manifest),
- supports iterative experimentation (adding/removing stages, adding refinement loops, future toolcalls/agentic subroutines).

The primary “product” is not just the image—it’s the **repeatable run**, **easy experimentation**, and the **audit trail**.

---

## Core invariants

### 1) No silent fallbacks
If something is missing, invalid, or fails, the system must:
- raise with a clear error (preferred), OR
- warn loudly and follow a clearly documented and logged best-effort behavior (ANY fallback MUST be human approved)

Avoid patterns that:
- set required modules to `None` and continue,
- swallow exceptions and keep running,
- default output paths to CWD without warning,

### 2) Clarity beats cleverness
A reader should be able to understand “what runs” from the flow definition without decoding predicates, magic indices, or hidden behavior.

### 3) NO BAND-AIDS OR WHACK-A-MOLE FIXES

When solving a problem, don’t patch the symptom. Fix the root cause. Design for the class of failure, not the single observed instance.

Think in architecture terms. If a component is intended to be generic, your fix must remain generic. If the issue is user-specific, it belongs in user-specific layers/config—not in shared pipeline logic.

Avoid “anti-pattern fixes.” Don’t hardcode exceptions, add one-off rules, or introduce narrowly scoped scoring knobs to chase a specific bug. Those tend to accumulate, interact unpredictably, and regress elsewhere.

Example (what not to do):
A user profile says “I don’t like disembodied arms.” The pipeline starts injecting “well-rendered arms” into every prompt. That creates irrelevant arms that are technically correct but narratively pointless.

Incorrect fixes (these were attempted and are not acceptable):

Adding “no arms” into the generic prompt pipeline. (Too specific to one preference; does not scale to other dislikes.)

Adding an “arm risk score” to the generic pipeline. (Narrowly targeted, brittle, and likely to be noisy/false-positive.)

Correct fix (generic and scalable):
Add a rule that prevents the entire class of mistake:

“Respect dislikes by avoiding them; do not add the opposite of a dislike unless the scene genuinely requires it.”

This addresses the underlying failure mode: naively converting negative preferences into universal, literal counter-instructions, which creates artificial, irrelevant artifacts.

### 4) Determinism when randomness is used
When randomness affects prompts:
- seed must be recorded and logged,
- seed must be stored in transcript,
- runs must be reproducible with the same seed.

### 5) Transcript is canonical for debugging
Transcript JSON must contain:
- step-by-step prompts/responses + params,
- nested pipeline paths,
- seed, selected concepts,
- error metadata on failure.
- ALL new categories of information added to the project not specifically listed here.

Transcript is not model context; it’s a recorder.

### 6) Offline testability
All tests must run offline:
- mock/fake LLM and image backends
- no network calls
- no real external binaries (use fakes or platform-safe stubs)

---

If ever creating a spec or plan, use .todos\spec_template.md as a reference.
