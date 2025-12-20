# Prompt Pipeline (Steps + Blocks)

This project models the prompt workflow as a nested execution tree:

- **Step**: one LLM chat call (`ChatStep`).
- **Block**: a named group of nodes (`Block`), where nodes can be steps or other blocks.
- **Merge**: controls what (if anything) is persisted back to the parent conversation.
- **Transcript**: always records *every* step that ran (even when `merge="none"`). The transcript is not used as model context.

## Always-copy semantics (single merge point)

Every node executes against a working copy of its parent conversation. The parent conversation is only updated during a single merge at the end of that node. This avoids:

- multiple mutation sites
- half-applied steps when the model call fails

## Merge modes

Steps and blocks support three merge modes:

- `merge="all_messages"`: parent receives all messages produced inside the node (for steps: the user prompt + assistant response).
- `merge="last_response"`: parent receives only the final assistant response produced by the node.
- `merge="none"`: parent conversation is unchanged.

Merge only controls **conversation state**. It does not undo side effects of future tool steps.

### Tiny example

If a step prompts `"p1"` and gets response `"r1"`:

- `all_messages` merges `["user: p1", "assistant: r1"]`
- `last_response` merges `["assistant: r1"]`
- `none` merges nothing

## "Threads" via merge="none"

You can simulate independent "parallel" threads (without real parallelism) by running multiple steps with `merge="none"`:

- each thread step runs against the same baseline conversation state
- thread outputs are recorded in the transcript
- use `capture_key` to store each thread output in `ctx.outputs`
- a final consensus step reads captured outputs and writes the refined result back into the conversation (default merge)

## Tree-of-Thought (ToT) / enclave pattern

To keep the model context small while still logging full internal reasoning, wrap each stage in a `merge="last_response"` stage block, and run a refinement sub-block inside it.

Use the stable inner name `draft` for the stage's main step so transcript paths stay readable (e.g. `pipeline/section_2_choice/draft`).

```python
tot_enclave = Block(
    name="tot_enclave",
    merge="all_messages",
    nodes=[
        # "Thread" steps: run and are recorded, but do not write into the conversation.
        ChatStep(
            name="thread_1",
            merge="none",
            capture_key="enclave.section_2_choice.thread_1",
            prompt=...,
            temperature=0.8,
        ),
        ChatStep(
            name="thread_2",
            merge="none",
            capture_key="enclave.section_2_choice.thread_2",
            prompt=...,
            temperature=0.8,
        ),
        # Consensus: reads captured thread outputs and writes the refined result into the conversation.
        ChatStep(name="consensus", prompt=..., temperature=0.8),
    ],
)

stage = Block(
    name="section_2_choice",
    merge="last_response",
    nodes=[
        ChatStep(name="draft", prompt=..., temperature=0.8),
        tot_enclave,
    ],
    capture_key="optional_output_key",
)

pipeline = Block(name="pipeline", merge="all_messages", nodes=[stage, ...])
```

Result:

- model context stays minimal (parent only sees the refined stage output)
- transcript still contains all internal steps with unique paths

## Step Parameters

- Set temperature via `ChatStep.temperature` (do not include `"temperature"` inside `ChatStep.params`).

## Naming rules

- `ChatStep.name` and `Block.name` are optional.
- Missing names are deterministically generated per-parent using the node index:
  - unnamed steps: `step_01`, `step_02`, ...
  - unnamed blocks: `block_01`, `block_02`, ...
- Sibling name collisions are errors (no silent disambiguation).

The transcript includes a `path` for every step (e.g. `pipeline/section_2_choice/tot_enclave/consensus`) so repeated sub-blocks remain uniquely identifiable.

