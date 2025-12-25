# Black-Box Scoring (Optional)

This project can optionally add a **black-box scoring + selection** stage before final prompt generation.

## Why

- Generate multiple candidates, score them numerically, then pick one (best-of-N) without leaking judge feedback into the downstream prompt context.
- Helps increase variety across runs while keeping concept alignment.

## Enable

In `config/config.yaml`:

```yaml
prompt:
  scoring:
    enabled: true
    num_ideas: 6
    exploration_rate: 0.15
    judge_temperature: 0.0
    judge_model: null
    generator_profile_abstraction: true
    novelty:
      enabled: true
      window: 25
      method: df_overlap_v1   # df_overlap_v1 | legacy_v0
      df_min: 3
      max_motifs: 200
      min_token_len: 3
      df_cap: 10
      max_penalty: 20
      alpha_only: true
      scaling: linear         # linear | sqrt | quadratic
      stopwords_extra: []
```

## Novelty Methods

- `df_overlap_v1` (default): builds a weighted motif set from recent history using **document frequency** and applies a **normalized overlap** penalty so a handful of generic overlaps won’t instantly saturate to the max penalty.
- `legacy_v0`: preserves the older term-frequency-based behavior for A/B comparisons (can saturate quickly).

## Flow

1. Prepare scoring context (`blackbox.prepare`) including novelty summary and default generator hints.
2. Generate a generator-safe profile summary (`blackbox.profile_abstraction`) when enabled.
3. Generate N idea cards (`blackbox.idea_cards_generate`) as strict JSON.
4. Score idea cards with a separate judge (`blackbox.idea_cards_judge_score`) as strict JSON `{ "scores": [{"id","score"}] }`.
5. Select a winner in code (`blackbox.select_idea_card`) using epsilon-greedy and optional novelty penalty from recent `prompt.generations_path` history.
6. Generate the final prompt from the selected idea card (`blackbox.image_prompt_creation`).

## Prompt Refinement Loop (Blackbox)

Two additional plans add an iterative *prompt refinement loop* on top of blackbox scoring:

- `prompt.plan: blackbox_refine`: uses the blackbox idea-card pipeline to produce a seed prompt, then refines it for `m` iterations.
- `prompt.plan: blackbox_refine_only`: uses a user-provided draft prompt as the seed (`prompt.refine_only.draft` / `draft_path`), then refines it for `m` iterations.

The loop is configured under `prompt.blackbox_refine` (algorithm, branching factor, iterations, multiple judges + aggregation, mutation directives, etc.).

If `prompt.blackbox_refine.max_prompt_chars` is set and a candidate prompt exceeds it, the candidate is **truncated for judging/selection** and the truncation is recorded in `blackbox_scoring.prompt_refine.iterations[*]` (warnings + per-candidate metadata).

## Selection algorithm

- With probability `1 - exploration_rate`, choose the top effective score.
- With probability `exploration_rate`, sample from the top quartile weighted by effective score.
- If novelty is enabled and history is available, repeated motifs from recent generations reduce effective score via a lightweight code-level penalty.

## Transcript

Transcripts include a top-level `blackbox_scoring` object containing:

- `enabled` and a `config_snapshot`
- the judge `score_table`
- `selected_id`, `selected_score`, `selection_mode`, and the exploration roll
- novelty summary used (if enabled)
- refinement history when using `blackbox_refine*` plans (`blackbox_scoring.prompt_refine`, including `config_snapshot` + per-iteration candidates, judge outputs, aggregation, and selections)

When novelty is enabled, per-candidate selection tables include `novelty_detail` with a short breakdown (including top contributing motifs for `df_overlap_v1`).

The judge prompt/response is still recorded as normal steps for audit, but scoring text is **not** merged into the downstream prompt context.

Selection is represented explicitly as an `action` step in the transcript (`pipeline/blackbox.select_idea_card/action`), making it easy to see where parsing/selection happened.

## Troubleshooting

- `invalid_idea_cards_json`: the generator returned JSON that failed schema validation.
- `invalid_judge_output`: the judge returned non-JSON or a schema-violating score table.

