# Prompt Stage Catalog

`stage_catalog.py` is the single source of truth for prompt-stage wiring (prompt builder, temperature, merge/capture behavior, default refinement policy, and provenance metadata).

To print the live catalog from code:

```bash
.\.venv\Scripts\python scripts/list_prompt_catalog.py stages
```

## Stages

### Preprompt

- `preprompt.select_concepts` — Select concepts for the run. (`prompts.select_random_concepts`)
- `preprompt.filter_concepts` — Apply configured concept filters and record outcomes. (`concept_filters.apply_concept_filters`)

### Standard

- `standard.initial_prompt` — Generate candidate themes/stories. (`prompts.generate_first_prompt`)
- `standard.section_2_choice` — Pick the most compelling choice. (`prompts.generate_second_prompt`)
- `standard.section_2b_title_and_story` — Generate title and story details. (`prompts.generate_secondB_prompt`)
- `standard.section_3_message_focus` — Clarify the message to convey. (`prompts.generate_third_prompt`)
- `standard.section_4_concise_description` — Write the concise detailed description. (`prompts.generate_fourth_prompt`)
- `standard.image_prompt_creation` — Create the final image prompt. (`prompts.generate_image_prompt`)

### Blackbox

- `blackbox.prepare` — Prepare blackbox scoring (novelty summary + default generator hints). (`blackbox_scoring.extract_recent_motif_summary`)  
  Action stage (no LLM call). Sets `generator_profile_hints` and `blackbox_scoring.novelty_summary`, merge: `none`
- `blackbox.profile_abstraction` — Create generator-safe profile hints. (`prompts.profile_abstraction_prompt`)  
  Output: `generator_profile_hints`, merge: `none`, refinement: `none`
- `blackbox.idea_cards_generate` — Generate idea cards (strict JSON). (`prompts.idea_cards_generate_prompt`)  
  Output: `idea_cards_json`, merge: `none`, refinement: `none`
- `blackbox.idea_cards_judge_score` — Judge idea cards and emit scores (strict JSON). (`prompts.idea_cards_judge_prompt`)  
  Output: `idea_scores_json`, merge: `none`, refinement: `none`
- `blackbox.select_idea_card` — Select an idea card using judge scores. (`blackbox_scoring.select_candidate`)  
  Action stage (no LLM call). Sets `selected_idea_card` and updates `blackbox_scoring`, merge: `none`
- `blackbox.image_prompt_creation` — Create final prompt from selected idea card. (`prompts.final_prompt_from_selected_idea_prompt`)

### Refine

- `refine.image_prompt_refine` — Refine a provided draft into the final image prompt. (`prompts.refine_image_prompt_prompt`)
