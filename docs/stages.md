# Prompt Stage Catalog

This file is generated from `image_project.impl.current.prompting.StageCatalog`.

Regenerate with:

```bash
pdm run update-stages-docs
# or
python tools/generate_stages_doc.py
```

Total stages: 22

## ab

- `ab.final_prompt_format` (chat): Format the refined scene into a strict single-line prompt template. `(prompts.ab_final_prompt_format)`
- `ab.final_prompt_format_from_scenespec` (chat): Format a SceneSpec JSON intermediary into the strict single-line prompt template. `(inline)`
- `ab.random_token` (action): Generate a deterministic per-run random token. `(inline)`
- `ab.scene_draft` (chat): Create a scene draft from a random token. `(prompts.ab_scene_draft)`
- `ab.scene_refine_no_block` (chat): Refine the draft scene with a minimal instruction set. `(prompts.ab_scene_refine_no_block)`
- `ab.scene_refine_with_block` (chat): Refine the draft scene with an explicit refinement block. `(prompts.ab_scene_refine_with_block)`
- `ab.scene_spec_json` (chat): Convert the scene draft into a strict SceneSpec JSON intermediary. `(inline)`

## blackbox

- `blackbox.idea_cards_generate` (chat): Generate idea cards (strict JSON). `(prompts.idea_cards_generate_prompt)`
- `blackbox.idea_cards_judge_score` (chat): Judge idea cards and emit scores (strict JSON). `(prompts.idea_cards_judge_prompt)`
- `blackbox.image_prompt_creation` (chat): Create final prompt from selected idea card. `(prompts.final_prompt_from_selected_idea_prompt)`
- `blackbox.prepare` (action): Prepare blackbox scoring (novelty summary + default generator hints). `(blackbox_scoring.extract_recent_motif_summary)`
- `blackbox.profile_abstraction` (chat): Create generator-safe profile hints. `(prompts.profile_abstraction_prompt)`
- `blackbox.select_idea_card` (action): Select an idea card using judge scores (and novelty penalties when enabled). `(blackbox_scoring.select_candidate)`

## preprompt

- `preprompt.filter_concepts` (action): Apply configured concept filters (in order) and record outcomes. `(concept_filters.apply_concept_filters)`
- `preprompt.select_concepts` (action): Select concepts (random/fixed/file) and store them on the run context. `(prompts.select_random_concepts)`

## refine

- `refine.image_prompt_refine` (chat): Refine a provided draft into the final image prompt. `(prompts.refine_image_prompt_prompt)`

## standard

- `standard.image_prompt_creation` (chat): Create the final image prompt. `(prompts.generate_image_prompt)`
- `standard.initial_prompt` (chat): Generate candidate themes/stories. `(prompts.generate_first_prompt)`
- `standard.section_2_choice` (chat): Pick the most compelling choice. `(prompts.generate_second_prompt)`
- `standard.section_2b_title_and_story` (chat): Generate title and story details. `(prompts.generate_secondB_prompt)`
- `standard.section_3_message_focus` (chat): Clarify the message to convey. `(prompts.generate_third_prompt)`
- `standard.section_4_concise_description` (chat): Write the concise detailed description. `(prompts.generate_fourth_prompt)`
