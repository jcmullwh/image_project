# Prompt Stage Catalog

This file is generated from `image_project.stages.registry.get_stage_registry()`.

Regenerate with:

```bash
pdm run update-stages-docs
# or
python tools/generate_stages_doc.py
```

Total stages: 38

## ab

- `ab.final_prompt_format` (chat): Format the refined scene into a strict single-line prompt template. `(prompts.ab_final_prompt_format)`
- `ab.final_prompt_format_from_scenespec` (chat): Format a SceneSpec JSON intermediary into the strict single-line prompt template. `(inline)`
- `ab.random_token` (action): Generate a deterministic per-run random token. `(inline)` [io: provides=ab_random_token; captures=ab_random_token]
- `ab.scene_draft` (chat): Create a scene draft from a random token. `(prompts.ab_scene_draft)` [io: requires=ab_random_token; provides=ab_scene_draft; captures=ab_scene_draft]
- `ab.scene_refine_no_block` (chat): Refine the draft scene with a minimal instruction set. `(prompts.ab_scene_refine_no_block)` [io: requires=ab_random_token, ab_scene_draft; provides=ab_scene_refined; captures=ab_scene_refined]
- `ab.scene_refine_with_block` (chat): Refine the draft scene with an explicit refinement block. `(prompts.ab_scene_refine_with_block)` [io: requires=ab_random_token, ab_scene_draft; provides=ab_scene_refined; captures=ab_scene_refined]
- `ab.scene_spec_json` (chat): Convert the scene draft into a strict SceneSpec JSON intermediary. `(inline)` [io: requires=ab_random_token, ab_scene_draft; provides=ab_scene_spec_json; captures=ab_scene_spec_json]

## blackbox

- `blackbox.generate_idea_cards` (composite): Generate N isolated idea cards then assemble into idea_cards_json. `(stages.blackbox.generate_idea_cards._build)` [io: provides=idea_cards_json; captures=idea_cards_json]
- `blackbox.generator_profile_hints` (composite): Produce generator_profile_hints from raw/profile file/abstraction. `(stages.blackbox.generator_profile_hints._build)` [io: provides=generator_profile_hints; captures=generator_profile_hints]
- `blackbox.idea_card_generate` (chat): Generate a single idea card (strict JSON) as an isolated stage. `(prompts.blackbox.idea_card_generate_prompt)` [io: requires=selected_concepts]
- `blackbox.idea_cards_assemble` (action): Assemble isolated per-idea JSON artifacts into idea_cards_json. `(framework.scoring.parse_idea_card_json)` [io: provides=idea_cards_json; captures=idea_cards_json]
- `blackbox.idea_cards_generate` (chat): Generate idea cards (strict JSON). `(prompts.blackbox.idea_cards_generate_prompt)` [io: provides=idea_cards_json; captures=idea_cards_json]
- `blackbox.idea_cards_judge_score` (chat): Judge idea cards and emit scores (strict JSON). `(prompts.blackbox.idea_cards_judge_prompt)` [io: requires=idea_cards_json; provides=idea_scores_json; captures=idea_scores_json]
- `blackbox.image_prompt_creation` (chat): Create final prompt from selected idea card. `(prompts.blackbox.final_prompt_from_selected_idea_prompt)` [io: requires=selected_idea_card]
- `blackbox.image_prompt_draft` (chat): Create a draft prompt from selected idea card (for downstream refinement). `(prompts.blackbox.draft_prompt_from_selected_idea_prompt)` [io: requires=selected_idea_card; provides=blackbox_draft_image_prompt; captures=blackbox_draft_image_prompt]
- `blackbox.image_prompt_openai` (chat): Create an OpenAI (GPT Image 1.5) formatted prompt from the selected idea card. `(prompts.blackbox.openai_image_prompt_from_selected_idea_prompt)`
- `blackbox.image_prompt_refine` (chat): Refine the draft prompt into a final prompt (no ToT). `(prompts.blackbox.refine_draft_prompt_from_selected_idea_prompt)` [io: requires=selected_idea_card, blackbox_draft_image_prompt]
- `blackbox.prepare` (action): Prepare blackbox scoring (novelty summary + default generator hints). `(blackbox_scoring.extract_recent_motif_summary)` [io: provides=generator_profile_hints]
- `blackbox.profile_abstraction` (chat): Create generator-safe profile hints. `(prompts.blackbox.profile_abstraction_prompt)` [io: provides=generator_profile_hints; captures=generator_profile_hints]
- `blackbox.profile_hints_load` (action): Load generator-safe profile hints from a file. `(framework.profile_io.load_generator_profile_hints)` [io: provides=generator_profile_hints; captures=generator_profile_hints]
- `blackbox.select_idea_card` (action): Select an idea card using judge scores (and novelty penalties when enabled). `(blackbox_scoring.select_candidate)` [io: requires=idea_cards_json, idea_scores_json; provides=selected_idea_card]

## blackbox_refine

- `blackbox_refine.loop` (composite): Run the blackbox refinement loop (init + N iterations + finalize). `(stages.blackbox_refine.loop._build_blackbox_refine_loop)` [io: requires=bbref.seed_prompt; provides=bbref.beams]
- `blackbox_refine.seed_from_draft` (action): Seed the blackbox refinement loop from prompt.refine_only.draft. `(stages.blackbox_refine.seed_from_draft._build)` [io: provides=bbref.seed_prompt; captures=bbref.seed_prompt]
- `blackbox_refine.seed_prompt` (chat): Generate the seed prompt for the blackbox refinement loop. `(prompts.blackbox.final_prompt_from_selected_idea_prompt)` [io: requires=selected_idea_card; provides=bbref.seed_prompt; captures=bbref.seed_prompt]

## direct

- `direct.image_prompt_creation` (chat): Create final prompt directly from concepts + profile. `(prompts.blackbox.final_prompt_from_concepts_and_profile_prompt)` [io: requires=selected_concepts]

## postprompt

- `postprompt.openai_format` (chat): Format the (nudged) prompt into OpenAI GPT Image 1.5 prompt text. `(prompts.postprompt.refine_image_prompt_prompt)`
- `postprompt.profile_nudge` (chat): Nudge the latest image prompt toward the user profile (small changes only). `(prompts.postprompt.profile_nudge_image_prompt_prompt)` [io: provides=postprompt.nudged_prompt; captures=postprompt.nudged_prompt]

## preprompt

- `preprompt.filter_concepts` (action): Apply configured concept filters (in order) and record outcomes. `(concept_filters.apply_concept_filters)` [io: requires=selected_concepts; provides=selected_concepts, concept_filter_log]
- `preprompt.select_concepts` (action): Select concepts (random/fixed/file) and store them on the run context. `(prompts.preprompt.select_random_concepts)` [io: provides=selected_concepts]

## refine

- `refine.image_prompt_refine` (chat): Refine a provided draft into the final image prompt. `(prompts.postprompt.refine_image_prompt_prompt)`
- `refine.tot_enclave` (composite): Refine the latest assistant output via a ToT enclave critique+consensus pass. `(stages.refine.tot_enclave_prompts.make_tot_enclave_block)`

## standard

- `standard.image_prompt_creation` (chat): Create the final image prompt. `(prompts.standard.generate_image_prompt)`
- `standard.initial_prompt` (chat): Generate candidate themes/stories. `(prompts.standard.generate_first_prompt)` [io: requires=selected_concepts]
- `standard.initial_prompt_freeform` (chat): Generate candidate themes/stories without concept selection (freeform). `(prompts.standard.standard_initial_prompt_freeform_prompt)`
- `standard.section_2_choice` (chat): Pick the most compelling choice. `(prompts.standard.generate_second_prompt)`
- `standard.section_2b_title_and_story` (chat): Generate title and story details. `(prompts.standard.generate_secondB_prompt)`
- `standard.section_3_message_focus` (chat): Clarify the message to convey. `(prompts.standard.generate_third_prompt)`
- `standard.section_4_concise_description` (chat): Write the concise detailed description. `(prompts.standard.generate_fourth_prompt)`
