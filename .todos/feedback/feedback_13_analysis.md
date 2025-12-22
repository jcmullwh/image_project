Feedback 13) "#024 Harbor Mirror Hearth" (random broken cup/phone/coins)

Run facts (from logs)
- Generation: `20251221_160614_e1cec6ef-e46c-4195-ac5d-169d52f0aab0` (`#024 - Harbor Mirror Hearth`)
- Selected concepts: `Mirror Reflections/Double Exposure`, `Reflective Perspective`, `Playful Neo-Dada Whimsy` (rewritten from `Dadaism`)
- Context injectors: season + holiday applied (winter + Christmas)
- Note: this run loaded `prompt_files/user_profile_v1.csv` (current `config/config.yaml` points to `prompt_files/user_profile_v3.csv`)
- Final prompt signature (`pipeline/image_prompt_creation/tot_enclave/final_consensus`): winter-night harbor with a couple on a small boat; a visibly cracked steaming mug and a smartphone on the deck; the water reflection rebuilds the skyline as a "Neo-Dada object-city" made from everyday items (cups/bowls/tickets/notebooks/cables/coins)

How strong was the feedback? Should we take it as a nudge or a strong change?

Strong negative and high-confidence. The user is not critiquing taste-level nuance; they are flagging core incoherence ("makes no sense") and object choices ("broken cup", "cell phone", "random coins") that directly conflict with their stated dislike of random disconnected elements and broken/dirty cues.

What are the ways that the feedback could be interpreted? What are the different things that it might be referring to?

1) Diegetic plausibility failure
- Objects (coins, phone, broken mug) read like staged props in a place they do not belong (on open water / on a lake).

2) Symbolism fatigue / "random symbolism pile-ups"
- Even if the objects are meant as metaphor (warmth vs distraction), the user experiences them as arbitrary rather than emotionally motivated.

3) Cleanliness/maintenance preference violation
- A cracked mug is a "broken item" cue; for this user it clusters with grit/cracks/patina and reads as depressing or dirty-adjacent even if framed as "fragile/beautiful."

4) Surrealism tolerance boundary
- They may like surreal *lighting* or *architecture*, but not surreal "found-object collage" where everyday clutter becomes the subject.

Then: how were those things driven by the prompt and where in the process did that component of the prompt enter? Was it a specific step?

It was prompted repeatedly and early:
- `pipeline/initial_prompt/draft` already introduces Neo-Dada "clusters of everyday objects" (tickets/keys/phone/cracked mug) and a harbor reflection where buildings morph into stacked everyday items.
- `pipeline/section_2b_title_and_story/draft` is where the reflection is explicitly framed as an object-city and where "coins as windows" style details start appearing.
- `pipeline/section_4_concise_description/draft` concretizes the core prop pairing (cracked mug vs smartphone) as the emotional mechanism of the scene.
- `pipeline/image_prompt_creation/tot_enclave/final_consensus` locks those props in as required foreground details.

Then: Why was it included? A specific prompt output suggested it? A specific point? What was the reasoning?

Primary drivers
- Concept mapping: `Playful Neo-Dada Whimsy` was implemented literally as "everyday objects arranged into a city" rather than as a stylistic sensibility.
- Mirror/reflection concepts (`Mirror Reflections/Double Exposure`, `Reflective Perspective`) provided the perfect excuse to split the world into "real harbor" + "surreal object-harbor," making the object pile-up feel justifiable to the prompt writer.
- Mechanism-driven storytelling was expressed as a prop-based moral: mug = warmth/vulnerability, phone = numb distraction, with visible "effects" in the reflection.

Secondary drivers
- The run used `user_profile_v1.csv`, which does not contain the stronger v3 guardrails like "clean, polished materials" and "single meaningful symbol only if diegetic" and explicit bans on "random staged symbolism pile-ups."

Then: What does that all tell us in general terms about the flow and prompts?

- When "surreal everyday objects" are allowed as a literal content channel, the pipeline happily escalates them into foreground props and large-scale structures, which reads as incoherent to this user.
- The pipeline is currently good at inventing a coherent metaphor, but not good at checking whether the metaphor is *acceptable* for the user (diegetic, clean, non-cluttered).

Then: What are some ways to incorporate the feedback without overfitting to this specific item?

Low-blast-radius (safe)
- Add a content validator that rejects/rewrites prompts containing "cracked/broken/chipped" household items and modern clutter props (phone/coins/receipts/keys) unless the user explicitly asked for them.
- For Dada/surreal concepts, constrain the implementation to: (a) playful geometry/patterning, (b) unexpected *architecture* materials, or (c) a single diegetic object that the couple is visibly using (lantern, map, telescope) rather than a scattered pile.

Concept-scoped (only when Dada-like concepts win)
- Replace "everyday object collage" with "Neo-Dada composition" expressed through signage-shape abstraction, odd juxtapositions of architectural forms, or whimsical but plausible harbor objects (buoys, ropes, lanterns, boats) kept tidy and integrated.

Then: What would it be most appropriate and effective? Remember that we need to use judgement.

Most appropriate: treat this as a strong, portable rule: for this user, surrealism should be environmental and luminous, not prop-pile or broken-object symbolism. The most effective fix is a rewrite/guardrail stage (either in concept filtering or final prompt validation) that forbids random household-object staging and forbids "broken/cracked" hero props, while still allowing reflective dual-perspective via water, architecture, and light.

