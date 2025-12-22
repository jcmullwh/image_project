Feedback 17) "#023 Pier Of Unlived Journeys" (awkward couple in middle/no legs; left/right composition again; prefers landscape-only)

Run facts (from logs)
- Generation: `20251221_145809_8e67958e-62c5-4af6-83ec-8c19f4d8e757` (`#023 - Pier Of Unlived Journeys`)
- Selected concepts: `Microcosms/Miniature Worlds`, `Idyllic/Pastoral`, `Age of Discovery`
- Context injectors: season + holiday applied (winter + Christmas)
- Note: this run loaded `prompt_files/user_profile_v1.csv` (current `config/config.yaml` points to `prompt_files/user_profile_v3.csv`)
- Final prompt signature (`pipeline/image_prompt_creation/tot_enclave/final_consensus`): a perfectly centered long pier; couple mid-pier holding a glowing glass sphere containing miniature harbors; symmetric rails/lanterns; city and ship aligned behind them; strong straight-on symmetry

How strong was the feedback? Should we take it as a nudge or a strong change?

Strong negative. It includes multiple concrete failure modes (awkward pasted couple, anatomy/cropping error, recurring composition pattern, and a clear preference statement: "I'd rather have nobody in the picture at all").

What are the ways that the feedback could be interpreted? What are the different things that it might be referring to?

1) Composition pattern aversion (not just "two halves")
- Even without an explicit split-scene, strong central symmetry can read as "two sides again" because the frame is dominated by mirrored left/right rails and lighting.

2) Human-figure quality risk
- Mid-distance figures often produce cut-off limbs; if the user is sensitive to this, figure presence becomes high-risk unless the prompt strongly constrains full-body visibility.

3) Preference shift: landscape-first mode
- The user is saying that in some cases they would prefer a pure environment/landscape without people, rather than "a couple who doesn't look like us."

4) "Random people" concern persists despite intent
- Even if the prompt describes the intended couple, the model may still render faces/bodies that do not feel like them; the user experiences that as "random strangers inserted."

Then: how were those things driven by the prompt and where in the process did that component of the prompt enter? Was it a specific step?

- Strong central symmetry is explicitly selected in `pipeline/section_2_choice/...` (first appearance of "central symmetry" language) and then reinforced in the final prompt ("wide shot, straight-on perspective... strong central symmetry").
- The microcosm "glass sphere with miniature harbors" is introduced later in `pipeline/section_3_message_focus/...` and carried into the final image prompt as the hero prop.
- The couple is mandated as the central axis anchor throughout, culminating in `pipeline/image_prompt_creation/tot_enclave/final_consensus` with the couple placed mid-pier on the centerline.

Then: Why was it included? A specific prompt output suggested it? A specific point? What was the reasoning?

- `Age of Discovery` pushed toward a pier/ship threshold narrative.
- `Microcosms/Miniature Worlds` was implemented as a literal "miniature worlds in a glass sphere" hero prop, because it's a crisp visual gimmick.
- The pipeline's "clear story" bias selected a highly legible, poster-like composition (central axis, single bright object in hands) to ensure readability.

Then: What does that all tell us in general terms about the flow and prompts?

- The system sometimes equates "classic/timeless" with "centered symmetry + couple centered," but that can reproduce the user's disliked left/right framing pattern in a different form.
- If we insist on people in every scene, we expose ourselves to high-variance failure (faces/limbs/likeness) and we may fight the user's desire to sometimes simply enjoy the landscape.

Then: What are some ways to incorporate the feedback without overfitting to this specific item?

Low-blast-radius (safe)
- Add a "landscape-first" option that allows zero people (or tiny silhouettes) for scenes where environment is the hero.
- Add a validator: if a prompt places the couple "centered" on a strong axis (pier/bridge seam), rewrite to a diagonal/curved composition (pier entering from a corner, S-curve path, off-center bridge).
- If people are included, explicitly require full-body visibility (feet on ground, legs visible, no cropping) and reduce "posed holding a glowing object" staging.

Concept-scoped (microcosms)
- Express miniature-world wonder through diegetic architecture (lighthouse lens, observatory, canal reflections) rather than a floating/glowing hand-held orb that can read as symbolic theater.

Then: What would it be most appropriate and effective? Remember that we need to use judgement.

Most appropriate: treat this as a strong signal to (a) stop defaulting to centered, symmetric "couple as the axis" compositions for this user, and (b) allow environment-only images when the scene is fundamentally about place. That reduces both the recurring composition complaint and the high-risk anatomy/likeness problems without overfitting to "this pier" specifically.

