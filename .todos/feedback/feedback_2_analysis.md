Feedback 2)

Analysis

Run facts (from logs)
- Generation: `20251220_200934_fdf3c776-76a9-4a5d-8fc7-c239da8b5839` (`#017 - Cracked Ornament Runway`)
- Selected concepts: Fashion/Beauty, Sequential Composition, Neo-noir
- Context injectors: winter + Christmas (primary theme)

How strong was the feedback? Nudge or strong change?

User: "this looks like an ad for lipstick. I don't like that."
- Treat as a strong negative against "commercial product / cosmetics-ad" aesthetics.
- Ambiguity: it may be rejecting lipstick specifically, or rejecting product-as-hero + editorial/photoreal framing; assume the broader "ad vibe" is the primary issue until we see more examples.

What could the feedback be referring to? (possible interpretations)

1) Product hero / packshot framing
- A single object is centered, oversized, glossy, and spotlighted; the scene reads like a commercial.
2) Cosmetics content is unwanted
- Even without logos, a lipstick/compact can feel like marketing.
3) Photoreal fashion-editorial style
- "Fashion editorial / cinematic / 35mm" language often pushes the image model toward a real-photo look (already in dislikes).
4) Story focus drift
- The couple becomes secondary, and the narrative turns into "show the product" instead of "show us".

How were those outcomes driven by the prompt, and where did they enter? (pipeline provenance)

- Seed pressure (concept selection):
  - "Fashion/Beauty" includes "makeup" in the category description, making cosmetics a high-probability symbol.
- Prompt refinement (first major injection point):
  - `pipeline/initial_prompt/tot_enclave/hemingway` suggested: "The lipstick is a signal."
  - `pipeline/initial_prompt/tot_enclave/consensus` adopted lipstick/compact mechanics into the selected story.
- Prompt hardening (locks the prop in):
  - `pipeline/section_4_concise_description/tot_enclave/consensus` specifies "a lipstick bullet nestled inside" the central ornament.
- Final image prompt (locks the composition + genre):
  - `pipeline/image_prompt_creation/draft` already framed the deliverable as "cinematic fashion-editorial...", so the editorial/photo bias was present before the persona refinements.
  - `pipeline/image_prompt_creation/tot_enclave/consensus` makes the ornament-compact the "primary focal point at center" and uses fashion-editorial / 35mm cues.
  - Evidence excerpt from that step: "Dead center under the arch ... a lipstick bullet nested inside ... ornament-compact is the primary focal point at center."

What were the drivers? (specific points that pushed us there)

- Seed semantics: Fashion/Beauty + "makeup" makes lipstick/compact an easy, high-signal token.
- Instruction pressure: "say something with conviction" + "not milquetoast" encourages a single iconic prop; cosmetics packaging is a common shortcut.
- Theme blending: "Christmas must be dominant" + "beauty/makeup" often merges into ornament-as-compact / ornament-containing-lipstick.
- Composition language: "dead center" + "primary focal point" + reflective materials yields product-photography framing.
- Style conflict: even with "not photoreal," terms like "fashion editorial", "cinematic", "35mm feel" push photo realism.
- Missing negative constraints: we banned logos/text, but never banned "advertisement / packshot / cosmetics packaging hero."

What does this tell us in general terms about the flow and prompts?

A) High-salience tokens introduced early survive to the end
Once lipstick/compact enters the narrative stages, later stages tend to preserve and amplify it rather than remove it.

B) "No logos/text" does not prevent ad aesthetics
A brandless product shot is still an ad in feel; we need to constrain framing, not just branding.

C) Style cues must be internally consistent
Mixing "fashion editorial / cinematic / lens language" with "not photoreal" often produces photoreal anyway; the image model follows the strongest genre cues.

D) Symmetry + centered focal point is not the problem; product-as-focal-point is
Lana likes symmetry; the risk is when the centered element is a consumer product rendered with commercial lighting.

What we learned (portable rules)

1) When Fashion/Beauty is selected, default to styling + craft (not cosmetics product imagery)
Favor wardrobe, textiles, silhouettes, set design, and character styling. If makeup appears, treat it as a color/finish choice on the character - not as a featured product or packaging.

2) Avoid packshot composition
If a single object is described as "dead center" + "primary focal point" with glossy reflections and spotlights, the model will trend toward commercial product photography.

3) If you need a "signal/clue", use non-product artifacts
Prefer garment tags, stitched symbols, brooches, ribbons, ornament etchings, constellation-like runway lights, folded notes, or an environmental "map" motif. Avoid lipstick-as-signal / microfilm-in-lipstick / compact-mirror packaging language.

4) "Not photoreal" requires explicit anti-photo language
Avoid "fashion editorial", "cinematic", and camera/lens callouts unless we actually want photo realism. Add explicit negatives like "no studio product photography", "no commercial lighting", "no catalog/packshot look."

5) Focal-point guardrail
Primary focal point should be Lana + Jason and their interaction. Props are secondary and must not dominate the frame like a featured product.

What are some ways to incorporate the feedback without overfitting to this specific item?

Low-blast-radius (applies broadly, likely safe for this user)
- Add a general dislike/guardrail: "no advertisements, no product shots/packshots, no commercial marketing vibe."
- Add a style guardrail: avoid "fashion editorial / cinematic / 35mm / DSLR" unless the user requests photoreal.

Concept-scoped (only when Fashion/Beauty is selected)
- Rewrite Fashion/Beauty internally as: wardrobe + textiles + silhouettes + runway/set design; makeup only as a look (color accents), never as a prop.
- Prefer non-product "signal" props: stitched tag, brooch, ribbon code, ornament etching, light-path motif.

Prompt-output-scoped (only when triggers appear)
- If the final prompt contains cosmetics-product words (lipstick/compact/mascara/etc) AND "center/primary focal point", auto-rewrite:
  - swap the prop to a non-product artifact, and/or
  - demote it to secondary detail, keeping Lana+Jason as the focal point.

What would be most appropriate and effective? (judgment + blast radius)

- Treat "avoid ad/packshot/commercial vibe" as a strong general rule for this user (it aligns with the existing dislike of realistic photographic style).
- Treat "ban cosmetics props" as targeted: default to none unless requested; especially avoid cosmetics as the hero object.
- Highest-leverage entry points are the last two stages (`section_4_concise_description` + `image_prompt_creation`), because that's where prop prominence and genre language get finalized.
- If we see this feedback repeat across multiple runs (especially when Fashion/Beauty is selected), then consider excluding that concept for this user rather than continuously rewriting around it.
