from __future__ import annotations

from typing import Any

from pipelinekit.engine.pipeline import Block, ChatStep
from pipelinekit.engine.patterns import fanout_then_reduce, generate_then_select
from image_project.framework.runtime import RunContext

ENCLAVE_ARTISTS: list[tuple[str, str, str]] = [
    (
        "hemingway",
        "Hemingway",
        "Terse, concrete, and opinionated. Iceberg theory: focus on what matters and cut fluff.",
    ),
    (
        "munch",
        "Munch",
        "Emotion-first expressionist. Prioritize mood, tension, symbolism, and visceral resonance.",
    ),
    (
        "da_vinci",
        "da Vinci",
        "Systems thinker. Tie disparate elements into a cohesive whole with strong composition and purpose. This artist is focused on ensuring all components work together harmoniously. Additionally, this artist pays close attention to ensure that the image is physically achievable (ie conflicting or paradoxical elements that cannot be represented in a single image should be avoided).",
    ),
    (
        "representative",
        "Representative",
        "Audience translator. Optimize for Lana's stated likes/dislikes; remove anything that will annoy her. You have veto authority. Allow the artists creative freedom but never allow clear conflicts with stated dislikes. Clearly state any vetoes in your critique. Clearly label any suggestions that are not vetoes as optional.",
    ),
    (
        "chameleon",
        "Chameleon",
        "Match the specific style and subject matter implied by the draft; sharpen genre conventions and specificity.",
    ),
]

CRITICS_BY_ID: dict[str, tuple[str, str]] = {
    key: (label, persona) for key, label, persona in ENCLAVE_ARTISTS
}


def enclave_thread_prompt(
    label: str,
    persona: str,
    *,
    preferences_guidance: str | None = None,
    first_prompt_random_values: list[str] | None = None,
) -> str:
    reference_sections: list[str] = []
    if preferences_guidance and str(preferences_guidance).strip():
        reference_sections.append(
            "Preferences guidance (authoritative):\n" + str(preferences_guidance).strip()
        )

    if first_prompt_random_values:
        values = [str(value).strip() for value in first_prompt_random_values if str(value).strip()]
        if values:
            reference_sections.append(
                "First-prompt random values (authoritative):\n"
                + "\n".join(f"- {value}" for value in values)
            )

    reference_block = ""
    if reference_sections:
        reference_block = (
            "\n\nReference material below is authoritative even if it is missing from the conversation context.\n\n"
            + "\n\n".join(reference_sections)
            + "\n\n"
        )

    return (
        f"You are {label}.\n"
        f"Persona: {persona}\n"
        + reference_block
        + "\nYou are a single voice.\n"
        "You do NOT see any other artists' feedback.\n"
        "Critique and refine ONLY the last assistant response in this conversation.\n"
        "Do not add meta commentary.\n\n"
        "Return a structured critique with two sections:\n"
        "## Issues\n"
        "- ...\n\n"
        "## Edits\n"
        "- ... (concrete replacements/rewrites)\n"
        "Keep it succinct and focused. No more than 1000 characters."
    )


def make_tot_enclave_block(
    stage_name: str,
    *,
    critics: list[str] | None = None,
    reduce_style: str = "best_of",
    capture_prefix: str | None = None,
    params: dict[str, Any] | None = None,
) -> Block:
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise TypeError("stage_name must be a non-empty string")
    stage_name = stage_name.strip()

    if capture_prefix is None:
        capture_prefix = stage_name
    if not isinstance(capture_prefix, str) or not capture_prefix.strip():
        raise TypeError("capture_prefix must be a non-empty string")
    capture_prefix = capture_prefix.strip()

    style = (reduce_style or "").strip().lower() or "best_of"
    if style not in ("best_of", "consensus"):
        raise ValueError(f"Unknown reduce_style: {reduce_style!r}")

    default_critics = [key for key, _label, _persona in ENCLAVE_ARTISTS]
    critic_ids = list(critics) if critics is not None else default_critics
    critic_ids = [str(item).strip() for item in critic_ids if str(item).strip()]
    if not critic_ids:
        raise ValueError("critics list is empty")

    shared_params = dict(params) if params else {}

    fanout: list[ChatStep] = []
    for artist_key in critic_ids:
        critic = CRITICS_BY_ID.get(artist_key)
        if critic is None:
            available = ", ".join(sorted(CRITICS_BY_ID.keys())) or "<none>"
            raise ValueError(
                f"Unknown critic id: {artist_key!r} (available: {available})"
            )

        label, persona = critic

        capture_key = f"enclave.{capture_prefix}.{artist_key}"
        doc = f"ToT enclave thread ({label}): critique + edits."

        fanout.append(
            ChatStep(
                name=artist_key,
                merge="none",
                capture_key=capture_key,
                prompt=lambda ctx, label=label, persona=persona, artist_key=artist_key: enclave_thread_prompt(
                    label,
                    persona,
                    preferences_guidance=(
                        ctx.outputs.get("preferences_guidance")
                        if artist_key == "representative"
                        else None
                    ),
                    first_prompt_random_values=(
                        ctx.selected_concepts if artist_key == "chameleon" else None
                    ),
                ),
                temperature=0.8,
                params=dict(shared_params),
                meta={
                    "source": "refinement_enclave.enclave_thread_prompt",
                    "doc": doc,
                },
            )
        )

    def consensus_prompt(
        ctx: RunContext, capture_prefix: str = capture_prefix, thread_label: str | None = None
    ) -> str:
        notes: list[str] = []
        for artist_key in critic_ids:
            label = CRITICS_BY_ID[artist_key][0]
            key = f"enclave.{capture_prefix}.{artist_key}"
            value = ctx.outputs.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing enclave thread output: {key}")
            notes.append(f"## {label}\n{value.strip()}")

        consensus_label = "You are the enclave consensus editor."
        if thread_label:
            consensus_label = f"You are the enclave consensus editor (thread {thread_label})."

        return (
            f"{consensus_label}\n"
            "Using the independent artist notes below, revise the last assistant response.\n"
            "Keep the original intent and constraints.\n"
            "Return ONLY the revised response (no preamble, no analysis).\n\n"
            "The Representative has veto authority. If the representative states something must be changed, it should be changed. However, other suggestions it gives don't have to be strictly followed.\n\n"
            + "\n\n".join(notes)
        )

    def final_consensus_prompt(ctx: RunContext, capture_prefix: str = capture_prefix) -> str:
        drafts: list[str] = []
        for idx in range(1, 4):
            key = f"enclave.{capture_prefix}.consensus_{idx}"
            value = ctx.outputs.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing enclave consensus output: {key}")
            drafts.append(f"## Consensus {idx}\n{value.strip()}")

        return (
            "You are the enclave final consensus editor.\n"
            "Review the three independent consensus drafts below and synthesize them into a single refined response.\n"
            "Keep the original intent and constraints. Prefer points of agreement; when opinions diverge, favor clarity, Lana's stated preferences, and the Representative's caution while retaining the Chameleon specificity when it does not conflict.\n"
            "Return ONLY the revised response (no preamble, no analysis).\n\n"
            + "\n\n".join(drafts)
        )

    consensus: Block
    if style == "best_of":
        consensus_drafts: list[ChatStep] = []
        for idx in range(1, 4):
            consensus_doc = f"ToT enclave consensus draft {idx}/3."
            consensus_drafts.append(
                ChatStep(
                    name=f"consensus_{idx}",
                    merge="none",
                    capture_key=f"enclave.{capture_prefix}.consensus_{idx}",
                    prompt=lambda ctx, idx=idx: consensus_prompt(ctx, capture_prefix, f"#{idx}"),
                    temperature=0.8,
                    params=dict(shared_params),
                    meta={
                        "source": "refinement_enclave.consensus_prompt",
                        "doc": consensus_doc,
                    },
                )
            )

        consensus = generate_then_select(
            name="consensus",
            generate=consensus_drafts,
            select=ChatStep(
                name="final_consensus",
                prompt=final_consensus_prompt,
                temperature=0.8,
                params=dict(shared_params),
                meta={
                    "source": "refinement_enclave.final_consensus_prompt",
                    "doc": "ToT enclave final consensus synthesis.",
                },
            ),
        )
    else:
        consensus = Block(
            name="consensus",
            merge="all_messages",
            nodes=[
                ChatStep(
                    name="final_consensus",
                    prompt=lambda ctx: consensus_prompt(ctx, capture_prefix, None),
                    temperature=0.8,
                    params=dict(shared_params),
                    meta={
                        "source": "refinement_enclave.consensus_prompt",
                        "doc": "ToT enclave consensus refinement.",
                    },
                )
            ],
        )

    return fanout_then_reduce(name="tot_enclave", fanout=fanout, reduce=[consensus])
