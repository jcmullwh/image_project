from __future__ import annotations

import textwrap
from typing import Any, Callable

from image_project.foundation.pipeline import Block, ChatStep, MergeMode
from image_project.framework.runtime import RunContext


class RefinementPolicy:
    def stage(
        self,
        stage_name: str,
        *,
        prompt: str | Callable[[RunContext], str],
        temperature: float,
        merge: MergeMode = "last_response",
        allow_empty_prompt: bool = False,
        allow_empty_response: bool = False,
        params: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        capture_key: str | None = None,
    ) -> Block:
        raise NotImplementedError

    def _validated_stage_name(self, stage_name: str) -> str:
        if not isinstance(stage_name, str):
            raise TypeError(f"Stage name must be a string (type={type(stage_name).__name__})")
        trimmed = stage_name.strip()
        if not trimmed:
            raise ValueError("Stage name cannot be empty")
        return trimmed

    def _validated_prompt(self, prompt: str | Callable[[RunContext], str]) -> str | Callable[[RunContext], str]:
        if isinstance(prompt, str) or callable(prompt):
            return prompt
        raise TypeError(
            f"Stage prompt must be a string or callable (type={type(prompt).__name__})"
        )

    def _validated_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise TypeError(
                f"Stage params must be a dict if provided (type={type(params).__name__})"
            )
        return dict(params)


class NoRefinement(RefinementPolicy):
    def stage(
        self,
        stage_name: str,
        *,
        prompt: str | Callable[[RunContext], str],
        temperature: float,
        merge: MergeMode = "last_response",
        allow_empty_prompt: bool = False,
        allow_empty_response: bool = False,
        params: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        capture_key: str | None = None,
    ) -> Block:
        stage_name = self._validated_stage_name(stage_name)
        prompt = self._validated_prompt(prompt)
        params = self._validated_params(params)
        stage_meta = dict(meta) if meta else {}

        draft_step = ChatStep(
            name="draft",
            prompt=prompt,
            temperature=temperature,
            allow_empty_prompt=allow_empty_prompt,
            allow_empty_response=allow_empty_response,
            params=params,
        )
        return Block(
            name=stage_name,
            merge=merge,
            nodes=[draft_step],
            capture_key=capture_key,
            meta=stage_meta,
        )


class TotEnclaveRefinement(RefinementPolicy):
    def stage(
        self,
        stage_name: str,
        *,
        prompt: str | Callable[[RunContext], str],
        temperature: float,
        merge: MergeMode = "last_response",
        allow_empty_prompt: bool = False,
        allow_empty_response: bool = False,
        params: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        capture_key: str | None = None,
    ) -> Block:
        stage_name = self._validated_stage_name(stage_name)
        prompt = self._validated_prompt(prompt)
        params = self._validated_params(params)
        stage_meta = dict(meta) if meta else {}

        draft_step = ChatStep(
            name="draft",
            prompt=prompt,
            temperature=temperature,
            allow_empty_prompt=allow_empty_prompt,
            allow_empty_response=allow_empty_response,
            params=params,
        )
        enclave_block = make_tot_enclave_block(stage_name, params=params)
        return Block(
            name=stage_name,
            merge=merge,
            nodes=[draft_step, enclave_block],
            capture_key=capture_key,
            meta=stage_meta,
        )


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


def make_tot_enclave_block(stage_name: str, *, params: dict[str, Any] | None = None) -> Block:
    nodes: list[Any] = []
    shared_params = dict(params) if params else {}

    for artist_key, label, persona in ENCLAVE_ARTISTS:
        capture_key = f"enclave.{stage_name}.{artist_key}"
        doc = f"ToT enclave thread ({label}): critique + edits."

        nodes.append(
            ChatStep(
                name=artist_key,
                merge="none",
                capture_key=capture_key,
                prompt=lambda ctx, label=label, persona=persona, artist_key=artist_key: enclave_thread_prompt(
                    label,
                    persona,
                    preferences_guidance=(
                        ctx.outputs.get("preferences_guidance") if artist_key == "representative" else None
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
        ctx: RunContext, stage_name: str = stage_name, thread_label: str | None = None
    ) -> str:
        notes: list[str] = []
        for artist_key, label, _persona in ENCLAVE_ARTISTS:
            key = f"enclave.{stage_name}.{artist_key}"
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

    def final_consensus_prompt(ctx: RunContext, stage_name: str = stage_name) -> str:
        drafts: list[str] = []
        for idx in range(1, 4):
            key = f"enclave.{stage_name}.consensus_{idx}"
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

    for idx in range(1, 4):
        consensus_doc = f"ToT enclave consensus draft {idx}/3."
        nodes.append(
            ChatStep(
                name=f"consensus_{idx}",
                merge="none",
                capture_key=f"enclave.{stage_name}.consensus_{idx}",
                prompt=lambda ctx, idx=idx, stage_name=stage_name: consensus_prompt(
                    ctx, stage_name, f"#{idx}"
                ),
                temperature=0.8,
                params=dict(shared_params),
                meta={
                    "source": "refinement_enclave.consensus_prompt",
                    "doc": consensus_doc,
                },
            )
        )

    nodes.append(
        ChatStep(
            name="final_consensus",
            prompt=final_consensus_prompt,
            temperature=0.8,
            params=dict(shared_params),
            meta={
                "source": "refinement_enclave.final_consensus_prompt",
                "doc": "ToT enclave final consensus synthesis.",
            },
        )
    )

    return Block(name="tot_enclave", merge="all_messages", nodes=nodes)
