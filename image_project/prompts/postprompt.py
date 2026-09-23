from __future__ import annotations

import textwrap


def refine_image_prompt_prompt(draft: str, *, max_chars: int | None = None) -> str:
    draft_text = (draft or "").strip()
    if not draft_text:
        raise ValueError("Draft image prompt cannot be empty")

    target = int(max_chars) if max_chars is not None else 3500
    return textwrap.dedent(
        f"""\
        Refine the following draft into a high-quality GPT Image 1.5 prompt.

        Goals:
        - Preserve the original intent and key details.
        - Remove contradictions, redundancies, and vague phrasing.
        - Make the prompt concrete and visually grounded.
        - Keep it under {target} characters.

        Output rules:
        - Output ONLY the final image prompt (no analysis, no commentary).
        - Use short labeled sections with line breaks (omit sections that don't apply).

        Draft:
        {draft_text}
        """
    ).strip()


def profile_nudge_image_prompt_prompt(
    *,
    draft_prompt: str,
    preferences_guidance: str,
    context_guidance: str | None = None,
    max_chars: int | None = None,
) -> str:
    draft_text = (draft_prompt or "").strip()
    if not draft_text:
        raise ValueError("Draft image prompt cannot be empty")

    preferences_text = (preferences_guidance or "").strip()
    if not preferences_text:
        raise ValueError("preferences_guidance cannot be empty")

    context_text = (context_guidance or "").strip()
    target = int(max_chars) if max_chars is not None else None
    limit_line = f"- Keep the output under {target} characters.\n" if target is not None else ""

    return textwrap.dedent(
        f"""\
        You will receive:
        - A draft image generation prompt.
        - A raw user profile describing preferences (likes/dislikes/loves/hates).

        Task:
        Make a subtle, non-obvious nudge to the draft prompt so it better matches the user's tastes where possible.

        Hard rules:
        - Preserve the core intent, subject, setting, and key composition. No major changes.
        - Do not introduce new major objects, characters, or story beats.
        - Do not contradict explicit constraints already in the draft prompt (e.g. "no people/no figures/no faces", "no text", "no cityscape").
        - Do not add preference items verbatim and do not copy profile phrases verbatim.
        - Do not make changes that feel shoe-horned, literal, or artificially forced.
        - Respect dislikes by avoiding them; do not add the "opposite" of a dislike unless the scene genuinely requires it.
        {limit_line.strip()}

        Draft prompt:
        {draft_text}

        User profile (authoritative):
        {preferences_text}

        Context guidance (optional):
        {context_text if context_text else "<none>"}

        Output rules:
        - Output ONLY the revised image prompt (no analysis, no commentary).
        - Keep the edits small: think "tighten", "nudge", and "align", not "rewrite".
        - Keep it concrete and renderable.
        """
    ).strip()

