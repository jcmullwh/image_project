from __future__ import annotations

import json
import re
import textwrap
from typing import Any


def build_dislike_rewrite_messages(
    *,
    selected_concepts: list[str],
    dislikes: list[str],
) -> list[dict[str, str]]:
    prompt = textwrap.dedent(
        f"""\
        Selected concepts (keep length and order):
        {json.dumps(list(selected_concepts), ensure_ascii=False)}

        User dislikes (avoid conflicts):
        {json.dumps(list(dislikes), ensure_ascii=False)}

        If any selected concept conflicts with a dislike, rewrite just that concept so it no longer conflicts but still fits the original creative intent and variety. If there is no conflict, keep the concept unchanged.

        Return ONLY a JSON array of the revised concepts (strings), same length and order as provided. Do not add commentary or keys.
        """
    ).strip()

    return [
        {
            "role": "system",
            "content": (
                "You rewrite selected creative concepts so none of them conflict with the user's dislikes. "
                "Keep variety, keep count, and avoid over-censoring."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_concept_list_response(response: Any) -> list[str]:
    """
    Try to coerce the model response into a clean list of concept strings.
    """
    if not isinstance(response, str):
        return []

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()

    candidates = [cleaned]

    bracket_match = re.search(r"\\[.*\\]", cleaned, flags=re.DOTALL)
    if bracket_match:
        candidates.insert(0, bracket_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if isinstance(parsed, list):
            coerced = [str(item).strip() for item in parsed if str(item).strip()]
            if coerced:
                return coerced

    return []

