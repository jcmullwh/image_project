from __future__ import annotations

from collections.abc import Sequence


def build_title_messages(
    *,
    image_prompt: str,
    avoid_titles: Sequence[str] | None,
    attempt: int,
) -> list[dict[str, str]]:
    system = (
        "You generate short image titles.\n"
        "Return a 2-4 word Title Case name.\n"
        "No quotes.\n"
        "No proper nouns (no people/place/brand names).\n"
        "No punctuation except optional hyphen.\n"
        "Return only the title text.\n"
    )

    avoid_titles_list = list(avoid_titles or [])
    avoid_block = ""
    if avoid_titles_list:
        avoid_block = "Do not reuse these titles (case-insensitive): " + "; ".join(
            avoid_titles_list[:25]
        )

    user = (
        f"Prompt: {str(image_prompt).strip()}\n"
        f"{avoid_block}\n"
        "Return ONLY the title text."
    ).strip()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if int(attempt) > 1:
        messages.append(
            {
                "role": "system",
                "content": "Reminder: output must be a single line containing only the title text.",
            }
        )
    return messages

