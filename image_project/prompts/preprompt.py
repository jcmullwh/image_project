from __future__ import annotations

import random
import textwrap

import pandas as pd

DEFAULT_SYSTEM_PROMPT = (
    "You are a highly skilled enclave of Artists trained to generate meaningful, edgy, artistic images on par "
    "with the greatest artists of any time, anywhere, past or future, Earth or any other planet. The enclave "
    "invents unique images that weave together seemingly disparate elements into cohesive wholes that push "
    "boundaries. Keep your responses concise and focused on the task at hand."
)

CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Subject Matter", "Narrative"),
    ("Mood", "Composition", "Perspective"),
    ("Style", "Time Period_Context", "Color Scheme"),
)


def build_preferences_guidance(user_profile: pd.DataFrame) -> str:
    if user_profile is None or user_profile.empty:
        return ""

    sections: list[str] = []
    columns = [str(column).strip() for column in user_profile.columns]
    if "Loves" in columns or "Hates" in columns:
        sections.append(
            textwrap.dedent(
                """\
                Preference strength legend:
                - Loves: will, by itself, make the user like the image.
                - Likes: shows a preference for but not strongly (feedback may be slightly conflicting or not absolute).
                - Dislikes: shows a dislike for but not strongly (feedback may be slightly conflicting or not absolute).
                - Hates: will, by itself, make the user dislike the image.
                """
            ).strip()
        )

    sections.append(
        textwrap.dedent(
            """\
            Preference interpretation rules:
            - Loves/Likes are positive guidance. Do not copy them verbatim; apply them as subtle direction.
            - Hates/Dislikes are avoid constraints. Do not "satisfy" a dislike by introducing the disliked thing with a positive modifier.
              Example: if a profile says it dislikes "wrong/incorrect X", that does NOT mean adding "correct X" improves the prompt.
              Instead, avoid introducing X unnecessarily; if X is already required by the scene, keep it natural, de-emphasized, and non-central (and do NOT call it out as "correct X").
            """
        ).strip()
    )
    for column in user_profile.columns:
        values = [str(value).strip() for value in user_profile[column].dropna().tolist()]
        values = [value for value in values if value]
        if values:
            sections.append(f"{column}:\n" + "\n".join(f"- {value}" for value in values))

    return "\n\n".join(sections).strip()


def load_prompt_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def get_random_value_from_group(
    group: tuple[str, ...], data: pd.DataFrame, rng: random.Random
) -> str | None:
    combined_values: list[str] = []
    for column in group:
        column_values = data[column].dropna().tolist()
        combined_values.extend(str(value) for value in column_values)
    return rng.choice(combined_values) if combined_values else None


def select_random_concepts(prompt_data: pd.DataFrame, rng: random.Random) -> list[str]:
    selections: list[str] = []
    for group in CONCEPT_GROUPS:
        value = get_random_value_from_group(group, prompt_data, rng)
        if value:
            selections.append(value)
    return selections

