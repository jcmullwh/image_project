from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import pandas as pd


_CATEGORY_CANONICAL = {
    "love": "Loves",
    "loves": "Loves",
    "like": "Likes",
    "likes": "Likes",
    "dislike": "Dislikes",
    "dislikes": "Dislikes",
    "hate": "Hates",
    "hates": "Hates",
    "note": "Notes",
    "notes": "Notes",
}

_PREFERRED_COLUMN_ORDER = ("Loves", "Likes", "Dislikes", "Hates", "Notes")


def load_user_profile(path: str) -> pd.DataFrame:
    """
    Load a user profile CSV and normalize it into a preferences DataFrame.

    Supported formats:
    - Column-based: Likes/Dislikes/(Notes/...) columns (v4 style).
    - Row-based: category,item,notes rows (v5 style, including love/like/dislike/hate).
    """

    profile_path = str(path or "").strip()
    if not profile_path:
        raise ValueError("User profile path is required")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"User profile not found: {profile_path}")
    if os.path.isdir(profile_path):
        raise ValueError(f"User profile path is a directory: {profile_path}")
    if os.path.getsize(profile_path) == 0:
        raise ValueError(f"User profile file is empty: {profile_path}")

    df = pd.read_csv(profile_path)
    return normalize_user_profile_df(df)


def normalize_user_profile_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    columns = [str(col).strip() for col in df.columns]
    column_map = {col.lower(): col for col in columns}

    if "item" in column_map:
        if "category" in column_map:
            return _normalize_row_based_profile(df, column_map=column_map)

        # v5 profiles may label the strength/category column differently.
        # Support: strength,item,(conditionals|notes)
        if "strength" in column_map:
            mapped = dict(column_map)
            mapped["category"] = column_map["strength"]
            if "notes" not in mapped and "conditionals" in mapped:
                mapped["notes"] = column_map["conditionals"]
            return _normalize_row_based_profile(df, column_map=mapped)

    return _normalize_column_based_profile(df)


def load_generator_profile_hints(path: str) -> str:
    """
    Load a generator-safe profile hints file (typically the output of blackbox.profile_abstraction).

    Supports:
    - Plaintext files (anything not ending in .csv)
    - CSV with a single text column (or a known column name like 'generator_profile_hints'/'abstraction').
    """

    hints_path = str(path or "").strip()
    if not hints_path:
        raise ValueError("Generator profile hints path is required")
    if not os.path.exists(hints_path):
        raise FileNotFoundError(f"Generator profile hints not found: {hints_path}")
    if os.path.isdir(hints_path):
        raise ValueError(f"Generator profile hints path is a directory: {hints_path}")
    if os.path.getsize(hints_path) == 0:
        raise ValueError(f"Generator profile hints file is empty: {hints_path}")

    if not hints_path.lower().endswith(".csv"):
        with open(hints_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()

    df = pd.read_csv(hints_path)
    if df is None or df.empty:
        return ""

    preferred = ("generator_profile_hints", "profile_abstraction", "abstraction", "text")
    column = _select_first_matching_column(df.columns, preferred)
    if column is None and len(df.columns) == 1:
        column = str(df.columns[0])
    if column is None:
        raise ValueError(
            "Generator profile hints CSV must have a single text column or one of: "
            f"{', '.join(preferred)} (got columns: {list(df.columns)})"
        )

    values = [str(value).strip() for value in df[column].dropna().tolist()]
    values = [value for value in values if value]
    return "\n".join(values).strip()


def _normalize_column_based_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename: dict[Any, str] = {}
    for col in out.columns:
        raw = str(col).strip()
        key = raw.lower()
        canonical = _CATEGORY_CANONICAL.get(key)
        if canonical:
            rename[col] = canonical

    if rename:
        out = out.rename(columns=rename)

    return out


def _normalize_row_based_profile(df: pd.DataFrame, *, column_map: dict[str, str]) -> pd.DataFrame:
    category_col = column_map["category"]
    item_col = column_map["item"]
    notes_col = column_map.get("notes")

    buckets: dict[str, list[str]] = {}

    def _add(category: str, text: str) -> None:
        if not category or not text:
            return
        buckets.setdefault(category, []).append(text)

    for _idx, row in df.iterrows():
        raw_category_value = row.get(category_col, "")
        raw_item_value = row.get(item_col, "")

        raw_category = "" if pd.isna(raw_category_value) else str(raw_category_value).strip()
        raw_item = "" if pd.isna(raw_item_value) else str(raw_item_value).strip()
        if not raw_item:
            continue

        raw_notes = ""
        if notes_col:
            raw_notes_value = row.get(notes_col, "")
            raw_notes = "" if pd.isna(raw_notes_value) else str(raw_notes_value).strip()

        combined = raw_item
        if raw_notes:
            combined = f"{raw_item} — {raw_notes}"

        key = raw_category.lower().strip()
        canonical = _CATEGORY_CANONICAL.get(key)
        if canonical is None:
            canonical = raw_category.strip().title() if raw_category.strip() else "Other"

        _add(canonical, combined)

    if not buckets:
        return pd.DataFrame()

    data = {name: pd.Series(values) for name, values in buckets.items() if values}
    out = pd.DataFrame(data)
    return _reorder_columns(out)


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    cols = [str(col) for col in df.columns]
    ordered = [c for c in _PREFERRED_COLUMN_ORDER if c in cols]
    remainder = sorted(c for c in cols if c not in ordered)
    return df[ordered + remainder]


def _select_first_matching_column(columns: Iterable[Any], allowed: Iterable[str]) -> str | None:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for key in allowed:
        resolved = lookup.get(str(key).strip().lower())
        if resolved is not None:
            return resolved
    return None
