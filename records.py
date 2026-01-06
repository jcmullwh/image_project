from __future__ import annotations

import csv
import os
from typing import Any, Iterable


def append_generation_row(csv_path: str, row: dict[str, Any], fieldnames: list[str]) -> None:
    if not fieldnames:
        raise ValueError("fieldnames must be non-empty")

    missing = [name for name in fieldnames if name not in row]
    if missing:
        raise KeyError(f"Generation row missing required keys: {missing}")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if file_exists:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            existing_header: list[str] = next(reader, [])
        if existing_header and existing_header != list(fieldnames):
            raise ValueError(
                f"Existing CSV header does not match expected schema at {csv_path}. "
                f"expected={list(fieldnames)} actual={existing_header}"
            )

    output_row = {name: row.get(name, "") for name in fieldnames}

    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(output_row)


def csv_fieldnames_reader(path: str) -> list[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        return next(reader, []) or []

