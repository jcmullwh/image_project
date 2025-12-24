from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


def append_run_index_entry(path: str, entry: Mapping[str, Any]) -> None:
    """
    Append a single JSON object to a JSONL run index file.

    The caller is responsible for building a schema_versioned entry object.
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), ensure_ascii=False))
        handle.write("\n")

