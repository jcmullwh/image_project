from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageDocRow:
    stage_id: str
    kind: str | None
    doc: str | None
    source: str | None
    tags: tuple[str, ...]
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    captures: tuple[str, ...]


def _normalize_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _normalize_str_items(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            trimmed = item.strip()
            if trimmed:
                items.append(trimmed)
    return tuple(items)


def _stage_rows() -> list[StageDocRow]:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from image_project.stages.registry import get_stage_registry  # noqa: PLC0415

    rows: list[StageDocRow] = []
    for entry in get_stage_registry().describe():
        stage_id = _normalize_str(entry.get("stage_id")) if isinstance(entry, dict) else None
        if not stage_id:
            continue

        io = entry.get("io") if isinstance(entry, dict) else None
        io = io if isinstance(io, dict) else {}

        rows.append(
            StageDocRow(
                stage_id=stage_id,
                kind=_normalize_str(entry.get("kind")),
                doc=_normalize_str(entry.get("doc")),
                source=_normalize_str(entry.get("source")),
                tags=_normalize_str_items(entry.get("tags")),
                requires=_normalize_str_items(io.get("requires")),
                provides=_normalize_str_items(io.get("provides")),
                captures=_normalize_str_items(io.get("captures")),
            )
        )

    rows.sort(key=lambda r: r.stage_id)
    return rows


def generate_markdown(*, rows: Iterable[StageDocRow]) -> str:
    rows = list(rows)
    groups: dict[str, list[StageDocRow]] = defaultdict(list)
    for row in rows:
        prefix = row.stage_id.split(".", 1)[0]
        groups[prefix].append(row)

    lines: list[str] = []
    lines.append("# Prompt Stage Catalog")
    lines.append("")
    lines.append("This file is generated from `image_project.stages.registry.get_stage_registry()`.")
    lines.append("")
    lines.append("Regenerate with:")
    lines.append("")
    lines.append("```bash")
    lines.append("pdm run update-stages-docs")
    lines.append("# or")
    lines.append("python tools/generate_stages_doc.py")
    lines.append("```")
    lines.append("")
    lines.append(f"Total stages: {len(rows)}")
    lines.append("")

    for group in sorted(groups.keys()):
        lines.append(f"## {group}")
        lines.append("")
        for row in sorted(groups[group], key=lambda r: r.stage_id):
            kind = f" ({row.kind})" if row.kind else ""
            doc = row.doc or ""
            source_suffix = f" `({row.source})`" if row.source else ""
            io_parts: list[str] = []
            if row.requires:
                io_parts.append("requires=" + ", ".join(row.requires))
            if row.provides:
                io_parts.append("provides=" + ", ".join(row.provides))
            if row.captures:
                io_parts.append("captures=" + ", ".join(row.captures))
            io_suffix = f" [io: {'; '.join(io_parts)}]" if io_parts else ""

            if doc:
                lines.append(f"- `{row.stage_id}`{kind}: {doc}{source_suffix}{io_suffix}")
            else:
                lines.append(f"- `{row.stage_id}`{kind}{source_suffix}{io_suffix}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_stages_doc(output_path: str) -> int:
    rows = _stage_rows()
    md = generate_markdown(rows=rows)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(md)

    print(f"Wrote {output_path} ({len(rows)} stages)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate_stages_doc")
    parser.add_argument(
        "--output",
        default=os.path.join("docs", "stages.md"),
        help="Output markdown path (default: docs/stages.md)",
    )
    args = parser.parse_args(argv)

    try:
        return write_stages_doc(args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
