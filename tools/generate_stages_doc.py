from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageDocRow:
    stage_id: str
    kind: str | None
    doc: str | None
    source: str | None
    tags: tuple[str, ...]


def _infer_stage_kind(builder: Any) -> str | None:
    annotation = getattr(builder, "__annotations__", {}).get("return")
    if annotation is None:
        return None
    if isinstance(annotation, str):
        if "ActionStageSpec" in annotation:
            return "action"
        if "StageSpec" in annotation:
            return "chat"
        return None
    name = getattr(annotation, "__name__", "")
    if name == "ActionStageSpec":
        return "action"
    if name == "StageSpec":
        return "chat"
    return None


def _normalize_doc(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value).strip() or None
    return value.strip() or None


def _stage_rows() -> list[StageDocRow]:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from image_project.impl.current.prompting import StageCatalog  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import StageCatalog: {exc}") from exc

    registry: Mapping[str, Any] = getattr(StageCatalog, "_REGISTRY", {})
    if not isinstance(registry, Mapping):
        raise RuntimeError("StageCatalog registry is not a mapping; cannot enumerate stages")

    rows: list[StageDocRow] = []
    for stage_id in StageCatalog.available():
        entry = registry.get(stage_id)
        if entry is None:
            raise RuntimeError(f"StageCatalog registry missing entry for {stage_id!r}")

        builder = getattr(entry, "builder", None)
        kind = _infer_stage_kind(builder)
        doc = _normalize_doc(getattr(entry, "doc", None))
        source = _normalize_doc(getattr(entry, "source", None))
        tags_raw = getattr(entry, "tags", ()) or ()
        tags = tuple(str(tag).strip() for tag in tags_raw if str(tag).strip())

        rows.append(
            StageDocRow(
                stage_id=str(stage_id),
                kind=kind,
                doc=doc,
                source=source,
                tags=tags,
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
    lines.append("This file is generated from `image_project.impl.current.prompting.StageCatalog`.")
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
            suffix = ""
            if row.source:
                suffix = f" `({row.source})`"
            if doc:
                lines.append(f"- `{row.stage_id}`{kind}: {doc}{suffix}")
            else:
                lines.append(f"- `{row.stage_id}`{kind}{suffix}")
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
