from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
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


def _infer_stage_kind_from_return_annotation(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None

    if isinstance(annotation, ast.Name):
        if annotation.id == "ActionStageSpec":
            return "action"
        if annotation.id == "StageSpec":
            return "chat"
        return None

    if isinstance(annotation, ast.Attribute):
        if annotation.attr == "ActionStageSpec":
            return "action"
        if annotation.attr == "StageSpec":
            return "chat"
        return None

    if isinstance(annotation, ast.Subscript):
        return _infer_stage_kind_from_return_annotation(annotation.value)

    return None


def _normalize_str_constant(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    return None


def _extract_string_tuple(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return ()
    items: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            item = elt.value.strip()
            if item:
                items.append(item)
    return tuple(items)


def _stage_rows_from_ast(prompting_path: Path) -> list[StageDocRow]:
    source = prompting_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(prompting_path))

    rows: list[StageDocRow] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "StageCatalog"
                and func.attr == "register"
            ):
                continue

            stage_id: str | None = None
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                stage_id = _normalize_str_constant(decorator.args[0].value)
            if not stage_id:
                continue

            doc: str | None = None
            source_ref: str | None = None
            tags: tuple[str, ...] = ()
            for kw in decorator.keywords:
                if not isinstance(kw, ast.keyword) or kw.arg is None:
                    continue
                if kw.arg == "doc" and isinstance(kw.value, ast.Constant):
                    doc = _normalize_str_constant(kw.value.value)
                elif kw.arg == "source" and isinstance(kw.value, ast.Constant):
                    source_ref = _normalize_str_constant(kw.value.value)
                elif kw.arg == "tags":
                    tags = _extract_string_tuple(kw.value)

            kind = _infer_stage_kind_from_return_annotation(node.returns)
            rows.append(
                StageDocRow(
                    stage_id=stage_id,
                    kind=kind,
                    doc=doc,
                    source=source_ref,
                    tags=tags,
                )
            )

    rows.sort(key=lambda r: r.stage_id)
    return rows


def _stage_rows() -> list[StageDocRow]:
    project_root = Path(__file__).resolve().parent.parent
    prompting_path = project_root / "image_project" / "impl" / "current" / "prompting.py"
    if not prompting_path.exists():
        raise RuntimeError(f"Prompting module not found: {prompting_path}")

    return _stage_rows_from_ast(prompting_path)


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
