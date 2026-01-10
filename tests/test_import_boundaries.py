import ast
from pathlib import Path


def test_foundation_does_not_import_framework_or_impl():
    repo_root = Path(__file__).resolve().parents[1]
    foundation_dir = repo_root / "image_project" / "foundation"

    forbidden_prefixes = ("image_project.framework", "image_project.impl")
    offenders: list[str] = []

    for path in sorted(foundation_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith(forbidden_prefixes):
                        offenders.append(f"{path}: import {name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                module = node.module
                if module.startswith(forbidden_prefixes):
                    offenders.append(f"{path}: from {module} import ...")

    assert offenders == []

