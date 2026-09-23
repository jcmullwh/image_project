import ast
from pathlib import Path


def test_foundation_does_not_import_framework_or_impl():
    repo_root = Path(__file__).resolve().parents[1]
    foundation_dir = repo_root / "image_project" / "foundation"

    forbidden_prefixes = ("image_project.framework", "image_project.impl", "image_project.stages")
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


def test_framework_source_does_not_import_stages_or_impl():
    repo_root = Path(__file__).resolve().parents[1]
    framework_dir = repo_root / "image_project" / "framework"

    forbidden_prefixes = ("image_project.impl", "image_project.stages")
    offenders: list[str] = []

    for path in sorted(framework_dir.rglob("*.py")):
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


def test_stages_source_does_not_import_impl():
    repo_root = Path(__file__).resolve().parents[1]
    stages_dir = repo_root / "image_project" / "stages"

    forbidden_prefixes = ("image_project.impl",)
    offenders: list[str] = []

    for path in sorted(stages_dir.rglob("*.py")):
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


def test_framework_contains_no_prompt_templates():
    repo_root = Path(__file__).resolve().parents[1]
    framework_dir = repo_root / "image_project" / "framework"

    forbidden_snippets = (
        "You generate short image titles.",
        "Selected concepts (keep length and order):",
        "You rewrite selected creative concepts so none of them conflict with the user's dislikes.",
    )

    offenders: list[str] = []
    for path in sorted(framework_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in text:
                offenders.append(f"{path}: contains prompt template snippet {snippet!r}")

    assert offenders == []


def test_importing_framework_modules_does_not_pull_in_stages_or_impl():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """\
        import importlib
        import pkgutil
        import sys

        import image_project.framework as pkg

        forbidden = ("image_project.impl", "image_project.stages")

        for module in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            before = set(sys.modules)
            importlib.import_module(module.name)
            loaded = sorted(name for name in (set(sys.modules) - before) if name.startswith(forbidden))
            if loaded:
                raise SystemExit(f"Importing {module.name} loaded forbidden modules: {loaded}")
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
