import ast
from pathlib import Path


def test_pipelinekit_source_does_not_import_image_project():
    repo_root = Path(__file__).resolve().parents[1]
    pipelinekit_dir = repo_root / "pipelinekit"

    forbidden_prefixes = ("image_project",)
    offenders: list[str] = []

    for path in sorted(pipelinekit_dir.rglob("*.py")):
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


def test_importing_pipelinekit_modules_does_not_pull_in_image_project():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """\
        import importlib
        import pkgutil
        import sys

        import pipelinekit as pkg

        forbidden = ("image_project",)

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

