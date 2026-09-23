import importlib.util
from pathlib import Path


def test_known_compatibility_shim_modules_are_gone():
    repo_root = Path(__file__).resolve().parents[1]

    shim_files = [
        repo_root / "image_project" / "foundation" / "pipeline.py",
        repo_root / "image_project" / "foundation" / "messages.py",
        repo_root / "image_project" / "framework" / "config_namespace.py",
        repo_root / "image_project" / "framework" / "stage_types.py",
        repo_root / "image_project" / "framework" / "stage_registry.py",
        repo_root / "image_project" / "impl" / "current" / "prompting.py",
        repo_root / "image_project" / "impl" / "current" / "blackbox_refine_prompts.py",
        repo_root / "image_project" / "stages" / "types.py",
    ]

    existing = [str(path) for path in shim_files if path.exists()]
    assert existing == []

    shim_modules = [
        "image_project.foundation.pipeline",
        "image_project.foundation.messages",
        "image_project.framework.config_namespace",
        "image_project.framework.stage_types",
        "image_project.framework.stage_registry",
        "image_project.impl.current.prompting",
        "image_project.impl.current.blackbox_refine_prompts",
        "image_project.stages.types",
    ]

    importable = [name for name in shim_modules if importlib.util.find_spec(name) is not None]
    assert importable == []


def test_no_compatibility_reexport_markers_in_python_source():
    repo_root = Path(__file__).resolve().parents[1]
    roots = [
        repo_root / "image_project",
        repo_root / "pipelinekit",
    ]

    markers = (
        "Compatibility re-export",
        "compatibility shim",
        "Deprecated entrypoint",
    )

    offenders: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for marker in markers:
                if marker in text:
                    offenders.append(f"{path}: contains marker {marker!r}")
    assert offenders == []

