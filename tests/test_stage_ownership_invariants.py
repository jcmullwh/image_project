from __future__ import annotations

import ast
import re
from dataclasses import fields
from pathlib import Path

from image_project.framework.config import RunConfig


def _repo_root() -> Path:
    """Return the repository root directory for test-time file reads."""

    return Path(__file__).resolve().parents[1]


def _read_text(rel_path: str) -> str:
    """Read a UTF-8 text file from the repository root."""

    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def _find_imports(py_source: str) -> list[str]:
    """Return a list of import module strings found in the given Python source."""

    tree = ast.parse(py_source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_run_config_is_infra_only() -> None:
    names = {field.name for field in fields(RunConfig)}
    assert not any(name.startswith("prompt_") for name in names), sorted(names)
    assert "prompt_stages_include" not in names
    assert "prompt_stages_exclude" not in names
    assert "prompt_stages_overrides" not in names
    assert "prompt_output_capture_stage" not in names
    assert "prompt_stage_configs_defaults" not in names
    assert "prompt_stage_configs_instances" not in names


def test_framework_config_does_not_import_stages_or_prompt_feature_configs() -> None:
    cfg_text = _read_text("image_project/framework/config.py")
    imports = _find_imports(cfg_text)
    assert not any(module.startswith("image_project.stages") for module in imports), imports
    assert not any(module.startswith("image_project.impl") for module in imports), imports

    banned_names = ("PromptScoringConfig", "PromptBlackboxRefineConfig", "PromptConceptsConfig")
    offenders = [name for name in banned_names if name in cfg_text]
    assert not offenders, offenders


def test_framework_prompt_pipeline_parsers_do_not_import_stages() -> None:
    for rel_path in (
        "image_project/framework/prompt_pipeline/pipeline_overrides.py",
        "image_project/framework/prompt_pipeline/stage_policies.py",
    ):
        text = _read_text(rel_path)
        imports = _find_imports(text)
        assert not any(module.startswith("image_project.stages") for module in imports), (
            rel_path,
            imports,
        )
        assert not any(module.startswith("image_project.impl") for module in imports), (
            rel_path,
            imports,
        )


def test_stages_do_not_reference_global_prompt_feature_config_objects() -> None:
    stage_root = _repo_root() / "image_project" / "stages"
    py_files = sorted(stage_root.rglob("*.py"))
    assert py_files, "Expected stage modules under image_project/stages"

    patterns = [
        re.compile(r"\binputs\.cfg\.prompt_scoring\b"),
        re.compile(r"\bcfg\.prompt_scoring\b"),
        re.compile(r"\binputs\.cfg\.prompt_blackbox_refine\b"),
        re.compile(r"\bcfg\.prompt_blackbox_refine\b"),
        re.compile(r"\binputs\.cfg\.prompt_concepts\b"),
        re.compile(r"\bcfg\.prompt_concepts\b"),
    ]

    offenders: list[str] = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                offenders.append(f"{path}: matched {pattern.pattern!r}")
                break

    assert not offenders, "\n".join(offenders)

