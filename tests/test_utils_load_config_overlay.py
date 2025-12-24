import os

import pytest

from utils import load_config


def test_load_config_base_only(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_IMAGE_PROJECT_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")

    cfg, meta = load_config(
        config_rel_path=str(tmp_path),
        config_name="config",
        config_type=".yaml",
        env_var="TEST_IMAGE_PROJECT_CONFIG",
    )

    assert cfg == {"a": 1, "b": {"c": 2}}
    assert meta["mode"] == "base"
    assert os.path.basename(meta["paths"][0]) == "config.yaml"


def test_load_config_base_plus_local_overlay(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_IMAGE_PROJECT_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    (tmp_path / "config.local.yaml").write_text("b:\n  c: 3\n  d: 4\n", encoding="utf-8")

    cfg, meta = load_config(
        config_rel_path=str(tmp_path),
        config_name="config",
        config_type=".yaml",
        env_var="TEST_IMAGE_PROJECT_CONFIG",
    )

    assert cfg == {"a": 1, "b": {"c": 3, "d": 4}}
    assert meta["mode"] == "base+local"
    assert len(meta["paths"]) == 2


def test_load_config_overlay_type_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_IMAGE_PROJECT_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text("a:\n  b: 1\n", encoding="utf-8")
    (tmp_path / "config.local.yaml").write_text("a: [1, 2]\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid config overlay merge at a"):
        load_config(
            config_rel_path=str(tmp_path),
            config_name="config",
            config_type=".yaml",
            env_var="TEST_IMAGE_PROJECT_CONFIG",
        )


def test_load_config_invalid_overlay_yaml_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_IMAGE_PROJECT_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "config.local.yaml").write_text("a: [1, 2\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_config(
            config_rel_path=str(tmp_path),
            config_name="config",
            config_type=".yaml",
            env_var="TEST_IMAGE_PROJECT_CONFIG",
        )

    assert "config.local.yaml" in str(excinfo.value)


def test_load_config_env_override_loads_single_file(tmp_path, monkeypatch):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    (base_dir / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    (base_dir / "config.local.yaml").write_text("a: 2\n", encoding="utf-8")
    env_path = env_dir / "my_config.yaml"
    env_path.write_text("a: 999\n", encoding="utf-8")

    monkeypatch.setenv("TEST_IMAGE_PROJECT_CONFIG", str(env_path))
    cfg, meta = load_config(
        config_rel_path=str(base_dir),
        config_name="config",
        config_type=".yaml",
        env_var="TEST_IMAGE_PROJECT_CONFIG",
    )

    assert cfg == {"a": 999}
    assert meta["mode"] == "env"
    assert meta["paths"] == [os.path.abspath(str(env_path))]

