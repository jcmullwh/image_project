from pathlib import Path

from image_project.foundation.config_io import load_config


def test_load_config_finds_repo_root_from_subdir(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]

    monkeypatch.delenv("TEST_IMAGE_PROJECT_CONFIG", raising=False)
    monkeypatch.chdir(repo_root / "docs")

    _cfg, meta = load_config(env_var="TEST_IMAGE_PROJECT_CONFIG")

    assert Path(meta["paths"][0]).resolve() == (repo_root / "config" / "config.yaml").resolve()
    assert Path(str(meta["repo_root"])).resolve() == repo_root.resolve()

