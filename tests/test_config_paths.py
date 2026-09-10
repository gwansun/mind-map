"""Config/env resolution must NOT depend on the process working directory.

Regression this pins (found via a live MCP e2e test):

    core/config.py used  config_path = Path("config.yaml")  -- relative to cwd.

The Hermes-spawned MCP server runs with cwd=~/.hermes, so ``load_config()``
silently read **Hermes's own** config.yaml, found no ``reasoning_llm`` key, and
``get_reasoning_llm()`` therefore fell back to its ``"minimax-direct"`` default.
Result: ``mind_map_ask`` called MiniMax instead of the configured CommandCode
route and failed with HTTP 429.

The same defect appeared at three more sites (bare ``load_dotenv()`` in
``core/config.py`` and ``app/api/routes.py``, and a config *write* path in
``processor/processing_llm.py`` that could create a stray config.yaml wherever
the process happened to be).
"""
from __future__ import annotations

from mind_map.core import config as cfg


def test_config_path_points_at_the_project_root():
    """A single explicit config path, derived from the package location."""
    assert cfg.CONFIG_PATH == cfg.PROJECT_ROOT / "config.yaml"
    assert cfg.CONFIG_PATH.exists()


def test_env_path_points_at_the_project_root():
    assert cfg.ENV_PATH == cfg.PROJECT_ROOT / ".env"


def test_load_config_is_cwd_independent(tmp_path, monkeypatch):
    """From an unrelated cwd, the PROJECT config must still be read."""
    monkeypatch.chdir(tmp_path)
    loaded = cfg.load_config()
    assert "reasoning_llm" in loaded, (
        "project config.yaml was not read when cwd was unrelated"
    )


def test_load_config_ignores_a_decoy_config_in_cwd(tmp_path, monkeypatch):
    """A config.yaml sitting in the cwd must not shadow the project's."""
    (tmp_path / "config.yaml").write_text("reasoning_llm:\n  provider: DECOY\n")
    monkeypatch.chdir(tmp_path)
    loaded = cfg.load_config()
    assert loaded.get("reasoning_llm", {}).get("provider") != "DECOY"


def test_processing_model_persist_writes_project_config_not_cwd(tmp_path, monkeypatch):
    """persist=True must target the project config, never the cwd."""
    from mind_map.processor import processing_llm as pl

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cfgfile = scratch / "config.yaml"
    cfgfile.write_text("processing_llm:\n  provider: ollama\n  model: phi3.5\n")
    monkeypatch.setattr(pl, "CONFIG_PATH", cfgfile)

    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert pl.set_processing_model("qwen2.5:3b", persist=True) is True

    assert "qwen2.5:3b" in cfgfile.read_text(), "project config was not updated"
    assert not (elsewhere / "config.yaml").exists(), (
        "a stray config.yaml was written into the cwd"
    )
