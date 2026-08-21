"""Env-driven base URL / API key support for the memo ``--local`` target."""
from __future__ import annotations

from mind_map.processor import cli_executor as ce


def test_default_base_url_and_no_auth_header(monkeypatch):
    monkeypatch.delenv("MIND_MAP_LOCAL_BASE_URL", raising=False)
    monkeypatch.delenv("MIND_MAP_LOCAL_API_KEY", raising=False)
    cmd = ce.build_cli_template(ce.LocalTarget(model="m"))
    assert "127.0.0.1:11435" in cmd
    assert "Authorization" not in cmd


def test_getters_read_env(monkeypatch):
    monkeypatch.setenv("MIND_MAP_LOCAL_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("MIND_MAP_LOCAL_API_KEY", "sk-test")
    assert ce.get_local_base_url() == "https://api.deepseek.com/v1"
    assert ce.get_local_api_key() == "sk-test"


def test_getters_default_when_unset(monkeypatch):
    monkeypatch.delenv("MIND_MAP_LOCAL_BASE_URL", raising=False)
    monkeypatch.delenv("MIND_MAP_LOCAL_API_KEY", raising=False)
    assert ce.get_local_base_url() == ce._DEFAULT_LOCAL_BASE_URL
    assert ce.get_local_api_key() is None


def test_explicit_base_url_used_in_command(monkeypatch):
    monkeypatch.delenv("MIND_MAP_LOCAL_API_KEY", raising=False)
    t = ce.LocalTarget(model="deepseek-chat", base_url="https://api.deepseek.com/v1")
    cmd = ce.build_cli_template(t)
    assert "https://api.deepseek.com/v1" in cmd
    assert "/chat/completions" in cmd


def test_api_key_sends_bearer_header():
    t = ce.LocalTarget(model="m", api_key="sk-test")
    cmd = ce.build_cli_template(t)
    assert "'Authorization'" in cmd
    assert "Bearer sk-test" in cmd


def test_empty_api_key_sends_no_header():
    t = ce.LocalTarget(model="m", api_key="")
    cmd = ce.build_cli_template(t)
    assert "Authorization" not in cmd


def test_parse_memo_target_threads_env_base_url(monkeypatch):
    from mind_map.app import services

    monkeypatch.setenv("MIND_MAP_LOCAL_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("MIND_MAP_LOCAL_API_KEY", "sk-test")
    captured = {}

    def fake_resolve(*, model=None, base_url=ce._DEFAULT_LOCAL_BASE_URL):
        captured["model"] = model
        captured["base_url"] = base_url
        return model or "resolved-model"

    monkeypatch.setattr(ce, "resolve_local_model", fake_resolve)
    t = services.parse_memo_target(local="", api_key=None)
    assert captured == {"model": None, "base_url": "https://api.deepseek.com/v1"}
    assert t.base_url == "https://api.deepseek.com/v1"
    assert t.model == "resolved-model"
    assert t.api_key == "sk-test"


def test_memo_ingest_threads_env_base_url(monkeypatch):
    """``memo_ingest`` duplicates resolution logic — must honor env too."""
    from mind_map.app import services

    monkeypatch.setenv("MIND_MAP_LOCAL_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("MIND_MAP_LOCAL_API_KEY", "sk-ingest")
    captured = {}

    def fake_resolve(*, model=None, base_url=ce._DEFAULT_LOCAL_BASE_URL):
        captured["base_url"] = base_url
        return model or "resolved"

    monkeypatch.setattr(ce, "resolve_local_model", fake_resolve)
    monkeypatch.setattr(
        "mind_map.app.pipeline.ingest_memo_cli",
        lambda text, store, *, target, source_id: (True, "ok", []),
    )
    ok, msg, ids = services.memo_ingest("hello", store=None, local="", source=None)
    assert ok is True
    assert captured["base_url"] == "https://api.example.com/v1"
