"""Unit tests for resolve_default_memo_target (DeepSeek-default switch).

Covers the 2026-08-24 change: default memo target priority is
DEEPSEEK_API_KEY → MINIMAX_API_KEY → ValueError. DeepSeek rides the
OpenAI-compatible LocalTarget transport (mechanism validated E2E in cf898c5).
"""
import pytest

from mind_map.app.services import resolve_default_memo_target
from mind_map.processor.cli_executor import LocalTarget, MiniMaxTarget


def test_deepseek_key_wins_over_minimax(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
    t = resolve_default_memo_target()
    assert isinstance(t, LocalTarget)
    assert t.model == "deepseek-chat"
    assert t.base_url == "https://api.deepseek.com/v1"
    assert t.api_key == "sk-test-deepseek"


def test_minimax_fallback_when_no_deepseek(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
    t = resolve_default_memo_target()
    assert isinstance(t, MiniMaxTarget)
    assert t.api_key == "mm-test"


def test_no_keys_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        resolve_default_memo_target()


def test_deepseek_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("MIND_MAP_DEEPSEEK_MODEL", "deepseek-reasoner")
    t = resolve_default_memo_target()
    assert t.model == "deepseek-reasoner"


def test_local_command_embeds_bearer_and_url(monkeypatch):
    """The resolved DeepSeek LocalTarget must produce a working curl command."""
    from mind_map.processor.cli_executor import build_cli_template

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    t = resolve_default_memo_target()
    cmd = build_cli_template(t)
    assert "https://api.deepseek.com/v1" in cmd
    assert "Bearer sk-test-deepseek" in cmd
    assert "response_format" in cmd and "json_object" in cmd
