"""Unit tests for resolve_default_memo_target (CommandCode-default route).

Covers the default memo-target priority:
COMMANDCODE_API_KEY → MINIMAX_API_KEY → ValueError.

The DeepSeek family rides the OpenAI-compatible LocalTarget transport, which
as of the 2026-09-10 route change points at the CommandCode gateway instead of
api.deepseek.com. The `deepseek` naming is retained because it denotes the
model family, not the endpoint.
"""
import pytest

from mind_map.app.services import resolve_default_memo_target
from mind_map.processor.cli_executor import LocalTarget, MiniMaxTarget


def test_commandcode_key_wins_over_minimax(monkeypatch):
    monkeypatch.setenv("COMMANDCODE_API_KEY", "cc-test")
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
    t = resolve_default_memo_target()
    assert isinstance(t, LocalTarget)
    assert t.model == "deepseek/deepseek-v4.1-flash"
    assert t.base_url == "https://api.commandcode.ai/provider/v1"
    assert t.api_key == "cc-test"


def test_minimax_fallback_when_no_commandcode(monkeypatch):
    monkeypatch.delenv("COMMANDCODE_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
    t = resolve_default_memo_target()
    assert isinstance(t, MiniMaxTarget)
    assert t.api_key == "mm-test"


def test_no_keys_raises(monkeypatch):
    monkeypatch.delenv("COMMANDCODE_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="COMMANDCODE_API_KEY"):
        resolve_default_memo_target()


def test_model_override(monkeypatch):
    monkeypatch.delenv("MIND_MAP_DEEPSEEK_MODEL", raising=False)
    monkeypatch.setenv("COMMANDCODE_API_KEY", "cc-test")
    monkeypatch.setenv("MIND_MAP_LLM_MODEL", "deepseek/deepseek-v4-pro")
    t = resolve_default_memo_target()
    assert t.model == "deepseek/deepseek-v4-pro"


def test_legacy_model_env_still_honoured(monkeypatch):
    """MIND_MAP_DEEPSEEK_MODEL stays a working alias for the new seam."""
    monkeypatch.delenv("MIND_MAP_LLM_MODEL", raising=False)
    monkeypatch.setenv("COMMANDCODE_API_KEY", "cc-test")
    monkeypatch.setenv("MIND_MAP_DEEPSEEK_MODEL", "deepseek/deepseek-v4-pro")
    t = resolve_default_memo_target()
    assert t.model == "deepseek/deepseek-v4-pro"


def test_no_direct_deepseek_route_remains(monkeypatch):
    """Guard: the default memo target must not reach api.deepseek.com."""
    monkeypatch.setenv("COMMANDCODE_API_KEY", "cc-test")
    t = resolve_default_memo_target()
    assert "api.deepseek.com" not in t.base_url


def test_local_command_embeds_bearer_and_url(monkeypatch):
    """The resolved default target must still produce a working curl command."""
    from mind_map.processor.cli_executor import build_cli_template

    monkeypatch.setenv("COMMANDCODE_API_KEY", "cc-test")
    t = resolve_default_memo_target()
    cmd = build_cli_template(t)
    assert "https://api.commandcode.ai/provider/v1" in cmd
    assert "Bearer cc-test" in cmd
    assert "response_format" in cmd and "json_object" in cmd
