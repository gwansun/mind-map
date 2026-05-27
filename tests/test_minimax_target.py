"""RED contract tests for MiniMaxTarget and build_minimax_http_command.

These tests define the expected contract for the MiniMax target dataclass
and HTTP command builder. They must FAIL because no implementation exists yet.
"""
from __future__ import annotations

import pytest

from mind_map.processor.cli_executor import MiniMaxTarget, build_minimax_http_command


class TestMiniMaxTargetDefaults:
    """Tests for MiniMaxTarget default values."""

    def test_target_has_default_model(self) -> None:
        """MiniMaxTarget should default model to 'MiniMax-M2.5'."""
        target = MiniMaxTarget(api_key="sk-test")
        assert target.model == "MiniMax-M2.5"

    def test_target_has_default_base_url(self) -> None:
        """MiniMaxTarget should default base_url to 'https://api.minimax.io'."""
        target = MiniMaxTarget(api_key="sk-test")
        assert target.base_url == "https://api.minimax.io"

    def test_target_has_default_max_tokens(self) -> None:
        """MiniMaxTarget should default max_tokens to 124000."""
        target = MiniMaxTarget(api_key="sk-test")
        assert target.max_tokens == 124000


class TestBuildMiniMaxHttpCommand:
    """Tests for build_minimax_http_command function."""

    def test_build_http_command_uses_correct_endpoint(self) -> None:
        """Command should target the MiniMax /v1/chat/completions endpoint."""
        target = MiniMaxTarget(api_key="sk-test")
        cmd = build_minimax_http_command(target)
        assert "/v1/chat/completions" in cmd
        assert "api.minimax.io" in cmd

    def test_build_http_command_contains_api_key(self) -> None:
        """Command should contain the API key (masked or actual)."""
        target = MiniMaxTarget(api_key="sk-abc123xyz")
        cmd = build_minimax_http_command(target)
        assert "sk-abc123xyz" in cmd