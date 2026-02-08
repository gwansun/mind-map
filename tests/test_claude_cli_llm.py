"""Tests for Claude CLI LLM integration."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestClaudeCLIInstalled:
    """Test CLI installation detection."""

    def test_installed_when_found(self):
        """Should return True when claude binary exists."""
        from mind_map.core.reasoning_llm import check_claude_cli_installed

        with patch("mind_map.core.reasoning_llm.shutil.which", return_value="/usr/local/bin/claude"):
            assert check_claude_cli_installed() is True

    def test_not_installed_when_missing(self):
        """Should return False when claude binary not found."""
        from mind_map.core.reasoning_llm import check_claude_cli_installed

        with patch("mind_map.core.reasoning_llm.shutil.which", return_value=None):
            assert check_claude_cli_installed() is False


class TestClaudeCLIAvailable:
    """Test CLI authentication check."""

    def test_available_when_authenticated(self):
        """Should return True when CLI is authenticated."""
        from mind_map.core.reasoning_llm import check_claude_cli_available

        with patch("mind_map.core.reasoning_llm.shutil.which", return_value="/usr/local/bin/claude"):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "OK"
            with patch("mind_map.core.reasoning_llm.subprocess.run", return_value=mock_result):
                assert check_claude_cli_available() is True

    def test_unavailable_when_not_authenticated(self):
        """Should return False when CLI returns error."""
        from mind_map.core.reasoning_llm import check_claude_cli_available

        with patch("mind_map.core.reasoning_llm.shutil.which", return_value="/usr/local/bin/claude"):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            with patch("mind_map.core.reasoning_llm.subprocess.run", return_value=mock_result):
                assert check_claude_cli_available() is False

    def test_unavailable_when_not_installed(self):
        """Should return False when CLI not installed."""
        from mind_map.core.reasoning_llm import check_claude_cli_available

        with patch("mind_map.core.reasoning_llm.shutil.which", return_value=None):
            assert check_claude_cli_available() is False

    def test_unavailable_on_timeout(self):
        """Should return False when CLI times out."""
        from mind_map.core.reasoning_llm import check_claude_cli_available

        with patch("mind_map.core.reasoning_llm.shutil.which", return_value="/usr/local/bin/claude"):
            with patch(
                "mind_map.core.reasoning_llm.subprocess.run",
                side_effect=subprocess.TimeoutExpired("claude", 30),
            ):
                assert check_claude_cli_available() is False


class TestClaudeCLILLM:
    """Test LLM wrapper."""

    def test_llm_type(self):
        """Should return correct LLM type."""
        from mind_map.core.reasoning_llm import ClaudeCLILLM

        llm = ClaudeCLILLM(model="sonnet")
        assert llm._llm_type == "claude-cli"

    def test_generate_success(self):
        """Should generate response from CLI output."""
        from langchain_core.messages import HumanMessage

        from mind_map.core.reasoning_llm import ClaudeCLILLM

        llm = ClaudeCLILLM(model="sonnet")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello! How can I help you today?"

        with patch("mind_map.core.reasoning_llm.subprocess.run", return_value=mock_result):
            result = llm._generate([HumanMessage(content="Hi")])
            assert "Hello" in result.generations[0].message.content

    def test_generate_with_system_message(self):
        """Should format system messages correctly."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from mind_map.core.reasoning_llm import ClaudeCLILLM

        llm = ClaudeCLILLM(model="sonnet")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Response"

        with patch("mind_map.core.reasoning_llm.subprocess.run", return_value=mock_result) as mock_run:
            llm._generate([
                SystemMessage(content="You are helpful"),
                HumanMessage(content="Hi"),
            ])
            # Check that the prompt was formatted correctly
            call_args = mock_run.call_args
            input_text = call_args.kwargs.get("input", "")
            assert "System: You are helpful" in input_text
            assert "Human: Hi" in input_text

    def test_generate_timeout(self):
        """Should raise RuntimeError on timeout."""
        from langchain_core.messages import HumanMessage

        from mind_map.core.reasoning_llm import ClaudeCLILLM

        llm = ClaudeCLILLM(model="sonnet", timeout=1)

        with patch(
            "mind_map.core.reasoning_llm.subprocess.run",
            side_effect=subprocess.TimeoutExpired("claude", 1),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                llm._generate([HumanMessage(content="Hi")])

    def test_generate_cli_error(self):
        """Should raise RuntimeError on CLI error."""
        from langchain_core.messages import HumanMessage

        from mind_map.core.reasoning_llm import ClaudeCLILLM

        llm = ClaudeCLILLM(model="sonnet")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Authentication failed"

        with patch("mind_map.core.reasoning_llm.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Claude CLI error"):
                llm._generate([HumanMessage(content="Hi")])

    def test_generate_cli_not_found(self):
        """Should raise RuntimeError when CLI not found."""
        from langchain_core.messages import HumanMessage

        from mind_map.core.reasoning_llm import ClaudeCLILLM

        llm = ClaudeCLILLM(model="sonnet")

        with patch(
            "mind_map.core.reasoning_llm.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            with pytest.raises(RuntimeError, match="Claude CLI not found"):
                llm._generate([HumanMessage(content="Hi")])


class TestGetClaudeCLILLM:
    """Test factory function."""

    def test_returns_llm_when_available(self):
        """Should return ClaudeCLILLM when CLI is available."""
        from mind_map.core.reasoning_llm import ClaudeCLILLM, get_claude_cli_llm

        with patch("mind_map.core.reasoning_llm.check_claude_cli_installed", return_value=True):
            with patch("mind_map.core.reasoning_llm.check_claude_cli_available", return_value=True):
                llm = get_claude_cli_llm("sonnet", 120)
                assert isinstance(llm, ClaudeCLILLM)
                assert llm.model == "sonnet"
                assert llm.timeout == 120

    def test_returns_none_when_not_installed(self):
        """Should return None when CLI not installed."""
        from mind_map.core.reasoning_llm import get_claude_cli_llm

        with patch("mind_map.core.reasoning_llm.check_claude_cli_installed", return_value=False):
            llm = get_claude_cli_llm()
            assert llm is None

    def test_returns_none_when_not_authenticated(self):
        """Should return None when CLI not authenticated."""
        from mind_map.core.reasoning_llm import get_claude_cli_llm

        with patch("mind_map.core.reasoning_llm.check_claude_cli_installed", return_value=True):
            with patch("mind_map.core.reasoning_llm.check_claude_cli_available", return_value=False):
                llm = get_claude_cli_llm()
                assert llm is None
