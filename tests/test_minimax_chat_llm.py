"""RED contract tests for MiniMaxChatLLM.

These tests define the expected contract for the MiniMaxChatLLM class
(which does not exist yet — all tests should FAIL in the RED phase).
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestMiniMaxChatLLMExists:
    """Test that MiniMaxChatLLM can be imported and instantiated."""

    def test_llm_type_is_minimax_direct(self):
        """MiniMaxChatLLM should exist and have _llm_type == 'minimax-direct'."""
        from mind_map.rag.reasoning_llm import MiniMaxChatLLM

        llm = MiniMaxChatLLM()
        assert llm._llm_type == "minimax-direct"


class TestMiniMaxChatLLMFactory:
    """Tests for get_minimax_llm() factory function."""

    def test_missing_api_key_returns_none_from_factory(self):
        """get_minimax_llm() should return None when MINIMAX_API_KEY env var is not set."""
        from mind_map.rag.reasoning_llm import get_minimax_llm

        # Ensure the env var is not set
        env_vars_to_clear = ["MINIMAX_API_KEY"]
        saved = {k: os.environ.pop(k, None) for k in env_vars_to_clear}
        try:
            result = get_minimax_llm()
            assert result is None
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    def test_api_key_from_environment(self):
        """get_minimax_llm() should read MINIMAX_API_KEY from environment."""
        from mind_map.rag.reasoning_llm import get_minimax_llm

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
            result = get_minimax_llm()
            assert result is not None
            from mind_map.rag.reasoning_llm import MiniMaxChatLLM
            assert isinstance(result, MiniMaxChatLLM)


class TestMiniMaxChatLLMMessages:
    """Tests for message format conversion to OpenAI API format."""

    def test_messages_converted_to_openai_format(self):
        """When _generate is called, messages should be sent in OpenAI chat format."""
        from mind_map.rag.reasoning_llm import MiniMaxChatLLM
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        llm = MiniMaxChatLLM(api_key="sk-test-123")

        system_msg = SystemMessage(content="You are a helpful assistant.")
        human_msg = HumanMessage(content="What is 2+2?")
        messages = [system_msg, human_msg]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "4"}}]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            llm._generate(messages)

            # Verify requests.post was called
            assert mock_post.called
            call_kwargs = mock_post.call_args
            url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("url")
            assert "api.minimax.io" in str(url) or "minimax" in str(url).lower()

            # Verify payload structure (OpenAI-compatible format)
            payload = call_kwargs[1].get("json") if "json" in call_kwargs[1] else call_kwargs[0][1]
            assert "messages" in payload
            assert isinstance(payload["messages"], list)
            # Check that messages have 'role' and 'content' fields
            for msg in payload["messages"]:
                assert "role" in msg
                assert "content" in msg


class TestMiniMaxChatLLMThinkTags:
    """Tests for <think>...</think> tag stripping."""

    def test_strips_think_tags(self):
        """Response containing <think>reasoning...</think> should have tags stripped, leaving JSON only."""
        from mind_map.rag.reasoning_llm import MiniMaxChatLLM
        from langchain_core.messages import HumanMessage

        llm = MiniMaxChatLLM(api_key="sk-test-123")
        messages = [HumanMessage(content="Extract the JSON")]

        # Simulate API response with think tags
        raw_response = (
            '<think>The user wants me to extract JSON data. '
            'Let me think about this carefully.</think>'
            '{"summary": "Test summary", "tags": ["test"]}'
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": raw_response}}]
        }

        with patch("requests.post", return_value=mock_response):
            result = llm._generate(messages)
            content = result.generations[0].message.content

            # The think tags should be stripped
            assert "<think>" not in content
            assert "</think>" not in content
            # The JSON content should remain
            assert '{"summary": "Test summary", "tags": ["test"]}' in content