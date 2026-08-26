"""Tests for DeepSeekChatLLM (reasoning provider)."""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from mind_map.rag.reasoning_llm import DeepSeekChatLLM


class TestDeepSeekChatLLM:
    def test_llm_type_is_deepseek(self):
        llm = DeepSeekChatLLM()
        assert llm._llm_type == "deepseek"

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-123"}):
            llm = DeepSeekChatLLM()
            assert llm.api_key == "sk-test-123"

    def test_defaults(self):
        llm = DeepSeekChatLLM(api_key="sk-test")
        assert llm.model == "deepseek-v4-flash"
        assert llm.base_url == "https://api.deepseek.com/v1"
        assert llm.max_tokens == 8192

    def test_model_override_env(self):
        with patch.dict(os.environ, {"MIND_MAP_DEEPSEEK_MODEL": "deepseek-v4-pro"}):
            llm = DeepSeekChatLLM(api_key="sk-test")
            assert llm.model == "deepseek-v4-pro"

    def test_messages_converted_to_openai_format(self):
        llm = DeepSeekChatLLM(api_key="sk-test")
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
        ]
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hi there!"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            llm._generate(messages)

            kwargs = mock_post.call_args.kwargs
            assert kwargs["json"]["messages"] == [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            assert kwargs["json"]["model"] == "deepseek-v4-flash"

    def test_missing_key_raises(self):
        llm = DeepSeekChatLLM(api_key="")
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY not set"):
            llm._generate([HumanMessage(content="hi")])


class TestDeepSeekFactory:
    def test_returns_none_when_key_missing(self):
        from mind_map.rag.reasoning_llm import get_deepseek_llm

        with patch.dict(os.environ, {}, clear=True):
            assert get_deepseek_llm() is None

    def test_check_available(self):
        from mind_map.rag.reasoning_llm import check_deepseek_available

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            assert check_deepseek_available() is True


class TestDeepSeekProviderDispatch:
    def test_configured_provider_deepseek_returns_deepseek_llm(self):
        from mind_map.rag.reasoning_llm import get_reasoning_llm

        config = {
            "reasoning_llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "temperature": 0.7,
                "timeout": 120,
            }
        }
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}),
            patch("mind_map.core.config.load_config", return_value=config),
        ):
            llm = get_reasoning_llm()
        assert llm is not None
        assert llm._llm_type == "deepseek"

    def test_deepseek_unavailable_falls_through(self):
        from mind_map.rag.reasoning_llm import get_reasoning_llm

        config = {
            "reasoning_llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "temperature": 0.7,
                "timeout": 120,
            }
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("mind_map.core.config.load_config", return_value=config),
            patch(
                "mind_map.rag.reasoning_llm.check_claude_cli_installed",
                return_value=False,
            ),
            patch(
                "mind_map.rag.reasoning_llm.check_gemini_available",
                return_value=False,
            ),
            patch(
                "mind_map.rag.reasoning_llm.check_anthropic_available",
                return_value=False,
            ),
            patch(
                "mind_map.rag.reasoning_llm.check_openai_available",
                return_value=False,
            ),
        ):
            llm = get_reasoning_llm()
        assert llm is None
