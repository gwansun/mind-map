"""Test OpenAI API connection."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


class TestOpenAIConnection:
    """Tests for OpenAI API connection."""

    def test_api_key_exists(self, openai_api_key: str):
        """Test that API key is configured."""
        assert openai_api_key is not None
        assert openai_api_key.startswith("sk-")

    def test_openai_client_connection(self, openai_api_key: str):
        """Test basic OpenAI client connection."""
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)

        # Simple API call to test connection
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for testing
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            max_tokens=5,
        )

        assert response.choices[0].message.content is not None

    def test_langchain_openai_integration(self, openai_api_key: str):
        """Test LangChain OpenAI integration."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0,
            max_tokens=5,
        )

        response = llm.invoke("Say 'ok' and nothing else.")

        assert response.content is not None

    def test_response_generator_with_openai(self, openai_api_key: str):
        """Test ResponseGenerator with OpenAI."""
        from langchain_openai import ChatOpenAI

        from mind_map.core.schemas import GraphNode, NodeMetadata, NodeType
        from mind_map.rag.response_generator import ResponseGenerator

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.7,
        )

        generator = ResponseGenerator(llm)

        # Create mock nodes
        nodes = [
            GraphNode(
                id="test-1",
                document="Cats often show food preferences by sniffing before eating.",
                metadata=NodeMetadata(
                    type=NodeType.CONCEPT,
                    created_at=0,
                    last_interaction=0,
                    importance_score=0.9,
                ),
            ),
            GraphNode(
                id="test-2",
                document="A cat that walks away from food may not like it.",
                metadata=NodeMetadata(
                    type=NodeType.CONCEPT,
                    created_at=0,
                    last_interaction=0,
                    importance_score=0.8,
                ),
            ),
        ]

        response = generator.generate_sync(
            "How to check if a cat has a food preference?",
            nodes,
        )

        assert response is not None
        assert len(response) > 0
        print(f"\nGenerated response:\n{response}")
