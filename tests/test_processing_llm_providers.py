"""Tests for multi-provider processing LLM (cloud-first + Ollama fallback)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mind_map.core.schemas import Edge, GraphNode, NodeMetadata, NodeType
from mind_map.rag.graph_store import GraphStore

# The canonical patch target for load_config (defined in config.py,
# imported by processing_llm.py, reasoning_llm.py, and llm.py)
LOAD_CONFIG_PATCH = "mind_map.core.config.load_config"


# ============== Fixtures ==============


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


@pytest.fixture
def _clean_env():
    """Ensure no cloud API keys leak into tests."""
    env_vars = ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    saved = {k: os.environ.pop(k, None) for k in env_vars}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ============== _try_cloud_processing_llm ==============


class TestTryCloudProcessingLLM:
    """Tests for _try_cloud_processing_llm()."""

    def test_returns_none_when_no_keys(self, _clean_env):
        """Should return (None, None) when no API keys are set."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        llm, provider = _try_cloud_processing_llm("auto")
        assert llm is None
        assert provider is None

    def test_returns_none_for_unknown_provider(self, _clean_env):
        """Should return (None, None) for unknown provider string."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        llm, provider = _try_cloud_processing_llm("unknown")
        assert llm is None
        assert provider is None

    def test_gemini_selected_when_key_present(self, _clean_env):
        """Should return gemini LLM when GOOGLE_API_KEY is set."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        mock_llm = MagicMock()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch.dict(
                "sys.modules",
                {"langchain_google_genai": MagicMock(
                    ChatGoogleGenerativeAI=MagicMock(return_value=mock_llm)
                )},
            ):
                llm, provider = _try_cloud_processing_llm("auto")
                assert provider == "gemini"
                assert llm is mock_llm

    def test_anthropic_selected_when_key_present(self, _clean_env):
        """Should return anthropic LLM when ANTHROPIC_API_KEY is set."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        mock_llm = MagicMock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(
                "sys.modules",
                {"langchain_anthropic": MagicMock(
                    ChatAnthropic=MagicMock(return_value=mock_llm)
                )},
            ):
                llm, provider = _try_cloud_processing_llm("auto")
                assert provider == "anthropic"
                assert llm is mock_llm

    def test_openai_selected_when_key_present(self, _clean_env):
        """Should return openai LLM when OPENAI_API_KEY is set."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        mock_llm = MagicMock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(
                "sys.modules",
                {"langchain_openai": MagicMock(
                    ChatOpenAI=MagicMock(return_value=mock_llm)
                )},
            ):
                llm, provider = _try_cloud_processing_llm("auto")
                assert provider == "openai"
                assert llm is mock_llm

    def test_priority_gemini_over_anthropic(self, _clean_env):
        """Gemini should be tried before Anthropic when both keys exist."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        mock_gemini = MagicMock()
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "gkey",
            "ANTHROPIC_API_KEY": "akey",
        }):
            with patch.dict(
                "sys.modules",
                {
                    "langchain_google_genai": MagicMock(
                        ChatGoogleGenerativeAI=MagicMock(return_value=mock_gemini)
                    ),
                    "langchain_anthropic": MagicMock(
                        ChatAnthropic=MagicMock(return_value=MagicMock())
                    ),
                },
            ):
                llm, provider = _try_cloud_processing_llm("auto")
                assert provider == "gemini"

    def test_specific_provider_only_tries_that(self, _clean_env):
        """provider='anthropic' should skip Gemini even if key exists."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        mock_anthropic = MagicMock()
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "gkey",
            "ANTHROPIC_API_KEY": "akey",
        }):
            with patch.dict(
                "sys.modules",
                {
                    "langchain_anthropic": MagicMock(
                        ChatAnthropic=MagicMock(return_value=mock_anthropic)
                    ),
                },
            ):
                llm, provider = _try_cloud_processing_llm("anthropic")
                assert provider == "anthropic"
                assert llm is mock_anthropic

    def test_fallthrough_when_import_fails(self, _clean_env):
        """Should skip provider when langchain package not installed."""
        from mind_map.processor.processing_llm import _try_cloud_processing_llm

        mock_openai = MagicMock()
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "gkey",
            "OPENAI_API_KEY": "okey",
        }):
            # Gemini import fails, should fall through to OpenAI
            gemini_mod = MagicMock()
            gemini_mod.ChatGoogleGenerativeAI = MagicMock(
                side_effect=ImportError("not installed")
            )
            with patch.dict(
                "sys.modules",
                {
                    "langchain_google_genai": gemini_mod,
                    "langchain_openai": MagicMock(
                        ChatOpenAI=MagicMock(return_value=mock_openai)
                    ),
                },
            ):
                llm, provider = _try_cloud_processing_llm("auto")
                assert provider == "openai"


# ============== detect_processing_provider ==============


class TestDetectProcessingProvider:
    """Tests for detect_processing_provider()."""

    def test_ollama_provider_config(self, _clean_env):
        """When provider=ollama, should return ollama regardless of API keys."""
        from mind_map.processor.processing_llm import detect_processing_provider

        config = {
            "processing_llm": {"provider": "ollama", "model": "mistral"}
        }
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}):
            with patch(LOAD_CONFIG_PATCH, return_value=config):
                provider, model = detect_processing_provider()
                assert provider == "ollama"
                assert model == "mistral"

    def test_auto_provider_with_google_key(self, _clean_env):
        """auto provider with GOOGLE_API_KEY should detect gemini."""
        from mind_map.processor.processing_llm import detect_processing_provider

        config = {"processing_llm": {"provider": "auto", "model": "phi3.5"}}
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}):
            with patch(LOAD_CONFIG_PATCH, return_value=config):
                provider, model = detect_processing_provider()
                assert provider == "gemini"
                assert model == "gemini-2.0-flash"

    def test_auto_provider_with_anthropic_key(self, _clean_env):
        """auto provider with ANTHROPIC_API_KEY should detect anthropic."""
        from mind_map.processor.processing_llm import detect_processing_provider

        config = {"processing_llm": {"provider": "auto", "model": "phi3.5"}}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "akey"}):
            with patch(LOAD_CONFIG_PATCH, return_value=config):
                provider, model = detect_processing_provider()
                assert provider == "anthropic"

    def test_auto_provider_with_openai_key(self, _clean_env):
        """auto provider with OPENAI_API_KEY should detect openai."""
        from mind_map.processor.processing_llm import detect_processing_provider

        config = {"processing_llm": {"provider": "auto", "model": "phi3.5"}}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "okey"}):
            with patch(LOAD_CONFIG_PATCH, return_value=config):
                provider, model = detect_processing_provider()
                assert provider == "openai"

    def test_auto_provider_no_keys_falls_back_to_ollama(self, _clean_env):
        """auto provider with no API keys should fall back to ollama."""
        from mind_map.processor.processing_llm import detect_processing_provider

        config = {"processing_llm": {"provider": "auto", "model": "phi3.5"}}
        with patch(LOAD_CONFIG_PATCH, return_value=config):
            provider, model = detect_processing_provider()
            assert provider == "ollama"
            assert model == "phi3.5"

    def test_default_provider_is_auto(self, _clean_env):
        """When no provider is configured, default should be auto."""
        from mind_map.processor.processing_llm import detect_processing_provider

        config = {"processing_llm": {"model": "phi3.5"}}
        with patch(LOAD_CONFIG_PATCH, return_value=config):
            provider, model = detect_processing_provider()
            # With no keys, auto falls to ollama
            assert provider == "ollama"
            assert model == "phi3.5"

    def test_specific_cloud_provider_without_key(self, _clean_env):
        """Specific cloud provider without key still returns that provider name."""
        from mind_map.processor.processing_llm import detect_processing_provider

        config = {"processing_llm": {"provider": "gemini", "model": "phi3.5"}}
        with patch(LOAD_CONFIG_PATCH, return_value=config):
            provider, model = detect_processing_provider()
            # Should still report gemini even without key
            assert provider == "gemini"
            assert model == "gemini-2.0-flash"


# ============== get_processing_llm provider routing ==============


class TestGetProcessingLLM:
    """Tests for get_processing_llm() provider routing."""

    def test_ollama_provider_skips_cloud(self, _clean_env):
        """provider=ollama should go directly to Ollama, skip cloud."""
        from mind_map.processor.processing_llm import get_processing_llm

        config = {
            "processing_llm": {
                "provider": "ollama",
                "model": "phi3.5",
                "temperature": 0.1,
            }
        }
        mock_ollama_llm = MagicMock()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}):
            with patch(LOAD_CONFIG_PATCH, return_value=config):
                with patch(
                    "mind_map.processor.processing_llm._try_cloud_processing_llm"
                ) as mock_cloud:
                    with patch(
                        "mind_map.processor.processing_llm._get_ollama_processing_llm",
                        return_value=mock_ollama_llm,
                    ):
                        llm = get_processing_llm()
                        mock_cloud.assert_not_called()
                        assert llm is mock_ollama_llm

    def test_auto_provider_tries_cloud_first(self, _clean_env):
        """provider=auto should try cloud first."""
        from mind_map.processor.processing_llm import get_processing_llm

        config = {
            "processing_llm": {
                "provider": "auto",
                "model": "phi3.5",
                "temperature": 0.1,
            }
        }
        mock_cloud_llm = MagicMock()
        with patch(LOAD_CONFIG_PATCH, return_value=config):
            with patch(
                "mind_map.processor.processing_llm._try_cloud_processing_llm",
                return_value=(mock_cloud_llm, "gemini"),
            ):
                llm = get_processing_llm()
                assert llm is mock_cloud_llm

    def test_auto_provider_falls_back_to_ollama(self, _clean_env):
        """provider=auto should fall back to Ollama when cloud fails."""
        from mind_map.processor.processing_llm import get_processing_llm

        config = {
            "processing_llm": {
                "provider": "auto",
                "model": "phi3.5",
                "temperature": 0.1,
            }
        }
        mock_ollama_llm = MagicMock()
        with patch(LOAD_CONFIG_PATCH, return_value=config):
            with patch(
                "mind_map.processor.processing_llm._try_cloud_processing_llm",
                return_value=(None, None),
            ):
                with patch(
                    "mind_map.processor.processing_llm._get_ollama_processing_llm",
                    return_value=mock_ollama_llm,
                ):
                    llm = get_processing_llm()
                    assert llm is mock_ollama_llm

    def test_specific_cloud_provider_falls_back_to_ollama(self, _clean_env):
        """Specific cloud provider should fall back to Ollama when unavailable."""
        from mind_map.processor.processing_llm import get_processing_llm

        config = {
            "processing_llm": {
                "provider": "gemini",
                "model": "phi3.5",
                "temperature": 0.1,
            }
        }
        mock_ollama_llm = MagicMock()
        with patch(LOAD_CONFIG_PATCH, return_value=config):
            with patch(
                "mind_map.processor.processing_llm._try_cloud_processing_llm",
                return_value=(None, None),
            ):
                with patch(
                    "mind_map.processor.processing_llm._get_ollama_processing_llm",
                    return_value=mock_ollama_llm,
                ):
                    llm = get_processing_llm()
                    assert llm is mock_ollama_llm


# ============== get_llm_status ==============


class TestGetLLMStatus:
    """Tests for get_llm_status() provider detection."""

    def test_reports_ollama_when_no_keys(self, _clean_env):
        """Should report ollama provider when no cloud keys available."""
        from mind_map.rag.llm_status import get_llm_status

        config = {
            "processing_llm": {"provider": "auto", "model": "phi3.5"},
            "reasoning_llm": {"provider": "claude-cli"},
        }
        with patch(LOAD_CONFIG_PATCH, return_value=config):
            with patch(
                "mind_map.rag.llm_status.check_ollama_available",
                return_value=True,
            ):
                with patch(
                    "mind_map.rag.llm_status.check_claude_cli_available",
                    return_value=False,
                ):
                    status = get_llm_status()
                    assert status["processing_llm"]["provider"] == "ollama"
                    assert status["processing_llm"]["model"] == "phi3.5"
                    assert status["processing_llm"]["status"] == "online"

    def test_reports_gemini_when_key_present(self, _clean_env):
        """Should report gemini provider when GOOGLE_API_KEY is set."""
        from mind_map.rag.llm_status import get_llm_status

        config = {
            "processing_llm": {"provider": "auto", "model": "phi3.5"},
            "reasoning_llm": {"provider": "claude-cli"},
        }
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}):
            with patch(LOAD_CONFIG_PATCH, return_value=config):
                with patch(
                    "mind_map.rag.llm_status.check_gemini_available",
                    return_value=True,
                ):
                    with patch(
                        "mind_map.rag.llm_status.check_claude_cli_available",
                        return_value=False,
                    ):
                        status = get_llm_status()
                        assert status["processing_llm"]["provider"] == "gemini"
                        assert status["processing_llm"]["model"] == "gemini-2.0-flash"
                        assert status["processing_llm"]["status"] == "online"

    def test_reports_ollama_explicit_provider(self, _clean_env):
        """Should report ollama when provider is explicitly set to ollama."""
        from mind_map.rag.llm_status import get_llm_status

        config = {
            "processing_llm": {"provider": "ollama", "model": "mistral"},
            "reasoning_llm": {"provider": "claude-cli"},
        }
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}):
            with patch(LOAD_CONFIG_PATCH, return_value=config):
                with patch(
                    "mind_map.rag.llm_status.check_ollama_available",
                    return_value=False,
                ):
                    with patch(
                        "mind_map.rag.llm_status.check_claude_cli_available",
                        return_value=False,
                    ):
                        status = get_llm_status()
                        assert status["processing_llm"]["provider"] == "ollama"
                        assert status["processing_llm"]["model"] == "mistral"
                        assert status["processing_llm"]["status"] == "offline"


# ============== GraphNode.relation_factor ==============


class TestGraphNodeRelationFactor:
    """Tests for the relation_factor field on GraphNode."""

    def test_default_is_none(self):
        """relation_factor defaults to None."""
        node = GraphNode(
            id="test-1",
            document="test doc",
            metadata=NodeMetadata(
                type=NodeType.CONCEPT,
                created_at=1000.0,
                last_interaction=1000.0,
            ),
        )
        assert node.relation_factor is None

    def test_can_set_value(self):
        """relation_factor can be set to a float."""
        node = GraphNode(
            id="test-1",
            document="test doc",
            metadata=NodeMetadata(
                type=NodeType.CONCEPT,
                created_at=1000.0,
                last_interaction=1000.0,
            ),
            relation_factor=0.75,
        )
        assert node.relation_factor == 0.75

    def test_serializes_in_model_dump(self):
        """relation_factor should appear in model_dump output."""
        node = GraphNode(
            id="test-1",
            document="test doc",
            metadata=NodeMetadata(
                type=NodeType.CONCEPT,
                created_at=1000.0,
                last_interaction=1000.0,
            ),
            relation_factor=0.5,
        )
        data = node.model_dump()
        assert data["relation_factor"] == 0.5


# ============== GraphStore.get_relation_factors ==============


class TestGetRelationFactors:
    """Tests for GraphStore.get_relation_factors()."""

    def test_empty_candidates_returns_empty(self, temp_store: GraphStore):
        """Should return empty dict for empty candidate list."""
        factors = temp_store.get_relation_factors("anchor", [])
        assert factors == {}

    def test_no_edges_returns_zeros(self, temp_store: GraphStore):
        """Nodes with no edges should all get 0.0."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "candidate", NodeType.CONCEPT)

        factors = temp_store.get_relation_factors("a", ["a", "b"])
        # Anchor with no edges → total_edges=0 → all zeros
        assert factors["a"] == 0.0
        assert factors["b"] == 0.0

    def test_anchor_self_factor_is_one(self, temp_store: GraphStore):
        """Anchor node should have relation_factor=1.0 when it has edges."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "related", NodeType.CONCEPT)
        temp_store.add_edge(Edge(source="a", target="b"))

        factors = temp_store.get_relation_factors("a", ["a", "b"])
        assert factors["a"] == 1.0

    def test_connected_candidate_gets_positive_factor(self, temp_store: GraphStore):
        """Candidate with edges to anchor should get positive factor."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "connected", NodeType.CONCEPT)
        temp_store.add_node("c", "unconnected", NodeType.CONCEPT)
        temp_store.add_edge(Edge(source="a", target="b"))

        factors = temp_store.get_relation_factors("a", ["b", "c"])
        assert factors["b"] > 0.0
        assert factors["c"] == 0.0

    def test_factor_proportional_to_shared_edges(self, temp_store: GraphStore):
        """Factor should be edges_between / total_edges."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "one-edge", NodeType.CONCEPT)
        temp_store.add_node("c", "other", NodeType.CONCEPT)
        temp_store.add_edge(Edge(source="a", target="b", relation_type="rel1"))
        temp_store.add_edge(Edge(source="a", target="c", relation_type="rel2"))

        factors = temp_store.get_relation_factors("a", ["b", "c"])
        # anchor has 2 edges total, each candidate shares 1
        assert factors["b"] == pytest.approx(0.5)
        assert factors["c"] == pytest.approx(0.5)

    def test_reverse_edge_counted(self, temp_store: GraphStore):
        """Edges from candidate→anchor should also count."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "connected", NodeType.CONCEPT)
        # Edge goes b→a (reverse direction)
        temp_store.add_edge(Edge(source="b", target="a"))

        factors = temp_store.get_relation_factors("a", ["b"])
        assert factors["b"] > 0.0


# ============== GraphStore.enrich_context_nodes ==============


class TestEnrichContextNodes:
    """Tests for GraphStore.enrich_context_nodes()."""

    def test_empty_list(self, temp_store: GraphStore):
        """Should handle empty list gracefully."""
        result = temp_store.enrich_context_nodes([])
        assert result == []

    def test_single_node_gets_factor_one(self, temp_store: GraphStore):
        """Single node should get relation_factor=1.0."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        node = temp_store.get_node("a")
        assert node is not None

        result = temp_store.enrich_context_nodes([node])
        assert len(result) == 1
        assert result[0].relation_factor == 1.0

    def test_sets_relation_factor_on_all_nodes(self, temp_store: GraphStore):
        """Should set relation_factor on every node."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "second", NodeType.CONCEPT)
        temp_store.add_node("c", "third", NodeType.CONCEPT)

        nodes = [temp_store.get_node(nid) for nid in ["a", "b", "c"]]
        nodes = [n for n in nodes if n is not None]

        result = temp_store.enrich_context_nodes(nodes)
        for node in result:
            assert node.relation_factor is not None

    def test_connected_nodes_boosted(self, temp_store: GraphStore):
        """Nodes connected to anchor should be boosted over unconnected ones."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "connected", NodeType.CONCEPT)
        temp_store.add_node("c", "unconnected", NodeType.CONCEPT)
        temp_store.add_edge(Edge(source="a", target="b"))

        node_a = temp_store.get_node("a")
        node_b = temp_store.get_node("b")
        node_c = temp_store.get_node("c")
        assert node_a and node_b and node_c

        # Give all the same importance
        node_a.metadata.importance_score = 0.8
        node_b.metadata.importance_score = 0.5
        node_c.metadata.importance_score = 0.5

        result = temp_store.enrich_context_nodes([node_a, node_b, node_c])

        # b should have higher relation_factor than c
        b_node = next(n for n in result if n.id == "b")
        c_node = next(n for n in result if n.id == "c")
        assert b_node.relation_factor is not None
        assert c_node.relation_factor is not None
        assert b_node.relation_factor > c_node.relation_factor

    def test_resort_by_combined_score(self, temp_store: GraphStore):
        """Nodes should be re-sorted by importance * (1 + relation_factor)."""
        temp_store.add_node("a", "anchor", NodeType.CONCEPT)
        temp_store.add_node("b", "high-conn", NodeType.CONCEPT)
        temp_store.add_node("c", "no-conn", NodeType.CONCEPT)
        temp_store.add_edge(Edge(source="a", target="b"))

        node_a = temp_store.get_node("a")
        node_b = temp_store.get_node("b")
        node_c = temp_store.get_node("c")
        assert node_a and node_b and node_c

        # c has higher raw importance but no connection to anchor
        node_a.metadata.importance_score = 0.9
        node_b.metadata.importance_score = 0.6
        node_c.metadata.importance_score = 0.7

        result = temp_store.enrich_context_nodes([node_a, node_b, node_c])

        # After enrichment with relation_factor, b (connected) should
        # have combined score: 0.6 * (1 + 1.0) = 1.2 vs c: 0.7 * (1 + 0.0) = 0.7
        ids_in_order = [n.id for n in result]
        assert ids_in_order.index("b") < ids_in_order.index("c")


# ============== ResponseGenerator._format_context ==============


class TestFormatContext:
    """Tests for _format_context() with relation_factor."""

    def _make_node(
        self, doc: str, importance: float, relation_factor: float | None = None
    ) -> GraphNode:
        return GraphNode(
            id="n1",
            document=doc,
            metadata=NodeMetadata(
                type=NodeType.CONCEPT,
                created_at=1000.0,
                last_interaction=1000.0,
                importance_score=importance,
            ),
            relation_factor=relation_factor,
        )

    def test_includes_relevance_when_set(self):
        """Should include 'relevance:' when relation_factor is set."""
        from mind_map.rag.response_generator import ResponseGenerator

        mock_llm = MagicMock()
        gen = ResponseGenerator(mock_llm)
        node = self._make_node("test doc", 0.85, 0.63)

        context = gen._format_context([node])
        assert "relevance: 0.63" in context
        assert "importance: 0.85" in context

    def test_no_relevance_when_none(self):
        """Should omit 'relevance:' when relation_factor is None."""
        from mind_map.rag.response_generator import ResponseGenerator

        mock_llm = MagicMock()
        gen = ResponseGenerator(mock_llm)
        node = self._make_node("test doc", 0.85, None)

        context = gen._format_context([node])
        assert "relevance:" not in context
        assert "importance: 0.85" in context

    def test_empty_nodes(self):
        """Should return empty string for empty node list."""
        from mind_map.rag.response_generator import ResponseGenerator

        mock_llm = MagicMock()
        gen = ResponseGenerator(mock_llm)

        context = gen._format_context([])
        assert context == ""

    def test_multiple_nodes_formatted(self):
        """Should format multiple nodes with sequential numbering."""
        from mind_map.rag.response_generator import ResponseGenerator

        mock_llm = MagicMock()
        gen = ResponseGenerator(mock_llm)
        nodes = [
            self._make_node("first doc", 0.9, 1.0),
            self._make_node("second doc", 0.5, 0.3),
        ]
        # Give unique IDs
        nodes[0].id = "n1"
        nodes[1].id = "n2"

        context = gen._format_context(nodes)
        assert "[1]" in context
        assert "[2]" in context
        assert "first doc" in context
        assert "second doc" in context


# ============== End-to-end: ingestion + enrichment ==============


class TestEnrichmentEndToEnd:
    """End-to-end test: ingest memos, then verify enrichment works on query results."""

    def test_ingest_and_enrich_flow(self, temp_store: GraphStore):
        """Ingest related memos, query, enrich, verify relation factors."""
        from mind_map.app.pipeline import ingest_memo

        # Ingest related content
        success1, _, ids1 = ingest_memo(
            "Python is great for #DataScience and #MachineLearning",
            temp_store,
        )
        assert success1
        success2, _, ids2 = ingest_memo(
            "Using #DataScience techniques with #Python for analysis",
            temp_store,
        )
        assert success2

        # Query for similar nodes
        nodes = temp_store.query_similar("Python data science", n_results=5)
        assert len(nodes) > 0

        # Enrich — should not crash, all nodes get relation_factor
        enriched = temp_store.enrich_context_nodes(nodes)
        assert len(enriched) == len(nodes)
        for node in enriched:
            assert node.relation_factor is not None

    def test_enrich_preserves_all_nodes(self, temp_store: GraphStore):
        """Enrichment should not drop any nodes."""
        from mind_map.app.pipeline import ingest_memo

        ingest_memo("Machine learning is a field of AI", temp_store)
        ingest_memo("Deep learning uses neural networks", temp_store)
        ingest_memo("Natural language processing with transformers", temp_store)

        nodes = temp_store.query_similar("AI and machine learning", n_results=5)
        original_count = len(nodes)
        original_ids = {n.id for n in nodes}

        enriched = temp_store.enrich_context_nodes(nodes)
        enriched_ids = {n.id for n in enriched}

        assert len(enriched) == original_count
        assert enriched_ids == original_ids
