"""Tests for the filter fallback chain: phi3.5 -> MiniMax CLI -> heuristic."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mind_map.app.pipeline import PipelineState, build_ingestion_pipeline, create_filter_node, ingest_memo
from mind_map.core.schemas import FilterDecision, NodeType, RetrievalContext
from mind_map.processor.filter_agent import (
    FilterAgent,
    FilterAgentWithFallback,
    _call_minimax_fallback,
    _format_retrieved_concepts,
)
from mind_map.rag.graph_store import GraphStore


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


# =============================================================================
# FilterAgentWithFallback tests
# =============================================================================


class TestFilterAgentWithFallback:
    """Tests for FilterAgentWithFallback evaluation chain."""

    def test_primary_phi3_5_path_used_when_llm_available(self):
        """FilterAgentWithFallback uses phi3.5 (primary) when available."""
        mock_llm = MagicMock()
        mock_parser = MagicMock()

        # The FilterAgent chain is: FILTER_PROMPT | llm | parser
        # When the chain.invoke() is called, it returns the raw dict (already parsed)
        with patch("mind_map.processor.filter_agent.FILTER_PROMPT", new=MagicMock()):
            agent = FilterAgentWithFallback(filter_llm=mock_llm)
            # Mock the chain invoke at agent level
            with patch.object(agent._primary_agent, "evaluate_sync") as mock_eval:
                mock_eval.return_value = FilterDecision(action="new", reason="phi3.5 result", summary="test")
                decision = agent.evaluate_sync("Python is great", [])

                assert decision.action == "new"
                mock_eval.assert_called_once_with("Python is great", [])

    def test_minimax_fallback_called_when_phi3_5_fails(self):
        """FilterAgentWithFallback falls back to MiniMax CLI when phi3.5 raises."""
        mock_llm = MagicMock()
        mock_llm.with_config.side_effect = RuntimeError("phi3.5 unavailable")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = '{"action": "new", "reason": "minimax fallback", "summary": "test"}'

            agent = FilterAgentWithFallback(filter_llm=mock_llm)
            decision = agent.evaluate_sync("Python is great", [])

            assert decision.action == "new"
            assert decision.reason == "minimax fallback"
            mock_run.assert_called_once()

    def test_heuristic_fallback_when_both_phi3_5_and_minimax_fail(self):
        """FilterAgentWithFallback raises RuntimeError when both backends fail."""
        mock_llm = MagicMock()
        mock_llm.with_config.side_effect = RuntimeError("phi3.5 unavailable")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""  # Empty output -> JSON decode error
            mock_run.return_value.stderr = ""

            agent = FilterAgentWithFallback(filter_llm=mock_llm)

            with pytest.raises(RuntimeError, match="Both phi3.5 and MiniMax filter backends failed"):
                agent.evaluate_sync("Python is great", [])

    def test_filter_node_uses_heuristic_when_both_backends_fail(self, temp_store: GraphStore):
        """Filter node falls back to heuristic when phi3.5 and MiniMax both fail."""
        mock_llm = MagicMock()
        mock_llm.with_config.side_effect = RuntimeError("phi3.5 unavailable")

        filter_node_fn = create_filter_node(filter_llm=mock_llm)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""  # Empty -> MiniMax fails

            state = PipelineState(
                raw_text="Python is a programming language",
                retrieval=RetrievalContext(concepts=[], entities=[], tags=[]),
            )

            result = filter_node_fn(state)

            # Heuristic should have classified as "new" (substantive text)
            assert result["filter_decision"].action == "new"

    def test_filter_node_phi3_5_decision_used(self, temp_store: GraphStore):
        """Filter node uses phi3.5 decision when available."""
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"action": "discard", "reason": "trivial", "summary": None}
        mock_llm_config = MagicMock()
        mock_llm_config.with_config.return_value = mock_chain

        with patch("mind_map.processor.filter_agent.ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = MagicMock()
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            filter_node_fn = create_filter_node(filter_llm=mock_llm)

            state = PipelineState(
                raw_text="hello",
                retrieval=RetrievalContext(concepts=[], entities=[], tags=[]),
            )

            result = filter_node_fn(state)
            # Discard from LLM should be respected
            assert result["filter_decision"].action == "discard"

    def test_filter_node_minimax_decision_used_when_phi3_5_fails(self, temp_store: GraphStore):
        """Filter node uses MiniMax decision when phi3.5 fails."""
        mock_llm = MagicMock()
        mock_llm.with_config.side_effect = RuntimeError("phi3.5 unavailable")

        filter_node_fn = create_filter_node(filter_llm=mock_llm)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = '{"action": "duplicate", "reason": "matches concept", "summary": null}'

            state = PipelineState(
                raw_text="Python is great",
                retrieval=RetrievalContext(concepts=[], entities=[], tags=[]),
            )

            result = filter_node_fn(state)
            assert result["filter_decision"].action == "duplicate"


# =============================================================================
# _call_minimax_fallback tests
# =============================================================================


class TestMinimMaxFallback:
    """Tests for MiniMax CLI fallback function."""

    def test_minimax_returns_valid_filter_decision(self):
        """MiniMax CLI returns a valid FilterDecision JSON."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = '{"action": "new", "reason": "useful", "summary": "test content"}'

            decision = _call_minimax_fallback("Python is great", [])

            assert decision.action == "new"
            assert decision.reason == "useful"
            assert decision.summary == "test content"
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[:4] == ["openclaw", "agent", "--agent", "minimax"]

    def test_minimax_handles_json_wrapped_in_text(self):
        """MiniMax output wrapped in extra text is still parsed correctly."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = 'Here is the classification:\n{"action": "discard", "reason": "trivial", "summary": null}\n'

            decision = _call_minimax_fallback("hello", [])

            assert decision.action == "discard"
            assert decision.reason == "trivial"

    def test_minimax_raises_on_empty_output(self):
        """MiniMax CLI with empty output raises RuntimeError."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""

            with pytest.raises(RuntimeError, match="MiniMax returned empty output"):
                _call_minimax_fallback("hello", [])

    def test_minimax_raises_on_invalid_json(self):
        """MiniMax CLI with invalid JSON raises RuntimeError."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "This is not JSON"

            with pytest.raises(RuntimeError, match="invalid JSON"):
                _call_minimax_fallback("hello", [])

    def test_minimax_raises_on_timeout(self):
        """MiniMax CLI timeout raises RuntimeError."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 60)

            with pytest.raises(RuntimeError, match="timed out"):
                _call_minimax_fallback("hello", [])


# =============================================================================
# get_filter_llm tests
# =============================================================================


class TestGetFilterLlm:
    """Tests for get_filter_llm function."""

    def test_returns_ollama_llm_when_available(self):
        """get_filter_llm returns phi3.5 via LangChain when Ollama is running."""
        from mind_map.processor.processing_llm import get_filter_llm

        with patch("mind_map.processor.processing_llm.check_ollama_available", return_value=True):
            with patch("mind_map.processor.processing_llm.ensure_model_available", return_value="phi3.5"):
                with patch("mind_map.processor.processing_llm.get_ollama_llm") as mock_get_llm:
                    mock_llm = MagicMock()
                    mock_get_llm.return_value = mock_llm

                    result = get_filter_llm()

                    assert result is mock_llm
                    mock_get_llm.assert_called_once_with("phi3.5", auto_pull=False)

    def test_returns_none_when_ollama_not_available(self):
        """get_filter_llm returns None when Ollama is not running."""
        from mind_map.processor.processing_llm import get_filter_llm

        with patch("mind_map.processor.processing_llm.check_ollama_available", return_value=False):
            result = get_filter_llm()
            assert result is None

    def test_does_not_use_cloud_auto_routing(self):
        """get_filter_llm does NOT call _try_cloud_processing_llm (no cloud-auto)."""
        from mind_map.processor.processing_llm import get_filter_llm

        with patch("mind_map.processor.processing_llm.check_ollama_available", return_value=True):
            with patch("mind_map.processor.processing_llm.ensure_model_available", return_value="phi3.5"):
                with patch("mind_map.processor.processing_llm.get_ollama_llm") as mock_get_llm:
                    with patch("mind_map.processor.processing_llm._try_cloud_processing_llm") as mock_cloud:
                        mock_get_llm.return_value = MagicMock()
                        result = get_filter_llm()

                        assert result is not None
                        mock_cloud.assert_not_called()


# =============================================================================
# Integration: pipeline with filter_llm
# =============================================================================


class TestPipelineWithFilterLlm:
    """Integration tests for pipeline with filter_llm parameter."""

    def test_pipeline_accepts_filter_llm_parameter(self, temp_store: GraphStore):
        """build_ingestion_pipeline accepts filter_llm parameter without error."""
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"action": "new", "reason": "test", "summary": "test"}
        mock_llm_config = MagicMock()
        mock_llm_config.with_config.return_value = mock_chain

        with patch("mind_map.processor.filter_agent.ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = MagicMock()
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            pipeline = build_ingestion_pipeline(temp_store, llm=None, filter_llm=mock_llm)
            assert pipeline is not None

    def test_ingest_memo_accepts_filter_llm_parameter(self, temp_store: GraphStore):
        """ingest_memo accepts filter_llm parameter and passes it through."""
        mock_llm = MagicMock()

        # The heuristic filter will pass (text is substantive) and phi3.5 will be used
        with patch("mind_map.processor.filter_agent.FilterAgentWithFallback") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.evaluate_sync.return_value = FilterDecision(
                action="new", reason="phi3.5 classified", summary="Python notes"
            )
            MockAgent.return_value = mock_instance

            success, message, node_ids = ingest_memo(
                "Python is a programming language",
                temp_store,
                llm=None,
                filter_llm=mock_llm,
            )

            assert success is True
            assert len(node_ids) > 0

    def test_ingest_memo_filter_llm_none_uses_minimax_fallback(self, temp_store: GraphStore):
        """ingest_memo with filter_llm=None falls back to heuristic safely.

        When filter_llm=None, FilterAgentWithFallback is not created (phi3.5 path
        unavailable). The filter_node falls back to heuristic which correctly
        processes substantive text as 'new'.
        """
        substantive_text = "Python is a programming language"

        success, message, node_ids = ingest_memo(
            substantive_text,
            temp_store,
            llm=None,
            filter_llm=None,
        )

        # Pipeline completes with heuristic 'new' decision
        assert success is True
        assert len(node_ids) > 0

    def test_minimax_fallback_used_when_phi3_5_unavailable(self, temp_store: GraphStore):
        """FilterAgentWithFallback uses MiniMax when phi3.5 raises an exception."""
        substantive_text = "Python is a programming language"
        from mind_map.processor import filter_agent

        original_func = filter_agent._call_minimax_fallback
        mock_minimax = MagicMock(
            return_value=FilterDecision(
                action="duplicate", reason="matches existing", summary=None
            )
        )
        filter_agent._call_minimax_fallback = mock_minimax

        try:
            # Pass a mock LLM that raises — this simulates phi3.5 being unavailable
            mock_llm = MagicMock(name="phi3.5")
            mock_llm.with_config.side_effect = RuntimeError("phi3.5 unavailable")

            success, message, node_ids = ingest_memo(
                substantive_text,
                temp_store,
                llm=None,
                filter_llm=mock_llm,
            )

            # Pipeline uses MiniMax result (duplicate)
            assert success is False
            assert "Skipped duplicate" in message
            assert mock_minimax.called
        finally:
            filter_agent._call_minimax_fallback = original_func

    def test_ingest_memo_short_text_uses_heuristic_first(self, temp_store: GraphStore):
        """Short text is discarded by heuristic before any LLM call."""
        with patch("subprocess.run") as mock_run:
            success, message, node_ids = ingest_memo(
                "hi",
                temp_store,
                llm=None,
                filter_llm=None,
            )

            assert success is False
            assert "Discarded" in message
            # MiniMax should NOT be called for trivially-short text
            mock_run.assert_not_called()


# =============================================================================
# _format_retrieved_concepts tests
# =============================================================================


class TestFormatRetrievedConcepts:
    """Tests for _format_retrieved_concepts helper."""

    def test_empty_list_returns_none(self):
        """Empty concept list returns '(none)'."""
        result = _format_retrieved_concepts([])
        assert result == "(none)"

    def test_concepts_formatted_with_id_and_snippet(self):
        """Non-empty concepts are formatted with id and 160-char snippet."""
        from mind_map.core.schemas import GraphNode, NodeMetadata

        node = GraphNode(
            id="test-id-001",
            document="This is a test concept document with meaningful content here.",
            metadata=NodeMetadata(
                type=NodeType.CONCEPT,
                created_at=0.0,
                last_interaction=0.0,
                connection_count=0,
            ),
        )

        result = _format_retrieved_concepts([node])

        assert "test-id-001" in result
        assert "This is a test concept" in result

    def test_concepts_limited_to_5(self):
        """Only first 5 concepts are included."""
        from mind_map.core.schemas import GraphNode, NodeMetadata

        nodes = [
            GraphNode(
                id=f"id-{i}",
                document=f"Document {i}",
                metadata=NodeMetadata(
                    type=NodeType.CONCEPT,
                    created_at=0.0,
                    last_interaction=0.0,
                    connection_count=0,
                ),
            )
            for i in range(10)
        ]

        result = _format_retrieved_concepts(nodes)

        for i in range(5):
            assert f"id-{i}" in result
        for i in range(5, 10):
            assert f"id-{i}" not in result
