"""Tests for strict-target filter behavior and legacy heuristic ingestion."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mind_map.app.pipeline import (
    PipelineState,
    build_legacy_ingestion_pipeline,
    create_filter_node_legacy,
    ingest_memo_internal,
)
from mind_map.core.schemas import FilterDecision, NodeType, RetrievalContext
from mind_map.processor.cli_executor import CLIExecutionError, MiniMaxTarget
from mind_map.processor.filter_agent import FilterAgent, _format_retrieved_concepts
from mind_map.rag.graph_store import GraphStore


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


# =============================================================================
# FilterAgent tests
# =============================================================================


class TestFilterAgent:
    """Tests for FilterAgent evaluation with explicit target execution."""

    def test_target_execution_used_for_substantive_text(self):
        """FilterAgent uses the resolved explicit target for substantive text."""
        agent = FilterAgent(target=MiniMaxTarget(api_key="sk-test-for-filter"))

        with patch("mind_map.processor.filter_agent._run_custom_filter") as mock_run:
            mock_run.return_value = FilterDecision(action="new", reason="target result", summary="test")
            decision = agent.evaluate_sync("Python is great", [])

            assert decision.action == "new"
            mock_run.assert_called_once()

    def test_heuristic_discard_happens_before_target_execution(self):
        """Trivial text is discarded before any target execution."""
        agent = FilterAgent(target=MiniMaxTarget(api_key="sk-test-for-filter"))

        with patch("mind_map.processor.filter_agent._run_custom_filter") as mock_run:
            decision = agent.evaluate_sync("hi", [])

            assert decision.action == "discard"
            mock_run.assert_not_called()

    def test_duplicate_is_returned_by_target_after_heuristic_pass(self):
        """Duplicate classification can be returned by the explicit target after heuristic pass."""
        from mind_map.core.schemas import GraphNode, NodeMetadata

        concept = GraphNode(
            id="concept-1",
            document="Python is great",
            metadata=NodeMetadata(
                type=NodeType.CONCEPT,
                created_at=0.0,
                last_interaction=0.0,
                connection_count=0,
            ),
        )
        agent = FilterAgent(target=MiniMaxTarget(api_key="sk-test-for-filter"))

        with patch("mind_map.processor.filter_agent._run_custom_filter") as mock_run:
            mock_run.return_value = FilterDecision(
                action="duplicate",
                reason="Matches existing concept concept-1",
                summary=None,
            )
            decision = agent.evaluate_sync("Python is great", [concept])

            assert decision.action == "duplicate"
            mock_run.assert_called_once()

    def test_missing_target_raises_for_substantive_text(self):
        """Substantive text without a target raises immediately."""
        agent = FilterAgent(target=None)

        with pytest.raises(ValueError, match="Memo target is required for filter evaluation"):
            agent.evaluate_sync("Python is great", [])

    def test_cli_failure_propagates(self):
        """Explicit target execution failure is propagated, no fallback."""
        agent = FilterAgent(target=MiniMaxTarget(api_key="sk-test-for-filter"))

        with patch("mind_map.processor.filter_agent._run_custom_filter") as mock_run:
            mock_run.side_effect = CLIExecutionError("boom")
            with pytest.raises(CLIExecutionError, match="boom"):
                agent.evaluate_sync("Python is great", [])


# =============================================================================
# Integration: pipeline with filter_llm
# =============================================================================


class TestLegacyPipeline:
    """Integration tests for the legacy internal ingestion pipeline."""

    def test_pipeline_builds_without_target(self, temp_store: GraphStore):
        """build_legacy_ingestion_pipeline supports internal ingestion without a target."""
        pipeline = build_legacy_ingestion_pipeline(temp_store, llm=None)
        assert pipeline is not None

    def test_ingest_memo_internal_uses_heuristic_path(self, temp_store: GraphStore):
        """ingest_memo_internal processes substantive internal text without a memo target."""
        success, message, node_ids = ingest_memo_internal(
            "Python is a programming language",
            temp_store,
            llm=None,
        )

        assert success is True
        assert len(node_ids) > 0

    def test_ingest_memo_internal_duplicate_detection(self, temp_store: GraphStore):
        """ingest_memo_internal still rejects duplicates in the legacy internal flow."""
        substantive_text = "Python is a programming language"

        first_success, _, first_ids = ingest_memo_internal(
            substantive_text,
            temp_store,
            llm=None,
        )
        second_success, second_message, second_ids = ingest_memo_internal(
            substantive_text,
            temp_store,
            llm=None,
        )

        assert first_success is True
        assert len(first_ids) > 0
        assert second_success is False
        assert "Skipped duplicate" in second_message
        assert second_ids == []

    def test_ingest_memo_internal_short_text_uses_heuristic_first(self, temp_store: GraphStore):
        """Short text is discarded by heuristic before any external call in legacy flow."""
        with patch("subprocess.run") as mock_run:
            success, message, node_ids = ingest_memo_internal(
                "hi",
                temp_store,
                llm=None,
            )

            assert success is False
            assert "Discarded" in message
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


# =============================================================================
# Integration: legacy pipeline summarises through the injected LLM
# =============================================================================


class _StubChatLlm:
    """Minimal chat-model stand-in that records how often it was invoked."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.calls = 0
        self._payload = payload
        self._error = error

    def invoke(self, messages):
        self.calls += 1
        if self._error is not None:
            raise self._error
        from langchain_core.messages import AIMessage

        return AIMessage(content=json.dumps(self._payload))


class TestLegacyExtractionWithLlm:
    """The internal path must summarise via the LLM when one is injected."""

    SUBSTANTIVE = (
        "Kubernetes pods are the smallest deployable units; a ReplicaSet keeps a "
        "stable number of replicas running, and a Service gives them a stable "
        "network identity, while Ingress exposes HTTP routes from outside the cluster."
    )

    def test_llm_summary_is_used_and_stored(self, temp_store: GraphStore):
        """An injected LLM produces the stored summary, not a text truncation."""
        llm = _StubChatLlm(
            payload={
                "summary": "Kubernetes primitives: pods, ReplicaSets, Services, Ingress.",
                "tags": ["#k8s"],
                "entities": ["Kubernetes", "ReplicaSet"],
                "relationships": [["Ingress", "exposes", "HTTP routes"]],
            }
        )

        success, _message, node_ids = ingest_memo_internal(
            self.SUBSTANTIVE, temp_store, llm=llm
        )

        assert success is True
        assert llm.calls == 1
        concept = temp_store.get_node(node_ids[0])
        assert concept is not None
        assert concept.document == "Kubernetes primitives: pods, ReplicaSets, Services, Ingress."
        assert concept.document != self.SUBSTANTIVE[:200]

    def test_no_llm_keeps_heuristic_and_makes_no_call(self, temp_store: GraphStore):
        """llm=None preserves the free heuristic path (health checks rely on it)."""
        with patch("mind_map.app.pipeline._llm_extraction") as mock_extract:
            success, _message, node_ids = ingest_memo_internal(
                self.SUBSTANTIVE, temp_store, llm=None
            )

        assert success is True
        mock_extract.assert_not_called()
        concept = temp_store.get_node(node_ids[0])
        assert concept is not None
        assert concept.document == self.SUBSTANTIVE[:200]

    def test_llm_failure_falls_back_to_heuristic(self, temp_store: GraphStore, caplog):
        """A failing LLM must not lose the memo; it degrades and logs."""
        llm = _StubChatLlm(error=RuntimeError("gateway 503"))

        with caplog.at_level("WARNING", logger="mind_map.app.pipeline"):
            success, _message, node_ids = ingest_memo_internal(
                self.SUBSTANTIVE, temp_store, llm=llm
            )

        assert success is True
        assert llm.calls == 1
        concept = temp_store.get_node(node_ids[0])
        assert concept is not None
        assert concept.document == self.SUBSTANTIVE[:200]
        assert "LLM extraction failed" in caplog.text

    def test_non_json_response_falls_back_to_heuristic(self, temp_store: GraphStore):
        """A non-JSON LLM response degrades instead of storing garbage."""

        class _ProseLlm:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                from langchain_core.messages import AIMessage

                return AIMessage(content="Sure! Here is the JSON you asked for: ...")

        llm = _ProseLlm()
        success, _message, node_ids = ingest_memo_internal(
            self.SUBSTANTIVE, temp_store, llm=llm
        )

        assert success is True
        assert llm.calls == 1
        concept = temp_store.get_node(node_ids[0])
        assert concept is not None
        assert concept.document == self.SUBSTANTIVE[:200]
