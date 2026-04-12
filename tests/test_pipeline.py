"""Tests for the ingestion pipeline."""

import tempfile
from pathlib import Path

import pytest

from mind_map.app.pipeline import PipelineState, build_ingestion_pipeline, create_storage_node, ingest_memo
from mind_map.core.schemas import ExtractionResult, FilterDecision, NodeType, RetrievalContext
from mind_map.rag.graph_store import GraphStore


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


class TestPipelineRetrievalStep:
    """Tests for the retrieval step in the pipeline."""

    def test_retrieval_step_populates_retrieval_context(self, temp_store: GraphStore):
        ingest_memo("Python is a programming language", temp_store)

        pipeline = build_ingestion_pipeline(temp_store)
        initial = PipelineState(raw_text="Tell me about Python language features")
        final = pipeline.invoke(initial)

        assert "retrieval" in final
        retrieval = final["retrieval"]
        assert hasattr(retrieval, "concepts")
        assert hasattr(retrieval, "entities")
        assert hasattr(retrieval, "tags")

    def test_retrieval_step_empty_for_new_graph(self, temp_store: GraphStore):
        pipeline = build_ingestion_pipeline(temp_store)
        initial = PipelineState(raw_text="Python is great for data science")
        final = pipeline.invoke(initial)

        retrieval = final.get("retrieval")
        assert retrieval is not None
        assert isinstance(retrieval.concepts, list)
        assert isinstance(retrieval.entities, list)
        assert isinstance(retrieval.tags, list)


class TestStorageBehavior:
    """Tests for storage behavior in the new pipeline."""

    def test_storage_links_new_concept_to_retrieved_concepts(self, temp_store: GraphStore):
        retrieved_concept = temp_store.add_node(
            node_id="known-node-001",
            document="Existing concept about Python",
            node_type=NodeType.CONCEPT,
        )

        storage_node_fn = create_storage_node(temp_store)

        extraction = ExtractionResult(
            summary="New Python note",
            tags=["#Python"],
            entities=[],
            relationships=[],
        )

        state = PipelineState(
            raw_text="New Python note",
            retrieval=RetrievalContext(concepts=[retrieved_concept], entities=[], tags=[]),
            filter_decision=FilterDecision(action="new", reason="new", summary="New Python note"),
            extraction=extraction,
        )

        result = storage_node_fn(state)

        edges = temp_store.get_edges(result["node_ids"][0])
        related_targets = [e.target for e in edges if e.relation_type == "related_context"]
        assert "known-node-001" in related_targets

    def test_standalone_entities_persisted(self, temp_store: GraphStore):
        storage_node_fn = create_storage_node(temp_store)

        extraction = ExtractionResult(
            summary="Note about machine learning",
            tags=["#ML"],
            entities=["TensorFlow", "Python"],
            relationships=[
                ["Python", "uses", "TensorFlow"],
            ],
        )

        state = PipelineState(
            raw_text="Note about machine learning",
            retrieval=RetrievalContext(concepts=[], entities=[], tags=[]),
            filter_decision=FilterDecision(action="new", reason="new", summary="Note about machine learning"),
            extraction=extraction,
        )

        result = storage_node_fn(state)
        node_ids = result["node_ids"]
        assert len(node_ids) >= 4

        tf_id = "entity_tensorflow"
        tf_node = temp_store.get_node(tf_id)
        assert tf_node is not None
        assert tf_node.document == "TensorFlow"

    def test_entity_mentions_edge_created_for_standalone(self, temp_store: GraphStore):
        storage_node_fn = create_storage_node(temp_store)

        extraction = ExtractionResult(
            summary="Kubernetes container note",
            tags=["#DevOps"],
            entities=["Kubernetes"],
            relationships=[],
        )

        state = PipelineState(
            raw_text="Kubernetes container note",
            retrieval=RetrievalContext(concepts=[], entities=[], tags=[]),
            filter_decision=FilterDecision(action="new", reason="new", summary="Kubernetes container note"),
            extraction=extraction,
        )

        result = storage_node_fn(state)
        concept_id = result["node_ids"][0]
        edges = temp_store.get_edges(concept_id)

        mentions_edges = [e for e in edges if e.relation_type == "mentions"]
        assert len(mentions_edges) >= 1
        k8s_edge = next((e for e in mentions_edges if e.target == "entity_kubernetes"), None)
        assert k8s_edge is not None


class TestIngestMemo:
    """Tests for ingest_memo function."""

    def test_ingest_substantive_text(self, temp_store: GraphStore):
        text = "Python is a programming language used for web development and data science."

        success, message, node_ids = ingest_memo(text, temp_store)

        assert success is True
        assert len(node_ids) > 0
        assert "Created" in message

    def test_ingest_short_text_discarded(self, temp_store: GraphStore):
        success, message, node_ids = ingest_memo("hi", temp_store)
        assert success is False
        assert "Discarded" in message
        assert len(node_ids) == 0

    def test_ingest_trivial_greeting_discarded(self, temp_store: GraphStore):
        success, message, _ = ingest_memo("hello", temp_store)
        assert success is False
        assert "Discarded" in message

    def test_ingest_exact_duplicate_skipped(self, temp_store: GraphStore):
        text = "Python is a programming language used for web development and data science."
        first_success, _, _ = ingest_memo(text, temp_store)
        second_success, second_message, second_node_ids = ingest_memo(text, temp_store)

        assert first_success is True
        assert second_success is False
        assert "Skipped duplicate" in second_message
        assert second_node_ids == []

    def test_ingest_with_hashtags(self, temp_store: GraphStore):
        text = "Learning about #Python and #MachineLearning today."
        success, _, node_ids = ingest_memo(text, temp_store)
        assert success is True
        assert len(node_ids) >= 2

    def test_ingest_with_source_id(self, temp_store: GraphStore):
        text = "Important meeting notes about the project architecture."
        source_id = "meeting-2024-01-15"

        success, _, node_ids = ingest_memo(text, temp_store, source_id=source_id)
        assert success is True
        node = temp_store.get_node(node_ids[0])
        assert node is not None
        assert node.metadata.original_source_id == source_id

    def test_ingest_creates_edges_for_tags(self, temp_store: GraphStore):
        text = "Exploring #API design patterns for #REST services."
        success, _, _ = ingest_memo(text, temp_store)
        assert success is True
        stats = temp_store.get_stats()
        assert stats["total_edges"] > 0

    def test_ingest_multiple_memos(self, temp_store: GraphStore):
        texts = [
            "First note about #Python programming.",
            "Second note about #Python and #Testing.",
            "Third note about #API development.",
        ]

        for text in texts:
            success, _, _ = ingest_memo(text, temp_store)
            assert success is True

        stats = temp_store.get_stats()
        assert stats["total_nodes"] >= 3

    def test_ingest_reuses_existing_tags(self, temp_store: GraphStore):
        ingest_memo("Learning #Python basics.", temp_store)
        stats_after_first = temp_store.get_stats()

        ingest_memo("Advanced #Python techniques.", temp_store)
        stats_after_second = temp_store.get_stats()

        assert stats_after_second["tag_nodes"] == stats_after_first["tag_nodes"]

    def test_ingest_extracts_capitalized_entities(self, temp_store: GraphStore):
        text = "Meeting with John about the React project at Google."
        success, _, node_ids = ingest_memo(text, temp_store)
        assert success is True
        assert len(node_ids) >= 1
