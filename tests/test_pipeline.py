"""Tests for the ingestion pipeline."""

import tempfile
from pathlib import Path

import pytest

from mind_map.app.pipeline import (
    PipelineState,
    build_legacy_ingestion_pipeline,
    build_memo_cli_ingestion_pipeline,
    create_storage_node,
    ingest_memo_cli,
    ingest_memo_internal,
)
from mind_map.core.schemas import ExtractionResult, FilterDecision, NodeType, RetrievalContext
from mind_map.processor.cli_executor import OpenClawTarget
from mind_map.rag.graph_store import GraphStore


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


@pytest.fixture
def openclaw_target() -> OpenClawTarget:
    return OpenClawTarget(agent="minimax")


class TestPipelineRetrievalStep:
    """Tests for the retrieval step in the pipeline."""

    def test_memo_cli_ingestion_requires_target_at_call_boundary(self, temp_store: GraphStore):
        with pytest.raises(TypeError):
            ingest_memo_cli("Python is a programming language", temp_store)  # type: ignore[misc]

    def test_legacy_internal_ingestion_allowed(self, temp_store: GraphStore):
        success, message, node_ids = ingest_memo_internal(
            "Python is a programming language used for web development and data science.",
            temp_store,
        )
        assert success is True
        assert len(node_ids) > 0
        assert "Created" in message

    def test_retrieval_step_populates_retrieval_context(self, temp_store: GraphStore, openclaw_target: OpenClawTarget):
        pipeline = build_memo_cli_ingestion_pipeline(temp_store, target=openclaw_target)
        initial = PipelineState(raw_text="Tell me about Python language features", target=openclaw_target)
        final = pipeline.invoke(initial)

        assert "retrieval" in final
        retrieval = final["retrieval"]
        assert hasattr(retrieval, "concepts")
        assert hasattr(retrieval, "entities")
        assert hasattr(retrieval, "tags")

    def test_retrieval_step_empty_for_new_graph(self, temp_store: GraphStore, openclaw_target: OpenClawTarget):
        pipeline = build_memo_cli_ingestion_pipeline(temp_store, target=openclaw_target)
        initial = PipelineState(raw_text="Python is great for data science", target=openclaw_target)
        final = pipeline.invoke(initial)

        retrieval = final.get("retrieval")
        assert retrieval is not None
        assert isinstance(retrieval.concepts, list)
        assert isinstance(retrieval.entities, list)
        assert isinstance(retrieval.tags, list)

    def test_legacy_pipeline_builds(self, temp_store: GraphStore):
        pipeline = build_legacy_ingestion_pipeline(temp_store)
        assert pipeline is not None


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
