"""Tests for the ingestion pipeline."""

import tempfile
from pathlib import Path

import pytest

from mind_map.app.pipeline import ingest_memo
from mind_map.rag.graph_store import GraphStore


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


class TestIngestMemo:
    """Tests for ingest_memo function."""

    def test_ingest_substantive_text(self, temp_store: GraphStore):
        """Test ingesting substantive text creates nodes."""
        text = "Python is a programming language used for web development and data science."

        success, message, node_ids = ingest_memo(text, temp_store)

        assert success is True
        assert len(node_ids) > 0
        assert "Created" in message

    def test_ingest_short_text_discarded(self, temp_store: GraphStore):
        """Test that very short text is discarded."""
        text = "hi"

        success, message, node_ids = ingest_memo(text, temp_store)

        assert success is False
        assert "Discarded" in message
        assert len(node_ids) == 0

    def test_ingest_trivial_greeting_discarded(self, temp_store: GraphStore):
        """Test that trivial greetings are discarded."""
        text = "hello"

        success, message, node_ids = ingest_memo(text, temp_store)

        assert success is False
        assert "Discarded" in message

    def test_ingest_with_hashtags(self, temp_store: GraphStore):
        """Test ingesting text with hashtags extracts tags."""
        text = "Learning about #Python and #MachineLearning today."

        success, message, node_ids = ingest_memo(text, temp_store)

        assert success is True
        # Should have concept node + tag nodes
        assert len(node_ids) >= 2

    def test_ingest_with_source_id(self, temp_store: GraphStore):
        """Test ingesting with source_id preserves it."""
        text = "Important meeting notes about the project architecture."
        source_id = "meeting-2024-01-15"

        success, message, node_ids = ingest_memo(
            text, temp_store, source_id=source_id
        )

        assert success is True
        # Verify the node has the source_id in metadata
        node = temp_store.get_node(node_ids[0])
        assert node is not None
        assert node.metadata.original_source_id == source_id

    def test_ingest_creates_edges_for_tags(self, temp_store: GraphStore):
        """Test that ingesting creates edges between concept and tags."""
        text = "Exploring #API design patterns for #REST services."

        success, _, node_ids = ingest_memo(text, temp_store)

        assert success is True
        # Check that edges were created
        stats = temp_store.get_stats()
        assert stats["total_edges"] > 0

    def test_ingest_multiple_memos(self, temp_store: GraphStore):
        """Test ingesting multiple memos accumulates nodes."""
        texts = [
            "First note about #Python programming.",
            "Second note about #Python and #Testing.",
            "Third note about #API development.",
        ]

        total_nodes = 0
        for text in texts:
            success, _, node_ids = ingest_memo(text, temp_store)
            assert success is True
            total_nodes += len(node_ids)

        stats = temp_store.get_stats()
        # Should have multiple nodes (some tags may be shared)
        assert stats["total_nodes"] >= 3  # At least 3 concept nodes

    def test_ingest_reuses_existing_tags(self, temp_store: GraphStore):
        """Test that existing tags are reused, not duplicated."""
        # Ingest first memo with #Python tag
        ingest_memo("Learning #Python basics.", temp_store)
        stats_after_first = temp_store.get_stats()

        # Ingest second memo with same #Python tag
        ingest_memo("Advanced #Python techniques.", temp_store)
        stats_after_second = temp_store.get_stats()

        # Tag count should increase by 1 (concept) not 2 (concept + duplicate tag)
        # The #Python tag should be reused
        assert stats_after_second["tag_nodes"] == stats_after_first["tag_nodes"]

    def test_ingest_extracts_capitalized_entities(self, temp_store: GraphStore):
        """Test that capitalized words are extracted as potential entities."""
        text = "Meeting with John about the React project at Google."

        success, _, node_ids = ingest_memo(text, temp_store)

        assert success is True
        # Should create nodes for extracted entities
        assert len(node_ids) >= 1
