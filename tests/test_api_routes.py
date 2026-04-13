"""Tests for FastAPI routes in mind_map.app.api.routes."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mind_map.core.schemas import Edge, NodeType
from mind_map.app.pipeline import ingest_memo
from mind_map.rag.graph_store import GraphStore


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_store(temp_data_dir: Path):
    """Create and initialize a GraphStore in a temp directory."""
    store = GraphStore(temp_data_dir)
    store.initialize()
    return store


@pytest.fixture
def client(temp_data_dir: Path, mock_store: GraphStore):
    """Create a TestClient with a mocked get_store that returns our temp store."""
    with patch("mind_map.app.api.routes.get_store", return_value=mock_store):
        from mind_map.app.api.routes import app
        with TestClient(app) as test_client:
            yield test_client


class TestMemoRouteBehavior:
    """Tests for memo ingestion behavior through shared route dependencies."""

    def test_ingest_duplicate_returns_no_new_nodes(self, mock_store: GraphStore):
        text = "Python is a programming language used for web development and data science."
        first_success, _, first_ids = ingest_memo(text, mock_store)
        second_success, second_message, second_ids = ingest_memo(text, mock_store)

        assert first_success is True
        assert len(first_ids) > 0
        assert second_success is False
        assert "Skipped duplicate" in second_message
        assert second_ids == []


class TestDeleteNode:
    """Tests for DELETE /node/{node_id}."""

    def test_delete_node_success(self, client: TestClient, mock_store: GraphStore):
        """Test deleting an existing node returns 200 and cascades edges."""
        # Setup: create two nodes and connect them
        mock_store.add_node("node_a", "Node A content", NodeType.CONCEPT)
        mock_store.add_node("node_b", "Node B content", NodeType.ENTITY)
        mock_store.add_edge(
            Edge(source="node_a", target="node_b", weight=1.0, relation_type="related_to")
        )

        # Verify edge exists
        assert len(mock_store.get_edges("node_a")) == 1
        assert len(mock_store.get_edges("node_b")) == 1

        # Act
        response = client.delete("/node/node_a")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "node_a"
        assert data["deleted_tag_ids"] == []
        assert data["deleted_edges_count"] == 1

        # Node should be gone from ChromaDB
        assert mock_store.get_node("node_a") is None

        # Edge should be gone
        assert mock_store.get_edges("node_a") == []
        assert mock_store.get_edges("node_b") == []

    def test_delete_node_with_multiple_edges(self, client: TestClient, mock_store: GraphStore):
        """Test deleting a node with multiple connections cascades all edges."""
        # Setup: one node connected to three others
        mock_store.add_node("node_center", "Center node", NodeType.CONCEPT)
        mock_store.add_node("node_1", "Neighbor 1", NodeType.TAG)
        mock_store.add_node("node_2", "Neighbor 2", NodeType.ENTITY)
        mock_store.add_node("node_3", "Neighbor 3", NodeType.CONCEPT)

        mock_store.add_edge(Edge(source="node_center", target="node_1", weight=1.0, relation_type="tagged_as"))
        mock_store.add_edge(Edge(source="node_center", target="node_2", weight=0.8, relation_type="related_to"))
        mock_store.add_edge(Edge(source="node_3", target="node_center", weight=1.0, relation_type="related_to"))

        # Act
        response = client.delete("/node/node_center")

        # Assert
        assert response.status_code == 200
        # node_center is a concept, so node_1 (TAG) is also deleted
        assert response.json()["node_id"] == "node_center"
        assert "node_1" in response.json()["deleted_tag_ids"]
        # Only three distinct edges existed and all should be removed
        assert response.json()["deleted_edges_count"] == 3

        # Center node gone
        assert mock_store.get_node("node_center") is None
        # node_1 (tag) also gone
        assert mock_store.get_node("node_1") is None

        # Neighbors node_2 and node_3 still exist but no edges
        assert mock_store.get_node("node_2") is not None
        assert mock_store.get_node("node_3") is not None
        assert mock_store.get_edges("node_2") == []
        assert mock_store.get_edges("node_3") == []

    def test_delete_node_not_found(self, client: TestClient):
        """Test deleting a non-existent node returns 404."""
        response = client.delete("/node/nonexistent_id")
        assert response.status_code == 404
        assert response.json()["detail"] == "Node not found"

    def test_delete_node_no_edges(self, client: TestClient, mock_store: GraphStore):
        """Test deleting a node with no edges works correctly."""
        mock_store.add_node("node_lone", "Lone node", NodeType.TAG)

        response = client.delete("/node/node_lone")

        assert response.status_code == 200
        assert response.json()["deleted_edges_count"] == 0
        assert response.json()["deleted_tag_ids"] == []
        assert mock_store.get_node("node_lone") is None

    def test_delete_tag_node(self, client: TestClient, mock_store: GraphStore):
        """Test deleting a tag node cascade-deletes its edges but not neighboring concepts."""
        mock_store.add_node("concept", "A concept", NodeType.CONCEPT)
        mock_store.add_node("tag", "#important", NodeType.TAG)
        mock_store.add_edge(
            Edge(source="concept", target="tag", weight=1.0, relation_type="tagged_as")
        )

        response = client.delete("/node/tag")

        assert response.status_code == 200
        assert response.json()["deleted_tag_ids"] == []
        assert mock_store.get_node("tag") is None
        assert mock_store.get_node("concept") is not None
        assert mock_store.get_edges("concept") == []

    def test_delete_entity_node(self, client: TestClient, mock_store: GraphStore):
        """Test deleting an entity node cascade-deletes its edges but not neighboring tags."""
        mock_store.add_node("concept", "A concept", NodeType.CONCEPT)
        mock_store.add_node("tag", "#tag", NodeType.TAG)
        mock_store.add_node("entity", "Tesla Inc.", NodeType.ENTITY)
        mock_store.add_edge(
            Edge(source="concept", target="tag", weight=1.0, relation_type="tagged_as")
        )
        mock_store.add_edge(
            Edge(source="entity", target="tag", weight=1.0, relation_type="related_to")
        )

        response = client.delete("/node/entity")

        assert response.status_code == 200
        assert response.json()["deleted_tag_ids"] == []
        assert mock_store.get_node("entity") is None
        # tag and concept survive
        assert mock_store.get_node("tag") is not None
        assert mock_store.get_node("concept") is not None

    def test_delete_node_with_slashes_in_id(self, client: TestClient, mock_store: GraphStore):
        """Test deleting a node whose ID contains slashes works."""
        node_id = "entity_/users/gwansun/desktop/projects/ontologist"
        mock_store.add_node(node_id, "/Users/gwansun/Desktop/projects/ontologist", NodeType.ENTITY)

        response = client.delete("/node/entity_%2Fusers%2Fgwansun%2Fdesktop%2Fprojects%2Fontologist")

        assert response.status_code == 200
        assert response.json()["node_id"] == node_id
        assert mock_store.get_node(node_id) is None

    def test_delete_concept_removes_shared_tags(self, client: TestClient, mock_store: GraphStore):
        """Deleting a concept also removes shared tags even if another concept uses them."""
        mock_store.add_node("concept_a", "Concept A", NodeType.CONCEPT)
        mock_store.add_node("concept_b", "Concept B", NodeType.CONCEPT)
        mock_store.add_node("shared_tag", "#shared", NodeType.TAG)
        mock_store.add_edge(
            Edge(source="concept_a", target="shared_tag", weight=1.0, relation_type="tagged_as")
        )
        mock_store.add_edge(
            Edge(source="concept_b", target="shared_tag", weight=1.0, relation_type="tagged_as")
        )

        response = client.delete("/node/concept_a")

        assert response.status_code == 200
        assert "shared_tag" in response.json()["deleted_tag_ids"]
        assert mock_store.get_node("concept_a") is None
        assert mock_store.get_node("shared_tag") is None
        assert mock_store.get_node("concept_b") is not None
        # concept_b's edge to shared_tag is gone
        assert mock_store.get_edges("concept_b") == []

    def test_delete_concept_first_layer_only(self, client: TestClient, mock_store: GraphStore):
        """Deleting a concept does not recursively delete beyond first-layer tags."""
        # chain: concept -> tag -> entity
        mock_store.add_node("concept", "A concept", NodeType.CONCEPT)
        mock_store.add_node("tag", "#tag", NodeType.TAG)
        mock_store.add_node("entity", "An entity", NodeType.ENTITY)
        mock_store.add_edge(
            Edge(source="concept", target="tag", weight=1.0, relation_type="tagged_as")
        )
        mock_store.add_edge(
            Edge(source="tag", target="entity", weight=1.0, relation_type="related_to")
        )

        response = client.delete("/node/concept")

        assert response.status_code == 200
        assert "tag" in response.json()["deleted_tag_ids"]
        # entity (2 hops away) is NOT deleted
        assert mock_store.get_node("entity") is not None
        # but its edge to tag is gone
        assert mock_store.get_edges("entity") == []


class TestGetNode:
    """Tests for GET /node/{node_id}."""

    def test_get_node_success(self, client: TestClient, mock_store: GraphStore):
        """Test fetching an existing node returns full details."""
        mock_store.add_node("node_test", "Test content", NodeType.CONCEPT)
        mock_store.add_node("node_neighbor", "Neighbor content", NodeType.ENTITY)
        mock_store.add_edge(
            Edge(source="node_test", target="node_neighbor", weight=0.9, relation_type="related_to")
        )

        response = client.get("/node/node_test")

        assert response.status_code == 200
        data = response.json()
        assert data["node"]["id"] == "node_test"
        assert data["node"]["document"] == "Test content"
        assert len(data["edges"]) == 1
        assert data["edges"][0]["relation_type"] == "related_to"
        assert "importance_score" in data

    def test_get_node_with_slashes_in_id(self, client: TestClient, mock_store: GraphStore):
        """Test fetching a node whose ID contains slashes works."""
        node_id = "entity_plan/meta/mind_map_memos.md"
        mock_store.add_node(node_id, "plan/meta/mind_map_memos.md", NodeType.ENTITY)

        response = client.get("/node/entity_plan%2Fmeta%2Fmind_map_memos.md")

        assert response.status_code == 200
        data = response.json()
        assert data["node"]["id"] == node_id
        assert data["node"]["document"] == "plan/meta/mind_map_memos.md"

    def test_get_node_not_found(self, client: TestClient):
        """Test fetching a non-existent node returns 404."""
        response = client.get("/node/nonexistent_id")
        assert response.status_code == 404


class TestGetGraph:
    """Tests for GET /graph."""

    def test_get_graph_empty(self, client: TestClient, mock_store: GraphStore):
        """Test fetching an empty graph returns empty arrays."""
        response = client.get("/graph")
        assert response.status_code == 200
        assert response.json()["nodes"] == []
        assert response.json()["edges"] == []

    def test_get_graph_with_data(self, client: TestClient, mock_store: GraphStore):
        """Test fetching a graph with nodes and edges."""
        mock_store.add_node("node_1", "First node", NodeType.CONCEPT)
        mock_store.add_node("node_2", "Second node", NodeType.TAG)
        mock_store.add_edge(
            Edge(source="node_1", target="node_2", weight=1.0, relation_type="tagged_as")
        )

        response = client.get("/graph")

        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        edge_ids = {data["edges"][0]["source"], data["edges"][0]["target"]}
        assert edge_ids == {"node_1", "node_2"}


class TestStats:
    """Tests for GET /stats."""

    def test_stats_empty(self, client: TestClient, mock_store: GraphStore):
        """Test stats on empty graph."""
        response = client.get("/stats")
        assert response.status_code == 200
        assert response.json()["total_nodes"] == 0
        assert response.json()["total_edges"] == 0

    def test_stats_with_nodes(self, client: TestClient, mock_store: GraphStore):
        """Test stats with mixed node types and edges."""
        mock_store.add_node("concept", "A concept", NodeType.CONCEPT)
        mock_store.add_node("entity", "An entity", NodeType.ENTITY)
        mock_store.add_node("tag", "#tag", NodeType.TAG)
        mock_store.add_edge(Edge(source="concept", target="entity", weight=1.0))
        mock_store.add_edge(Edge(source="concept", target="tag", weight=1.0, relation_type="tagged_as"))

        response = client.get("/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 2
