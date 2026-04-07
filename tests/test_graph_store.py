"""Tests for GraphStore class."""

import tempfile
from pathlib import Path

import pytest

from mind_map.core.schemas import Edge, NodeType
from mind_map.rag.graph_store import GraphStore


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


class TestGetConnectedContext:
    """Tests for get_connected_context method."""

    def test_empty_node_list(self, temp_store: GraphStore):
        """Test with empty node_ids returns empty dict."""
        result = temp_store.get_connected_context([])
        assert result == {}

    def test_single_node_with_no_connections(self, temp_store: GraphStore):
        """Test single node with no edges returns empty list."""
        # Create a node without any edges
        temp_store.add_node(
            node_id="node_test_node",
            document="Test node",
            node_type=NodeType.CONCEPT,
        )

        result = temp_store.get_connected_context(["node_test_node"])
        assert result == {"node_test_node": []}

    def test_single_node_with_one_connection(self, temp_store: GraphStore):
        """Test node with one edge to another node."""
        # Create two nodes with an edge between them
        temp_store.add_node(
            node_id="node_node_a",
            document="Node A",
            node_type=NodeType.CONCEPT,
        )
        temp_store.add_node(
            node_id="node_node_b",
            document="Node B",
            node_type=NodeType.ENTITY,
        )
        temp_store.add_edge(
            Edge(source="node_node_a", target="node_node_b", weight=1.0, relation_type="related_to")
        )

        result = temp_store.get_connected_context(["node_node_a"])

        assert "node_node_a" in result
        assert len(result["node_node_a"]) == 1
        neighbor_node, edge = result["node_node_a"][0]
        assert neighbor_node.document == "Node B"
        assert edge.relation_type == "related_to"

    def test_multiple_nodes_with_shared_connections(self, temp_store: GraphStore):
        """Test multiple nodes share a common neighbor."""
        # Create three nodes: A, B both connected to C
        temp_store.add_node(node_id="node_node_a", document="Node A", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_node_b", document="Node B", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_node_c", document="Node C", node_type=NodeType.TAG)
        temp_store.add_edge(Edge(source="node_node_a", target="node_node_c", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="node_node_b", target="node_node_c", weight=1.0, relation_type="tagged_as"))

        result = temp_store.get_connected_context(["node_node_a", "node_node_b"])

        # Both A and B should show C as connected
        assert len(result["node_node_a"]) == 1
        assert len(result["node_node_b"]) == 1
        assert result["node_node_a"][0][0].id == "node_node_c"
        assert result["node_node_b"][0][0].id == "node_node_c"

    def test_filters_internal_connections_between_matched_nodes(self, temp_store: GraphStore):
        """Test that connections between matched nodes are NOT returned."""
        # Create A, B connected to each other, both also connected to C
        temp_store.add_node(node_id="node_node_a", document="Node A", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_node_b", document="Node B", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_node_c", document="Node C", node_type=NodeType.TAG)
        temp_store.add_edge(Edge(source="node_node_a", target="node_node_b", weight=1.0, relation_type="related_to"))
        temp_store.add_edge(Edge(source="node_node_a", target="node_node_c", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="node_node_b", target="node_node_c", weight=1.0, relation_type="tagged_as"))

        # Query for A and B
        result = temp_store.get_connected_context(["node_node_a", "node_node_b"])

        # A should only show C (not B, since B is in node_ids)
        assert len(result["node_node_a"]) == 1
        assert result["node_node_a"][0][0].id == "node_node_c"
        # B should only show C (not A)
        assert len(result["node_node_b"]) == 1
        assert result["node_node_b"][0][0].id == "node_node_c"

    def test_bidirectional_edge_handling(self, temp_store: GraphStore):
        """Test that edges work in both directions (source/target)."""
        temp_store.add_node(node_id="node_node_a", document="Node A", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_node_b", document="Node B", node_type=NodeType.ENTITY)
        # Edge from B to A (B is source)
        temp_store.add_edge(Edge(source="node_node_b", target="node_node_a", weight=1.0, relation_type="mentions"))

        # Query for A should still find the connection
        result = temp_store.get_connected_context(["node_node_a"])

        assert len(result["node_node_a"]) == 1
        assert result["node_node_a"][0][0].id == "node_node_b"

    def test_nonexistent_node_returns_empty(self, temp_store: GraphStore):
        """Test querying non-existent node IDs returns empty lists."""
        result = temp_store.get_connected_context(["nonexistent_id"])
        assert result == {"nonexistent_id": []}

    def test_multiple_edges_same_neighbor(self, temp_store: GraphStore):
        """Test multiple edges to the same neighbor (different relation types)."""
        temp_store.add_node(node_id="node_node_a", document="Node A", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_node_b", document="Node B", node_type=NodeType.ENTITY)
        temp_store.add_edge(Edge(source="node_node_a", target="node_node_b", weight=1.0, relation_type="mentions"))
        temp_store.add_edge(Edge(source="node_node_a", target="node_node_b", weight=1.0, relation_type="derived_from"))

        result = temp_store.get_connected_context(["node_node_a"])

        # Both edges should be returned
        assert len(result["node_node_a"]) == 2
        relation_types = {edge.relation_type for _, edge in result["node_node_a"]}
        assert relation_types == {"mentions", "derived_from"}


class TestGetEdges:
    """Tests for get_edges method."""

    def test_get_edges_empty(self, temp_store: GraphStore):
        """Test getting edges for node with no connections."""
        temp_store.add_node(node_id="node_lone_node", document="Lone node", node_type=NodeType.CONCEPT)

        edges = temp_store.get_edges("node_lone_node")
        assert edges == []

    def test_get_edges_returns_bidirectional(self, temp_store: GraphStore):
        """Test that get_edges finds edges regardless of direction."""
        temp_store.add_node(node_id="node_node_a", document="Node A", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_node_b", document="Node B", node_type=NodeType.ENTITY)
        temp_store.add_edge(Edge(source="node_node_a", target="node_node_b", weight=1.0, relation_type="related_to"))

        # Query from either direction should return the edge
        edges_from_a = temp_store.get_edges("node_node_a")
        edges_from_b = temp_store.get_edges("node_node_b")

        assert len(edges_from_a) == 1
        assert len(edges_from_b) == 1
        assert edges_from_a[0].relation_type == "related_to"
        assert edges_from_b[0].relation_type == "related_to"
