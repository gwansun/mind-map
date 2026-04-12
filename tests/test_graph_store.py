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

    def test_connected_context_sorted_deterministically(self, temp_store: GraphStore):
        """Test connected context is sorted by weight, importance, relation, document, id."""
        temp_store.add_node(node_id="node_anchor", document="Anchor", node_type=NodeType.CONCEPT)
        temp_store.add_node(node_id="node_b", document="Beta", node_type=NodeType.ENTITY)
        temp_store.add_node(node_id="node_a", document="Alpha", node_type=NodeType.ENTITY)
        temp_store.add_node(node_id="node_c", document="Gamma", node_type=NodeType.ENTITY)

        # Set explicit importance scores for deterministic secondary ordering
        temp_store.collection.update(ids=["node_a"], metadatas=[{
            "type": NodeType.ENTITY.value,
            "created_at": 0.0,
            "last_interaction": 0.0,
            "connection_count": 0,
            "importance_score": 0.9,
        }])
        temp_store.collection.update(ids=["node_b"], metadatas=[{
            "type": NodeType.ENTITY.value,
            "created_at": 0.0,
            "last_interaction": 0.0,
            "connection_count": 0,
            "importance_score": 0.2,
        }])
        temp_store.collection.update(ids=["node_c"], metadatas=[{
            "type": NodeType.ENTITY.value,
            "created_at": 0.0,
            "last_interaction": 0.0,
            "connection_count": 0,
            "importance_score": 0.9,
        }])

        temp_store.add_edge(Edge(source="node_anchor", target="node_b", weight=1.0, relation_type="mentions"))
        temp_store.add_edge(Edge(source="node_anchor", target="node_a", weight=1.0, relation_type="mentions"))
        temp_store.add_edge(Edge(source="node_anchor", target="node_c", weight=2.0, relation_type="mentions"))

        result = temp_store.get_connected_context(["node_anchor"])
        ordered_ids = [neighbor.id for neighbor, _ in result["node_anchor"]]

        # Highest weight first => node_c, then among equal weight higher importance => node_a, then node_b
        assert ordered_ids == ["node_c", "node_a", "node_b"]


class TestFirstHopNeighbors:
    """Tests for get_first_hop_neighbors method."""

    def test_returns_only_entity_and_tag_neighbors(self, temp_store: GraphStore):
        temp_store.add_node("concept_a", "Concept A", NodeType.CONCEPT)
        temp_store.add_node("entity_a", "Entity A", NodeType.ENTITY)
        temp_store.add_node("tag_a", "#tag", NodeType.TAG)
        temp_store.add_node("concept_b", "Concept B", NodeType.CONCEPT)
        temp_store.add_edge(Edge(source="concept_a", target="entity_a", relation_type="mentions"))
        temp_store.add_edge(Edge(source="concept_a", target="tag_a", relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="concept_a", target="concept_b", relation_type="related_context"))

        neighbors = temp_store.get_first_hop_neighbors(["concept_a"])
        neighbor_ids = {n.id for n in neighbors}

        assert neighbor_ids == {"entity_a", "tag_a"}

    def test_deduplicates_shared_neighbors(self, temp_store: GraphStore):
        temp_store.add_node("concept_a", "Concept A", NodeType.CONCEPT)
        temp_store.add_node("concept_b", "Concept B", NodeType.CONCEPT)
        temp_store.add_node("entity_shared", "Shared", NodeType.ENTITY)
        temp_store.add_edge(Edge(source="concept_a", target="entity_shared", relation_type="mentions"))
        temp_store.add_edge(Edge(source="concept_b", target="entity_shared", relation_type="mentions"))

        neighbors = temp_store.get_first_hop_neighbors(["concept_a", "concept_b"])
        neighbor_ids = [n.id for n in neighbors]

        assert neighbor_ids == ["entity_shared"]


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


class TestDeleteNodeTypeAware:
    """Tests for the type-aware delete_node method."""

    def test_concept_delete_removes_first_layer_tags(self, temp_store: GraphStore):
        """Deleting a concept removes all directly-connected tag neighbors."""
        temp_store.add_node("concept", "A concept", NodeType.CONCEPT)
        temp_store.add_node("tag1", "#tag1", NodeType.TAG)
        temp_store.add_node("tag2", "#tag2", NodeType.TAG)
        temp_store.add_edge(Edge(source="concept", target="tag1", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="concept", target="tag2", weight=1.0, relation_type="tagged_as"))

        result = temp_store.delete_node("concept")

        assert result.deleted_node_id == "concept"
        assert set(result.deleted_tag_ids) == {"tag1", "tag2"}
        assert result.deleted_edges_count == 2  # only the two concept-tag edges existed

        assert temp_store.get_node("concept") is None
        assert temp_store.get_node("tag1") is None
        assert temp_store.get_node("tag2") is None

    def test_concept_delete_shared_tags_are_deleted_too(self, temp_store: GraphStore):
        """Shared tags — connected to multiple concepts — are still deleted when one concept is deleted."""
        temp_store.add_node("concept1", "Concept 1", NodeType.CONCEPT)
        temp_store.add_node("concept2", "Concept 2", NodeType.CONCEPT)
        temp_store.add_node("shared_tag", "#shared", NodeType.TAG)
        temp_store.add_edge(Edge(source="concept1", target="shared_tag", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="concept2", target="shared_tag", weight=1.0, relation_type="tagged_as"))

        result = temp_store.delete_node("concept1")

        assert "shared_tag" in result.deleted_tag_ids
        assert temp_store.get_node("concept1") is None
        assert temp_store.get_node("shared_tag") is None
        # concept2 survives
        assert temp_store.get_node("concept2") is not None

    def test_concept_delete_removes_edges_from_deleted_tags_to_surviving_nodes(self, temp_store: GraphStore):
        """When a concept deletes its tag neighbors, edges from those tags to surviving nodes are also removed."""
        temp_store.add_node("concept", "A concept", NodeType.CONCEPT)
        temp_store.add_node("entity", "An entity", NodeType.ENTITY)
        temp_store.add_node("tag", "#tag", NodeType.TAG)
        temp_store.add_edge(Edge(source="concept", target="tag", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="tag", target="entity", weight=1.0, relation_type="related_to"))

        result = temp_store.delete_node("concept")

        assert "tag" in result.deleted_tag_ids
        # The tag and its edge to entity are gone
        assert temp_store.get_edges("tag") == []
        assert temp_store.get_edges("entity") == []
        # Entity itself survived
        assert temp_store.get_node("entity") is not None

    def test_delete_tag_directly_does_not_delete_neighboring_concepts(self, temp_store: GraphStore):
        """Deleting a tag node does NOT delete neighboring concepts — only tag+edges are removed."""
        temp_store.add_node("concept", "A concept", NodeType.CONCEPT)
        temp_store.add_node("tag", "#tag", NodeType.TAG)
        temp_store.add_edge(Edge(source="concept", target="tag", weight=1.0, relation_type="tagged_as"))

        result = temp_store.delete_node("tag")

        assert result.deleted_node_id == "tag"
        assert result.deleted_tag_ids == []  # No extra tags deleted
        assert temp_store.get_node("tag") is None
        assert temp_store.get_node("concept") is not None
        # Edge is gone but concept survives
        assert temp_store.get_edges("concept") == []

    def test_delete_entity_directly_does_not_delete_neighboring_tags(self, temp_store: GraphStore):
        """Deleting an entity node does NOT delete neighboring tags."""
        temp_store.add_node("concept", "A concept", NodeType.CONCEPT)
        temp_store.add_node("entity", "An entity", NodeType.ENTITY)
        temp_store.add_node("tag", "#tag", NodeType.TAG)
        temp_store.add_edge(Edge(source="concept", target="tag", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="entity", target="tag", weight=1.0, relation_type="related_to"))

        result = temp_store.delete_node("entity")

        assert result.deleted_node_id == "entity"
        assert result.deleted_tag_ids == []
        assert temp_store.get_node("entity") is None
        # Tag and concept survive
        assert temp_store.get_node("tag") is not None
        assert temp_store.get_node("concept") is not None

    def test_first_layer_only_no_recursive_tag_deletion(self, temp_store: GraphStore):
        """Deleting a concept only removes direct tag neighbors — not tags' neighbors."""
        # chain: concept -> tag1 -> entity
        temp_store.add_node("concept", "A concept", NodeType.CONCEPT)
        temp_store.add_node("tag1", "#tag1", NodeType.TAG)
        temp_store.add_node("entity", "An entity", NodeType.ENTITY)
        temp_store.add_edge(Edge(source="concept", target="tag1", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="tag1", target="entity", weight=1.0, relation_type="related_to"))

        result = temp_store.delete_node("concept")

        # tag1 is deleted but entity is NOT deleted (it's 2 hops away)
        assert "tag1" in result.deleted_tag_ids
        assert temp_store.get_node("tag1") is None
        assert temp_store.get_node("entity") is not None
        # The edge from tag1 to entity is gone
        assert temp_store.get_edges("entity") == []

    def test_delete_nonexistent_node_returns_empty_result(self, temp_store: GraphStore):
        """Deleting a non-existent node returns a valid result with zero counts."""
        result = temp_store.delete_node("nonexistent")

        assert result.deleted_node_id == "nonexistent"
        assert result.deleted_tag_ids == []
        assert result.deleted_edges_count == 0

    def test_concept_delete_no_tags_connected(self, temp_store: GraphStore):
        """Deleting a concept with no tag neighbors returns empty tag list."""
        temp_store.add_node("concept", "A concept", NodeType.CONCEPT)
        temp_store.add_node("entity", "An entity", NodeType.ENTITY)
        temp_store.add_edge(Edge(source="concept", target="entity", weight=1.0, relation_type="related_to"))

        result = temp_store.delete_node("concept")

        assert result.deleted_node_id == "concept"
        assert result.deleted_tag_ids == []
        assert result.deleted_edges_count == 1
        assert temp_store.get_node("concept") is None
        assert temp_store.get_node("entity") is not None

    def test_concept_delete_tag_also_connected_to_other_concepts(self, temp_store: GraphStore):
        """A tag shared between two concepts is fully removed when either concept is deleted."""
        temp_store.add_node("concept_a", "Concept A", NodeType.CONCEPT)
        temp_store.add_node("concept_b", "Concept B", NodeType.CONCEPT)
        temp_store.add_node("shared_tag", "#shared", NodeType.TAG)
        temp_store.add_edge(Edge(source="concept_a", target="shared_tag", weight=1.0, relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="concept_b", target="shared_tag", weight=1.0, relation_type="tagged_as"))

        result = temp_store.delete_node("concept_a")

        # shared_tag is deleted despite concept_b still referencing it
        assert "shared_tag" in result.deleted_tag_ids
        assert temp_store.get_node("shared_tag") is None
        # concept_b survives but its edge to shared_tag is gone
        assert temp_store.get_node("concept_b") is not None
        assert temp_store.get_edges("concept_b") == []
