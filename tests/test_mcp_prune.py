"""Tests for the MCP mind_map_prune tool."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_map.core.schemas import Edge, NodeType
from mind_map.mcp.server import get_store, mind_map_prune as _prune_tool, stores
from mind_map.rag.graph_store import GraphStore

# @mcp.tool() wraps functions into FunctionTool objects; access the raw callable via .fn
_prune = _prune_tool.fn


@pytest.fixture(autouse=True)
def clear_stores():
    """Clear the global stores dict before each test."""
    stores.clear()
    yield
    stores.clear()


@pytest.fixture
def temp_store():
    """Create a temporary GraphStore and register it as the default workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        stores["default"] = store
        yield store


def _add_concept(store: GraphStore, node_id: str, text: str):
    """Helper: add a concept node."""
    return store.add_node(node_id, text, NodeType.CONCEPT)


def _add_tag(store: GraphStore, node_id: str, tag_name: str):
    """Helper: add a tag node."""
    return store.add_node(node_id, tag_name, NodeType.TAG)


def _add_entity(store: GraphStore, node_id: str, entity_name: str):
    """Helper: add an entity node."""
    return store.add_node(node_id, entity_name, NodeType.ENTITY)


def _link(store: GraphStore, source: str, target: str, relation: str = "related_context"):
    """Helper: add an edge between two nodes."""
    store.add_edge(Edge(source=source, target=target, relation_type=relation))


class TestPruneEmptyGraph:
    """Tests for prune on an empty knowledge graph."""

    def test_empty_graph_returns_valid_json(self, temp_store: GraphStore):
        result = json.loads(_prune())
        assert "deleted_nodes" in result
        assert "deleted_tags" in result
        assert "deleted_edges_count" in result
        assert "summary" in result

    def test_empty_graph_deletes_nothing(self, temp_store: GraphStore):
        result = json.loads(_prune())
        assert result["deleted_nodes"] == []
        assert result["deleted_tags"] == []
        assert result["deleted_edges_count"] == 0

    def test_empty_graph_summary_mentions_empty(self, temp_store: GraphStore):
        result = json.loads(_prune())
        assert "empty" in result["summary"].lower()


class TestPruneNodeSelection:
    """Tests for correct selection of bottom 10% nodes."""

    def test_single_node_pruned(self, temp_store: GraphStore):
        """A single concept node should be pruned (floor(1*0.1) = 0, clamped to 1)."""
        _add_concept(temp_store, "c1", "Only concept")
        result = json.loads(_prune())
        assert len(result["deleted_nodes"]) == 1
        assert result["deleted_nodes"][0]["id"] == "c1"

    def test_prune_at_least_one_node(self, temp_store: GraphStore):
        """With fewer than 10 nodes, at least 1 should still be pruned."""
        for i in range(5):
            _add_concept(temp_store, f"c{i}", f"Concept {i}")
        result = json.loads(_prune())
        assert len(result["deleted_nodes"]) >= 1

    def test_prune_ten_percent(self, temp_store: GraphStore):
        """With 20 concepts, exactly 2 should be pruned (floor(20*0.1))."""
        for i in range(20):
            _add_concept(temp_store, f"c{i}", f"Concept {i}")
        result = json.loads(_prune())
        assert len(result["deleted_nodes"]) == 2

    def test_lowest_importance_pruned(self, temp_store: GraphStore):
        """The node with lowest importance (fewest edges) should be pruned."""
        # Create 10 concepts so floor(10*0.1) = 1 pruned
        for i in range(10):
            _add_concept(temp_store, f"c{i}", f"Concept {i}")
        # Give c1 through c9 edges to boost importance; leave c0 isolated
        for i in range(1, 10):
            _link(temp_store, f"c{i}", f"c{(i % 9) + 1}")

        result = json.loads(_prune())
        pruned_ids = {n["id"] for n in result["deleted_nodes"]}
        assert "c0" in pruned_ids

    def test_tags_not_direct_prune_candidates(self, temp_store: GraphStore):
        """Tags should not appear in deleted_nodes — only in deleted_tags."""
        _add_concept(temp_store, "c1", "Concept")
        _add_tag(temp_store, "t1", "python")
        _link(temp_store, "c1", "t1", "tagged_as")

        result = json.loads(_prune())
        pruned_ids = {n["id"] for n in result["deleted_nodes"]}
        assert "t1" not in pruned_ids

    def test_entities_are_prune_candidates(self, temp_store: GraphStore):
        """Entity nodes should be eligible for pruning."""
        _add_entity(temp_store, "e1", "Isolated entity")
        result = json.loads(_prune())
        pruned_ids = {n["id"] for n in result["deleted_nodes"]}
        assert "e1" in pruned_ids

    def test_nodes_removed_from_store(self, temp_store: GraphStore):
        """Pruned nodes should no longer exist in the store."""
        _add_concept(temp_store, "c1", "To be pruned")
        _prune()
        assert temp_store.get_node("c1") is None


class TestPruneEdgeRemoval:
    """Tests for edge cleanup during pruning."""

    def test_edges_deleted_for_pruned_node(self, temp_store: GraphStore):
        """Edges connected to pruned nodes should be removed."""
        _add_concept(temp_store, "c1", "Prunable concept")
        _add_concept(temp_store, "c2", "Surviving concept")
        _link(temp_store, "c1", "c2")

        result = json.loads(_prune())
        assert result["deleted_edges_count"] >= 1
        # Surviving node should have no edges left to the pruned node
        remaining_edges = temp_store.get_edges("c2")
        connected_ids = {e.source for e in remaining_edges} | {e.target for e in remaining_edges}
        pruned_ids = {n["id"] for n in result["deleted_nodes"]}
        assert not connected_ids.intersection(pruned_ids)

    def test_surviving_node_connection_count_updated(self, temp_store: GraphStore):
        """After pruning, surviving nodes should have updated connection counts."""
        # 10 concepts: c0 is isolated, c1-c9 form a chain
        for i in range(10):
            _add_concept(temp_store, f"c{i}", f"Concept {i}")
        for i in range(1, 9):
            _link(temp_store, f"c{i}", f"c{i+1}")

        _prune()  # c0 is pruned (isolated, lowest importance)
        # c1 had edges to c2; check it still reflects correct count
        node = temp_store.get_node("c1")
        assert node is not None
        assert node.metadata.connection_count == 1  # only edge to c2


class TestPruneTagHandling:
    """Tests for tag cleanup logic during pruning."""

    def test_single_connected_tag_removed(self, temp_store: GraphStore):
        """A tag connected only to pruned nodes should be removed."""
        _add_concept(temp_store, "c1", "Low importance concept")
        _add_tag(temp_store, "t1", "orphan-tag")
        _link(temp_store, "c1", "t1", "tagged_as")

        result = json.loads(_prune())
        deleted_tag_ids = {t["id"] for t in result["deleted_tags"]}
        assert "t1" in deleted_tag_ids
        assert temp_store.get_node("t1") is None

    def test_shared_tag_preserved(self, temp_store: GraphStore):
        """A tag connected to both pruned and surviving nodes should be kept."""
        # 10 concepts so only 1 pruned
        for i in range(10):
            _add_concept(temp_store, f"c{i}", f"Concept {i}")
        _add_tag(temp_store, "t_shared", "shared-tag")
        # c0 is isolated (lowest importance) — will be pruned
        # Give c1-c9 edges to boost them
        for i in range(1, 10):
            _link(temp_store, f"c{i}", f"c{(i % 9) + 1}")
        # Connect shared tag to both c0 (pruned) and c1 (surviving)
        _link(temp_store, "c0", "t_shared", "tagged_as")
        _link(temp_store, "c1", "t_shared", "tagged_as")

        result = json.loads(_prune())
        deleted_tag_ids = {t["id"] for t in result["deleted_tags"]}
        assert "t_shared" not in deleted_tag_ids
        assert temp_store.get_node("t_shared") is not None

    def test_tag_only_graph_prunes_nothing(self, temp_store: GraphStore):
        """A graph with only tags and no concepts/entities prunes nothing."""
        _add_tag(temp_store, "t1", "lonely-tag")
        _add_tag(temp_store, "t2", "another-tag")

        result = json.loads(_prune())
        assert result["deleted_nodes"] == []
        assert result["deleted_tags"] == []


class TestPruneReport:
    """Tests for the JSON report structure."""

    def test_report_has_all_keys(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Concept")
        result = json.loads(_prune())
        assert set(result.keys()) == {
            "deleted_nodes", "deleted_tags", "deleted_edges_count", "summary",
        }

    def test_deleted_node_has_expected_fields(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Test concept")
        result = json.loads(_prune())
        node = result["deleted_nodes"][0]
        assert set(node.keys()) == {"id", "document", "type"}
        assert node["type"] == "concept"

    def test_deleted_tag_has_expected_fields(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Concept")
        _add_tag(temp_store, "t1", "python")
        _link(temp_store, "c1", "t1", "tagged_as")

        result = json.loads(_prune())
        if result["deleted_tags"]:
            tag = result["deleted_tags"][0]
            assert set(tag.keys()) == {"id", "document"}

    def test_summary_mentions_counts(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Concept")
        result = json.loads(_prune())
        summary = result["summary"]
        assert "1" in summary  # at least the node count
        assert "default" in summary  # workspace name

    def test_edges_count_is_integer(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Concept")
        result = json.loads(_prune())
        assert isinstance(result["deleted_edges_count"], int)

    def test_error_returns_json_with_error_key(self):
        """If the store raises, we get a JSON error instead of a crash."""
        stores.clear()
        with patch("mind_map.mcp.server.get_store", side_effect=RuntimeError("boom")):
            raw = _prune()
            result = json.loads(raw)
            assert "error" in result
            assert "boom" in result["error"]


class TestPruneWorkspace:
    """Tests for multi-workspace isolation."""

    def test_prune_uses_specified_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mind_map.mcp.server.DEFAULT_DATA_DIR", Path(tmpdir)):
                alice_store = get_store("alice")
                _add_concept(alice_store, "a1", "Alice concept")

                result = json.loads(_prune(workspace_id="alice"))
                assert len(result["deleted_nodes"]) == 1
                assert result["deleted_nodes"][0]["id"] == "a1"

    def test_prune_does_not_affect_other_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mind_map.mcp.server.DEFAULT_DATA_DIR", Path(tmpdir)):
                alice_store = get_store("alice")
                bob_store = get_store("bob")
                _add_concept(alice_store, "a1", "Alice concept")
                _add_concept(bob_store, "b1", "Bob concept")

                _prune(workspace_id="alice")
                # Bob's data should be untouched
                assert bob_store.get_node("b1") is not None

    def test_prune_summary_includes_workspace_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mind_map.mcp.server.DEFAULT_DATA_DIR", Path(tmpdir)):
                ws_store = get_store("myspace")
                _add_concept(ws_store, "c1", "Concept")

                result = json.loads(_prune(workspace_id="myspace"))
                assert "myspace" in result["summary"]
