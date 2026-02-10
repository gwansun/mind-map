"""Tests for the MCP mind_map_report tool."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_map.core.schemas import Edge, NodeType
from mind_map.mcp.server import get_store, mind_map_report as _report_tool, stores
from mind_map.rag.graph_store import GraphStore

# @mcp.tool() wraps functions into FunctionTool objects; access the raw callable via .fn
_report = _report_tool.fn


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


def _add_concept(store: GraphStore, node_id: str, text: str, source_id: str | None = None):
    """Helper: add a concept node."""
    return store.add_node(node_id, text, NodeType.CONCEPT, source_id=source_id)


def _add_tag(store: GraphStore, node_id: str, tag_name: str):
    """Helper: add a tag node."""
    return store.add_node(node_id, tag_name, NodeType.TAG)


def _add_entity(store: GraphStore, node_id: str, entity_name: str):
    """Helper: add an entity node."""
    return store.add_node(node_id, entity_name, NodeType.ENTITY)


def _link(store: GraphStore, source: str, target: str, relation: str = "related_context"):
    """Helper: add an edge between two nodes."""
    store.add_edge(Edge(source=source, target=target, relation_type=relation))


class TestReportEmptyGraph:
    """Tests for report on an empty knowledge graph."""

    def test_empty_graph_returns_valid_json(self, temp_store: GraphStore):
        result = json.loads(_report())
        assert "summary" in result
        assert "top_nodes" in result

    def test_empty_graph_summary_all_zeros(self, temp_store: GraphStore):
        result = json.loads(_report())
        summary = result["summary"]
        assert summary["total_nodes"] == 0
        assert summary["total_edges"] == 0
        assert summary["concepts"] == 0
        assert summary["entities"] == 0
        assert summary["tags"] == 0

    def test_empty_graph_top_nodes_empty(self, temp_store: GraphStore):
        result = json.loads(_report())
        assert result["top_nodes"] == []

    def test_empty_graph_default_workspace(self, temp_store: GraphStore):
        result = json.loads(_report())
        assert result["summary"]["workspace"] == "default"


class TestReportSummary:
    """Tests for the summary section of the report."""

    def test_summary_counts_nodes_by_type(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "A concept")
        _add_concept(temp_store, "c2", "Another concept")
        _add_tag(temp_store, "t1", "python")
        _add_entity(temp_store, "e1", "Google")

        result = json.loads(_report())
        summary = result["summary"]
        assert summary["total_nodes"] == 4
        assert summary["concepts"] == 2
        assert summary["tags"] == 1
        assert summary["entities"] == 1

    def test_summary_counts_edges(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "A concept")
        _add_tag(temp_store, "t1", "python")
        _add_tag(temp_store, "t2", "coding")
        _link(temp_store, "c1", "t1", "tagged_as")
        _link(temp_store, "c1", "t2", "tagged_as")

        result = json.loads(_report())
        assert result["summary"]["total_edges"] == 2

    def test_summary_avg_connections(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Node A")
        _add_concept(temp_store, "c2", "Node B")
        _link(temp_store, "c1", "c2")
        # 1 edge, 2 nodes => avg = (1*2)/2 = 1.0
        result = json.loads(_report())
        assert result["summary"]["avg_connections"] == 1.0


class TestReportTopNodes:
    """Tests for the top_nodes section of the report."""

    def test_single_node_appears_in_top(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Only concept")

        result = json.loads(_report())
        assert len(result["top_nodes"]) == 1
        assert result["top_nodes"][0]["id"] == "c1"
        assert result["top_nodes"][0]["document"] == "Only concept"
        assert result["top_nodes"][0]["type"] == "concept"

    def test_top_nodes_capped_at_five(self, temp_store: GraphStore):
        for i in range(8):
            _add_concept(temp_store, f"c{i}", f"Concept {i}")

        result = json.loads(_report())
        assert len(result["top_nodes"]) == 5

    def test_top_nodes_sorted_by_importance(self, temp_store: GraphStore):
        # Create nodes with varying connectivity — more edges = higher importance
        _add_concept(temp_store, "low", "Low importance")
        _add_concept(temp_store, "high", "High importance")
        _add_tag(temp_store, "t1", "tag1")
        _add_tag(temp_store, "t2", "tag2")
        _add_tag(temp_store, "t3", "tag3")

        # "high" gets 3 edges, "low" gets 0
        _link(temp_store, "high", "t1", "tagged_as")
        _link(temp_store, "high", "t2", "tagged_as")
        _link(temp_store, "high", "t3", "tagged_as")

        result = json.loads(_report())
        top_ids = [n["id"] for n in result["top_nodes"]]
        # "high" should be first (most connections)
        assert top_ids[0] == "high"

    def test_top_node_has_importance_score(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "A concept")

        result = json.loads(_report())
        node = result["top_nodes"][0]
        assert "importance_score" in node
        assert isinstance(node["importance_score"], float)
        assert 0.0 <= node["importance_score"] <= 1.0

    def test_top_node_has_connection_count(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "A concept")
        _add_tag(temp_store, "t1", "python")
        _link(temp_store, "c1", "t1", "tagged_as")

        result = json.loads(_report())
        node = next(n for n in result["top_nodes"] if n["id"] == "c1")
        assert node["connection_count"] == 1


class TestReportEdges:
    """Tests for edges included in top_nodes."""

    def test_node_includes_its_edges(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Main concept")
        _add_tag(temp_store, "t1", "python")
        _link(temp_store, "c1", "t1", "tagged_as")

        result = json.loads(_report())
        node = next(n for n in result["top_nodes"] if n["id"] == "c1")
        assert len(node["edges"]) == 1
        edge = node["edges"][0]
        assert edge["source"] == "c1"
        assert edge["target"] == "t1"
        assert edge["relation_type"] == "tagged_as"

    def test_edge_includes_weight(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Concept")
        _add_concept(temp_store, "c2", "Related")
        _link(temp_store, "c1", "c2")

        result = json.loads(_report())
        node = next(n for n in result["top_nodes"] if n["id"] in ("c1", "c2"))
        assert "weight" in node["edges"][0]

    def test_node_with_no_edges_has_empty_list(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Isolated concept")

        result = json.loads(_report())
        assert result["top_nodes"][0]["edges"] == []


class TestReportTags:
    """Tests for connected tags included in top_nodes."""

    def test_node_includes_connected_tags(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Main concept")
        _add_tag(temp_store, "t1", "python")
        _add_tag(temp_store, "t2", "coding")
        _link(temp_store, "c1", "t1", "tagged_as")
        _link(temp_store, "c1", "t2", "tagged_as")

        result = json.loads(_report())
        node = next(n for n in result["top_nodes"] if n["id"] == "c1")
        assert sorted(node["tags"]) == ["coding", "python"]

    def test_non_tag_neighbors_excluded_from_tags(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Main concept")
        _add_entity(temp_store, "e1", "Google")
        _add_tag(temp_store, "t1", "search")
        _link(temp_store, "c1", "e1", "mentions")
        _link(temp_store, "c1", "t1", "tagged_as")

        result = json.loads(_report())
        node = next(n for n in result["top_nodes"] if n["id"] == "c1")
        # Only the tag should appear, not the entity
        assert node["tags"] == ["search"]

    def test_node_with_no_tag_neighbors(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Concept A")
        _add_concept(temp_store, "c2", "Concept B")
        _link(temp_store, "c1", "c2")

        result = json.loads(_report())
        node = next(n for n in result["top_nodes"] if n["id"] == "c1")
        assert node["tags"] == []


class TestReportWorkspace:
    """Tests for multi-workspace support."""

    def test_custom_workspace_in_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "mind_map.mcp.server.DEFAULT_DATA_DIR", Path(tmpdir)
            ):
                result = json.loads(_report(workspace_id="alice"))
                assert result["summary"]["workspace"] == "alice"

    def test_separate_workspaces_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "mind_map.mcp.server.DEFAULT_DATA_DIR", Path(tmpdir)
            ):
                # Add data to workspace "alice"
                alice_store = get_store("alice")
                _add_concept(alice_store, "a1", "Alice concept")

                # Workspace "bob" should be empty
                result_bob = json.loads(_report(workspace_id="bob"))
                assert result_bob["summary"]["total_nodes"] == 0

                result_alice = json.loads(_report(workspace_id="alice"))
                assert result_alice["summary"]["total_nodes"] == 1


class TestReportJsonStructure:
    """Tests for the JSON output structure and data types."""

    def test_output_is_valid_json(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Test concept")
        raw = _report()
        result = json.loads(raw)
        assert isinstance(result, dict)

    def test_summary_has_all_expected_keys(self, temp_store: GraphStore):
        result = json.loads(_report())
        expected_keys = {
            "workspace", "total_nodes", "total_edges",
            "concepts", "entities", "tags", "avg_connections",
        }
        assert set(result["summary"].keys()) == expected_keys

    def test_top_node_has_all_expected_keys(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Test")
        result = json.loads(_report())
        expected_keys = {
            "id", "document", "type", "importance_score",
            "connection_count", "edges", "tags",
        }
        assert set(result["top_nodes"][0].keys()) == expected_keys

    def test_error_returns_json_with_error_key(self):
        """If the store raises, we get a JSON error instead of a crash."""
        stores.clear()
        with patch("mind_map.mcp.server.get_store", side_effect=RuntimeError("boom")):
            raw = _report()
            result = json.loads(raw)
            assert "error" in result
            assert "boom" in result["error"]
