"""Tests for the MCP mind_map_health tool."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_map.core.schemas import Edge, NodeType
from mind_map.mcp.server import get_store, mind_map_health as _health_tool, stores
from mind_map.rag.graph_store import GraphStore

# @mcp.tool() wraps functions into FunctionTool objects; access the raw callable via .fn
_health = _health_tool.fn


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


def _link(store: GraphStore, source: str, target: str, relation: str = "related_context"):
    """Helper: add an edge between two nodes."""
    store.add_edge(Edge(source=source, target=target, relation_type=relation))


# -- Mock helpers for external dependencies --

def _mock_ollama_up(monkeypatch=None):
    """Return patch context managers for Ollama available with a model."""
    return {
        "mind_map.mcp.server.check_ollama_available": True,
        "mind_map.mcp.server.get_selected_model": "phi3.5",
        "mind_map.mcp.server.get_available_models": ["phi3.5", "mistral"],
    }


def _mock_ollama_down():
    """Return patch values for Ollama unavailable."""
    return {
        "mind_map.mcp.server.check_ollama_available": False,
        "mind_map.mcp.server.get_selected_model": "phi3.5",
        "mind_map.mcp.server.get_available_models": [],
    }


def _mock_llm_status_online():
    """Return LLM status dict where both LLMs are online."""
    return {
        "processing_llm": {"model": "gemini-2.0-flash", "status": "online", "provider": "gemini"},
        "reasoning_llm": {"provider": "claude-cli", "status": "online"},
    }


def _mock_llm_status_offline():
    """Return LLM status dict where both LLMs are offline."""
    return {
        "processing_llm": {"model": "phi3.5", "status": "offline", "provider": "ollama"},
        "reasoning_llm": {"provider": "claude-cli", "status": "offline"},
    }


class TestHealthBasicStructure:
    """Tests for the basic JSON structure of the health check response."""

    def test_returns_valid_json(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert isinstance(result, dict)

    def test_has_required_top_level_keys(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert "status" in result
            assert "checks" in result
            assert "timestamp" in result
            assert "workspace" in result

    def test_status_is_valid_enum(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["status"] in ("healthy", "degraded", "unhealthy")

    def test_timestamp_is_numeric(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert isinstance(result["timestamp"], float)

    def test_default_workspace(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["workspace"] == "default"

    def test_custom_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mind_map.mcp.server.DEFAULT_DATA_DIR", Path(tmpdir)), \
                 patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
                 patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
                result = json.loads(_health(workspace_id="alice"))
                assert result["workspace"] == "alice"

    def test_checks_has_all_sections(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            checks = result["checks"]
            assert "ollama_connection" in checks
            assert "chromadb_connection" in checks
            assert "sqlite_connection" in checks
            assert "processing_llm" in checks
            assert "integration_tests" in checks

    def test_integration_tests_has_all_subtests(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            integration = result["checks"]["integration_tests"]
            assert "similarity_search" in integration
            assert "memo_ingestion" in integration
            assert "data_persistence" in integration


class TestHealthDatabaseChecks:
    """Tests for ChromaDB and SQLite connection checks."""

    def test_chromadb_pass_with_store(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["checks"]["chromadb_connection"]["status"] == "pass"

    def test_chromadb_reports_node_count(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Test concept")
        _add_concept(temp_store, "c2", "Another concept")
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["checks"]["chromadb_connection"]["node_count"] == 2

    def test_sqlite_pass_with_store(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["checks"]["sqlite_connection"]["status"] == "pass"

    def test_sqlite_reports_edge_count(self, temp_store: GraphStore):
        _add_concept(temp_store, "c1", "Node A")
        _add_concept(temp_store, "c2", "Node B")
        _link(temp_store, "c1", "c2")
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["checks"]["sqlite_connection"]["edge_count"] == 1

    def test_chromadb_fail_on_exception(self):
        """If get_store raises during ChromaDB check, it should fail gracefully."""
        stores.clear()
        with patch("mind_map.mcp.server.get_store", side_effect=RuntimeError("ChromaDB down")), \
             patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            # The outer try/except catches the failure of get_store for chromadb check
            # and subsequent checks also fail since they use get_store
            result = json.loads(_health())
            assert result["checks"]["chromadb_connection"]["status"] == "fail"


class TestHealthOllamaCheck:
    """Tests for Ollama connection checks."""

    def test_ollama_pass_when_running_with_model(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=True), \
             patch("mind_map.mcp.server.get_selected_model", return_value="phi3.5"), \
             patch("mind_map.mcp.server.get_available_models", return_value=["phi3.5", "mistral"]), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            check = result["checks"]["ollama_connection"]
            assert check["status"] == "pass"
            assert check["model"] == "phi3.5"

    def test_ollama_fail_when_server_down(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            check = result["checks"]["ollama_connection"]
            assert check["status"] == "fail"
            assert "not running" in check["details"]

    def test_ollama_fail_when_model_missing(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=True), \
             patch("mind_map.mcp.server.get_selected_model", return_value="nonexistent"), \
             patch("mind_map.mcp.server.get_available_models", return_value=["phi3.5"]), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            check = result["checks"]["ollama_connection"]
            assert check["status"] == "fail"
            assert "not found" in check["details"]

    def test_ollama_exception_handled(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", side_effect=RuntimeError("boom")), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            check = result["checks"]["ollama_connection"]
            assert check["status"] == "fail"
            assert "boom" in check["details"]


class TestHealthLLMStatus:
    """Tests for processing LLM status check."""

    def test_processing_llm_available(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_online()):
            result = json.loads(_health())
            assert result["checks"]["processing_llm"]["status"] == "available"
            assert result["checks"]["processing_llm"]["provider"] == "gemini"

    def test_processing_llm_unavailable(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["checks"]["processing_llm"]["status"] == "unavailable"

    def test_llm_status_exception_handled(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", side_effect=RuntimeError("status error")):
            result = json.loads(_health())
            assert result["checks"]["processing_llm"]["status"] == "unavailable"

    def test_processing_llm_has_model_field(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_online()):
            result = json.loads(_health())
            assert "model" in result["checks"]["processing_llm"]
            assert result["checks"]["processing_llm"]["model"] == "gemini-2.0-flash"

    def test_no_reasoning_llm_in_checks(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_online()):
            result = json.loads(_health())
            assert "reasoning_llm" not in result["checks"]


class TestHealthIntegrationSearch:
    """Tests for the similarity search integration test."""

    def test_similarity_search_passes(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            check = result["checks"]["integration_tests"]["similarity_search"]
            assert check["status"] == "pass"

    def test_similarity_search_cleans_up(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            before_count = temp_store.collection.count()
            _health()
            after_count = temp_store.collection.count()
            assert after_count == before_count

    def test_similarity_search_fail_on_exception(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()), \
             patch.object(temp_store, "query_similar", side_effect=RuntimeError("query fail")):
            result = json.loads(_health())
            check = result["checks"]["integration_tests"]["similarity_search"]
            assert check["status"] == "fail"
            assert "query fail" in check["details"]


class TestHealthIntegrationMemo:
    """Tests for the memo ingestion integration test."""

    def test_memo_ingestion_passes(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            check = result["checks"]["integration_tests"]["memo_ingestion"]
            assert check["status"] == "pass"
            assert check["nodes_created"] > 0

    def test_memo_ingestion_cleans_up(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            before_count = temp_store.collection.count()
            _health()
            after_count = temp_store.collection.count()
            assert after_count == before_count

    def test_memo_ingestion_fail_on_exception(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()), \
             patch("mind_map.mcp.server.ingest_memo", side_effect=RuntimeError("ingest fail")):
            result = json.loads(_health())
            check = result["checks"]["integration_tests"]["memo_ingestion"]
            assert check["status"] == "fail"
            assert "ingest fail" in check["details"]


class TestHealthIntegrationPersistence:
    """Tests for the data persistence integration test."""

    def test_persistence_passes(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            check = result["checks"]["integration_tests"]["data_persistence"]
            assert check["status"] == "pass"

    def test_persistence_cleans_up(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            before_nodes = temp_store.collection.count()
            cursor = temp_store.sqlite.execute("SELECT COUNT(*) FROM edges")
            before_edges = cursor.fetchone()[0]
            _health()
            after_nodes = temp_store.collection.count()
            cursor = temp_store.sqlite.execute("SELECT COUNT(*) FROM edges")
            after_edges = cursor.fetchone()[0]
            assert after_nodes == before_nodes
            assert after_edges == before_edges

    def test_persistence_fail_on_read_error(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()), \
             patch.object(temp_store, "get_node", return_value=None):
            result = json.loads(_health())
            check = result["checks"]["integration_tests"]["data_persistence"]
            assert check["status"] == "fail"
            assert "read back node" in check["details"]


class TestHealthOverallStatus:
    """Tests for overall status determination (healthy/degraded/unhealthy)."""

    def test_healthy_when_all_pass(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=True), \
             patch("mind_map.mcp.server.get_selected_model", return_value="phi3.5"), \
             patch("mind_map.mcp.server.get_available_models", return_value=["phi3.5"]), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_online()):
            result = json.loads(_health())
            assert result["status"] == "healthy"

    def test_degraded_when_ollama_down_but_dbs_ok(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            # DBs pass, integration tests pass, but Ollama + LLMs down => degraded
            assert result["status"] == "degraded"

    def test_degraded_when_llms_unavailable(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=True), \
             patch("mind_map.mcp.server.get_selected_model", return_value="phi3.5"), \
             patch("mind_map.mcp.server.get_available_models", return_value=["phi3.5"]), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            # Ollama OK, DBs OK, integration OK, but LLMs offline => degraded
            assert result["status"] == "degraded"

    def test_unhealthy_when_integration_test_fails(self, temp_store: GraphStore):
        with patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()), \
             patch.object(temp_store, "query_similar", side_effect=RuntimeError("fail")):
            result = json.loads(_health())
            assert result["status"] == "unhealthy"

    def test_unhealthy_when_chromadb_fails(self):
        stores.clear()
        with patch("mind_map.mcp.server.get_store", side_effect=RuntimeError("db down")), \
             patch("mind_map.mcp.server.check_ollama_available", return_value=False), \
             patch("mind_map.mcp.server.get_llm_status", return_value=_mock_llm_status_offline()):
            result = json.loads(_health())
            assert result["status"] == "unhealthy"


class TestHealthErrorHandling:
    """Tests for critical error handling."""

    def test_critical_failure_returns_error_json(self):
        """If something unexpected blows up, we get a JSON error."""
        with patch("mind_map.mcp.server.check_ollama_available", side_effect=RuntimeError("total crash")), \
             patch("mind_map.mcp.server.get_llm_status", side_effect=RuntimeError("also crashed")), \
             patch("mind_map.mcp.server.get_store", side_effect=RuntimeError("store crashed")):
            raw = _health()
            result = json.loads(raw)
            # Should still be valid JSON with unhealthy status
            assert result["status"] == "unhealthy"

    def test_error_json_is_valid(self):
        """Even on catastrophic failure, output is parseable JSON."""
        # Patch time.time to raise to trigger outer except
        with patch("mind_map.mcp.server.time") as mock_time:
            mock_time.time.side_effect = RuntimeError("time broken")
            # The outer try/except should catch this
            raw = _health()
            result = json.loads(raw)
            assert "status" in result
