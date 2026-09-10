"""Unit tests for mind_map.app.services.

These tests pin down the single source-of-truth contract that both the CLI
Typer commands and the MCP FastMCP tools must satisfy. Each test calls a
service function directly (no CLI invocation, no MCP server), so the contract
is enforced regardless of which surface a caller uses.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_map.app import services
from mind_map.core.schemas import Edge, NodeType
from mind_map.processor.cli_executor import LocalTarget, MiniMaxTarget
from mind_map.rag.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_store() -> GraphStore:
    """Create and initialize a temporary GraphStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        yield store


@pytest.fixture
def seeded_store() -> tuple[GraphStore, str]:
    """Create a store with: 1 anchor concept, 1 tag, 2 entities, 3 edges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(Path(tmpdir))
        store.initialize()
        store.add_node("anchor", "Anchor concept about retrieval", NodeType.CONCEPT)
        store.add_node("tag1", "#TagOne", NodeType.TAG)
        store.add_node("entity1", "Entity One", NodeType.ENTITY)
        store.add_node("entity2", "Entity Two", NodeType.ENTITY)

        store.add_edge(Edge(source="anchor", target="tag1", weight=3.0, relation_type="tagged_as"))
        store.add_edge(Edge(source="anchor", target="entity1", weight=2.0, relation_type="mentions"))
        store.add_edge(Edge(source="anchor", target="entity2", weight=1.0, relation_type="related_to"))
        yield store, tmpdir


# ---------------------------------------------------------------------------
# resolve_store
# ---------------------------------------------------------------------------


class TestResolveStore:
    """Tests for resolve_store precedence."""

    def test_data_dir_takes_precedence_over_workspace_id(self, tmp_path):
        custom_dir = tmp_path / "custom"
        store, path = services.resolve_store(data_dir=custom_dir, workspace_id="alice")
        assert path == custom_dir.resolve()

    def test_workspace_id_used_when_no_data_dir(self, monkeypatch, tmp_path):
        # Patch DEFAULT_DATA_DIR to a controlled tmp location
        fake_default = tmp_path / "fake_default"
        monkeypatch.setattr("mind_map.app.services.DEFAULT_DATA_DIR", fake_default)
        store, path = services.resolve_store(workspace_id="alice")
        assert path == (fake_default / "workspaces" / "alice").resolve()

    def test_neither_uses_default(self, monkeypatch, tmp_path):
        fake_default = tmp_path / "fake_default"
        monkeypatch.setattr("mind_map.app.services.DEFAULT_DATA_DIR", fake_default)
        store, path = services.resolve_store()
        assert path == fake_default.resolve()

    def test_data_dir_string_accepted(self, tmp_path):
        custom = tmp_path / "str_path"
        store, path = services.resolve_store(data_dir=str(custom))
        assert path == custom.resolve()


# ---------------------------------------------------------------------------
# parse_memo_target
# ---------------------------------------------------------------------------


class TestParseMemoTarget:
    """Tests for parse_memo_target."""

    def test_local_with_model_id(self):
        target = services.parse_memo_target(local="my-model", api_key=None)
        assert isinstance(target, LocalTarget)
        assert target.model == "my-model"
        assert target.base_url == "http://127.0.0.1:11435/v1"

    def test_local_empty_auto_resolves(self):
        with patch(
            "mind_map.processor.cli_executor.resolve_local_model",
            return_value="auto-model",
        ):
            target = services.parse_memo_target(local="", api_key=None)
        assert isinstance(target, LocalTarget)
        assert target.model == "auto-model"

    def test_api_key_builds_minimax(self):
        target = services.parse_memo_target(local=None, api_key="sk-test")
        assert isinstance(target, MiniMaxTarget)
        assert target.api_key == "sk-test"
        assert target.model == "MiniMax-M2.5"

    def test_local_wins_over_api_key(self):
        with patch(
            "mind_map.processor.cli_executor.resolve_local_model",
            return_value="local-model",
        ):
            target = services.parse_memo_target(local="local-model", api_key="sk-key")
        assert isinstance(target, LocalTarget)
        assert target.model == "local-model"

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            services.parse_memo_target(local=None, api_key=None)


# ---------------------------------------------------------------------------
# memo_ingest
# ---------------------------------------------------------------------------


class TestMemoIngest:
    """Tests for memo_ingest target resolution and delegation."""

    def test_no_local_no_key_raises(self, temp_store: GraphStore):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                services.memo_ingest("hello world", temp_store)

    def test_local_triggers_ingest_memo_cli(self, temp_store: GraphStore):
        with patch(
            "mind_map.processor.cli_executor.resolve_local_model",
            return_value="mlx-community/test",
        ):
            with patch(
                # services.memo_ingest lazy-imports from pipeline
                "mind_map.app.pipeline.ingest_memo_cli",
                return_value=(True, "Created 1 nodes", ["n1"]),
            ) as mock_ingest:
                success, message, node_ids = services.memo_ingest(
                    "hello world", temp_store, local=""
                )
                assert success is True
                assert node_ids == ["n1"]
                target = mock_ingest.call_args.kwargs["target"]
                assert target.model == "mlx-community/test"

    def test_minimax_target_with_env_key(self, temp_store: GraphStore):
        # MINIMAX is now the LEGACY fallback — ensure DeepSeek key (default)
        # isn't picked up from .env so we exercise the fallback path.
        env = {k: v for k, v in os.environ.items() if k != "COMMANDCODE_API_KEY"}
        env["MINIMAX_API_KEY"] = "sk-env-key"
        with patch.dict(os.environ, env, clear=True):
            with patch(
                # services.memo_ingest lazy-imports from pipeline
                "mind_map.app.pipeline.ingest_memo_cli",
                return_value=(True, "Created 2 nodes", ["n1", "n2"]),
            ) as mock_ingest:
                success, message, node_ids = services.memo_ingest(
                    "a memo", temp_store, source="test-source"
                )
                target = mock_ingest.call_args.kwargs["target"]
                assert isinstance(target, MiniMaxTarget)
                assert target.api_key == "sk-env-key"
                assert mock_ingest.call_args.kwargs["source_id"] == "test-source"

    def test_deepseek_default_target(self, temp_store: GraphStore):
        """Default path resolves the CommandCode LocalTarget when its key is set."""
        env = {"COMMANDCODE_API_KEY": "sk-env-deepseek"}
        with patch.dict(os.environ, env, clear=True), patch(
            "mind_map.app.pipeline.ingest_memo_cli",
            return_value=(True, "Created 2 nodes", ["n1", "n2"]),
        ) as mock_ingest:
            success, message, node_ids = services.memo_ingest(
                "a memo", temp_store, source="test-source"
            )
            assert success is True
            target = mock_ingest.call_args.kwargs["target"]
            assert isinstance(target, LocalTarget)
            assert target.model == "deepseek/deepseek-v4.1-flash"
            assert target.base_url == "https://api.commandcode.ai/provider/v1"
            assert target.api_key == "sk-env-deepseek"


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    """Tests for retrieve_context output format."""

    def test_header_present(self, seeded_store):
        store, _ = seeded_store
        lines = services.retrieve_context("anchor", store, n_results=1)
        assert lines[0] == "### Relevant Context from Mind Map:"

    def test_match_line_format(self, seeded_store):
        store, _ = seeded_store
        lines = services.retrieve_context("anchor", store, n_results=1)
        # Find the concept line
        concept_lines = [l for l in lines if "[concept]" in l]
        assert len(concept_lines) == 1
        assert "(Relevance:" in concept_lines[0]
        assert "Anchor concept about retrieval" in concept_lines[0]

    def test_show_context_includes_related(self, seeded_store):
        store, _ = seeded_store
        lines = services.retrieve_context("anchor", store, n_results=1, show_context=True)
        related = [l for l in lines if "└─ related" in l]
        assert len(related) == 3  # tag + 2 entities

    def test_show_context_false_omits_related(self, seeded_store):
        store, _ = seeded_store
        lines = services.retrieve_context("anchor", store, n_results=1, show_context=False)
        related = [l for l in lines if "└─ related" in l]
        assert related == []

    def test_max_context_per_node_truncates(self, seeded_store):
        store, _ = seeded_store
        lines = services.retrieve_context(
            "anchor", store, n_results=1, show_context=True, max_context_per_node=2
        )
        related = [l for l in lines if "└─ related" in l]
        assert len(related) == 2

    def test_empty_graph_returns_single_line(self, temp_store: GraphStore):
        lines = services.retrieve_context("nothing here", temp_store, n_results=5)
        assert lines == ["No relevant information found in the knowledge graph."]


# ---------------------------------------------------------------------------
# ask_question
# ---------------------------------------------------------------------------


class TestAskQuestion:
    """Tests for ask_question read-only default and back-feed opt-in."""

    def test_no_llm_returns_no_llm_status(self, temp_store: GraphStore):
        with patch("mind_map.rag.reasoning_llm.get_reasoning_llm", return_value=None):
            result = services.ask_question("what?", temp_store)
        assert result["status"] == "no_llm"
        assert "Raw context" in result["response"] or "no relevant context" in result["response"]
        assert result["qa_node_ids"] == []
        assert result["context_nodes"] == []

    def test_back_feed_false_does_not_write(self, temp_store: GraphStore):
        # Seed one node so retrieval returns something
        temp_store.add_node("n1", "Some concept", NodeType.CONCEPT)

        # ResponseGenerator is lazy-imported inside ask_question; patch at the
        # source module to intercept the import.
        fake_response = "Generated answer text"
        with patch(
            "mind_map.rag.response_generator.ResponseGenerator"
        ) as MockGen, patch(
            "mind_map.rag.reasoning_llm.get_reasoning_llm",
            return_value="fake-llm",
        ):
            MockGen.return_value.generate_sync.return_value = fake_response
            with patch(
                "mind_map.app.services.ingest_memo_internal"
            ) as mock_internal:
                before_count = temp_store.collection.count()
                result = services.ask_question(
                    "question?", temp_store, back_feed=False
                )
                after_count = temp_store.collection.count()

                assert result["status"] == "answered"
                assert result["response"] == fake_response
                assert result["qa_node_ids"] == []
                # CRITICAL: no nodes added, no internal ingestion called
                assert after_count == before_count
                mock_internal.assert_not_called()

    def test_back_feed_true_writes_to_graph(self, temp_store: GraphStore):
        temp_store.add_node("ctx1", "context for question", NodeType.CONCEPT)

        fake_response = "Generated answer"
        with patch(
            "mind_map.rag.response_generator.ResponseGenerator"
        ) as MockGen, patch(
            "mind_map.rag.reasoning_llm.get_reasoning_llm",
            return_value="fake-llm",
        ):
            MockGen.return_value.generate_sync.return_value = fake_response
            with patch(
                "mind_map.app.services.ingest_memo_internal",
                return_value=(True, "ok", ["qa_node_1"]),
            ) as mock_internal:
                before = temp_store.collection.count()
                result = services.ask_question(
                    "q?", temp_store, back_feed=True
                )
                after = temp_store.collection.count()
                mock_internal.assert_called_once()
                assert result["qa_node_ids"] == ["qa_node_1"]


# ---------------------------------------------------------------------------
# graph_stats + format_stats_text
# ---------------------------------------------------------------------------


class TestGraphStats:
    """Tests for graph_stats and format_stats_text."""

    def test_empty_graph(self, temp_store: GraphStore):
        stats = services.graph_stats(temp_store)
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_counts_by_type(self, temp_store: GraphStore):
        temp_store.add_node("c1", "concept one", NodeType.CONCEPT)
        temp_store.add_node("c2", "concept two", NodeType.CONCEPT)
        temp_store.add_node("t1", "tag one", NodeType.TAG)
        temp_store.add_node("e1", "entity one", NodeType.ENTITY)
        stats = services.graph_stats(temp_store)
        assert stats["total_nodes"] == 4
        assert stats["concept_nodes"] == 2
        assert stats["tag_nodes"] == 1
        assert stats["entity_nodes"] == 1


class TestFormatStatsText:
    """Tests for format_stats_text (locked by MCP contract)."""

    def test_header_uses_workspace(self):
        stats = {
            "total_nodes": 5, "total_edges": 3,
            "concept_nodes": 2, "entity_nodes": 1, "tag_nodes": 2,
            "avg_connections": 1.2,
        }
        out = services.format_stats_text(stats, workspace_id="alice")
        assert "### Knowledge Graph Statistics (alice):" in out

    def test_default_workspace(self):
        stats = {
            "total_nodes": 0, "total_edges": 0,
            "concept_nodes": 0, "entity_nodes": 0, "tag_nodes": 0,
            "avg_connections": 0,
        }
        out = services.format_stats_text(stats)
        assert "### Knowledge Graph Statistics (default):" in out
        assert "- Total Nodes: 0" in out
        assert "- Avg Connections: 0" in out

    def test_all_six_metrics_present(self):
        stats = {
            "total_nodes": 1, "total_edges": 2,
            "concept_nodes": 1, "entity_nodes": 0, "tag_nodes": 0,
            "avg_connections": 4.0,
        }
        out = services.format_stats_text(stats, workspace_id="ws")
        assert "- Total Nodes: 1" in out
        assert "- Total Edges: 2" in out
        assert "- Concepts: 1" in out
        assert "- Entities: 0" in out
        assert "- Tags: 0" in out
        assert "- Avg Connections: 4.0" in out


# ---------------------------------------------------------------------------
# prune_graph
# ---------------------------------------------------------------------------


class TestPruneGraph:
    """Tests for prune_graph contract (matches MCP JSON schema)."""

    def test_empty_graph_returns_safe_defaults(self, temp_store: GraphStore):
        result = services.prune_graph(temp_store, percent=0.1, workspace_id="default")
        assert result["deleted_nodes"] == []
        assert result["deleted_tags"] == []
        assert result["deleted_edges_count"] == 0
        assert "empty" in result["summary"].lower()
        assert "default" in result["summary"]

    def test_single_node_pruned(self, temp_store: GraphStore):
        temp_store.add_node("c1", "only concept", NodeType.CONCEPT)
        result = services.prune_graph(temp_store, percent=0.1, workspace_id="default")
        assert len(result["deleted_nodes"]) == 1
        assert result["deleted_nodes"][0]["id"] == "c1"
        assert result["deleted_nodes"][0]["type"] == "concept"
        # Node actually gone from store
        assert temp_store.get_node("c1") is None

    def test_prune_at_least_one_with_few_nodes(self, temp_store: GraphStore):
        for i in range(5):
            temp_store.add_node(f"c{i}", f"Concept {i}", NodeType.CONCEPT)
        result = services.prune_graph(temp_store, percent=0.1)
        assert len(result["deleted_nodes"]) >= 1

    def test_prune_exact_count(self, temp_store: GraphStore):
        for i in range(20):
            temp_store.add_node(f"c{i}", f"Concept {i}", NodeType.CONCEPT)
        result = services.prune_graph(temp_store, percent=0.1)
        assert len(result["deleted_nodes"]) == 2  # floor(20 * 0.1)

    def test_lowest_importance_pruned(self, temp_store: GraphStore):
        for i in range(10):
            temp_store.add_node(f"c{i}", f"Concept {i}", NodeType.CONCEPT)
        # Boost c1-c9 with edges; c0 stays isolated
        for i in range(1, 10):
            temp_store.add_edge(
                Edge(source=f"c{i}", target=f"c{(i % 9) + 1}", relation_type="related_to")
            )
        result = services.prune_graph(temp_store, percent=0.1)
        pruned_ids = {n["id"] for n in result["deleted_nodes"]}
        assert "c0" in pruned_ids

    def test_entities_eligible(self, temp_store: GraphStore):
        temp_store.add_node("e1", "isolated entity", NodeType.ENTITY)
        result = services.prune_graph(temp_store, percent=0.1)
        pruned_ids = {n["id"] for n in result["deleted_nodes"]}
        assert "e1" in pruned_ids

    def test_tags_not_direct_candidates(self, temp_store: GraphStore):
        temp_store.add_node("c1", "concept", NodeType.CONCEPT)
        temp_store.add_node("t1", "tag", NodeType.TAG)
        temp_store.add_edge(Edge(source="c1", target="t1", relation_type="tagged_as"))
        result = services.prune_graph(temp_store, percent=0.1)
        pruned_node_ids = {n["id"] for n in result["deleted_nodes"]}
        assert "t1" not in pruned_node_ids

    def test_orphan_tag_removed(self, temp_store: GraphStore):
        temp_store.add_node("c1", "low importance", NodeType.CONCEPT)
        temp_store.add_node("t1", "orphan tag", NodeType.TAG)
        temp_store.add_edge(Edge(source="c1", target="t1", relation_type="tagged_as"))
        result = services.prune_graph(temp_store, percent=0.1)
        deleted_tag_ids = {t["id"] for t in result["deleted_tags"]}
        assert "t1" in deleted_tag_ids

    def test_shared_tag_preserved(self, temp_store: GraphStore):
        # 10 concepts so only 1 pruned
        for i in range(10):
            temp_store.add_node(f"c{i}", f"Concept {i}", NodeType.CONCEPT)
        temp_store.add_node("t_shared", "shared tag", NodeType.TAG)
        # Boost c1-c9 so c0 (isolated) gets pruned
        for i in range(1, 10):
            temp_store.add_edge(
                Edge(source=f"c{i}", target=f"c{(i % 9) + 1}", relation_type="related_to")
            )
        temp_store.add_edge(Edge(source="c0", target="t_shared", relation_type="tagged_as"))
        temp_store.add_edge(Edge(source="c1", target="t_shared", relation_type="tagged_as"))

        result = services.prune_graph(temp_store, percent=0.1)
        deleted_tag_ids = {t["id"] for t in result["deleted_tags"]}
        assert "t_shared" not in deleted_tag_ids
        assert temp_store.get_node("t_shared") is not None

    def test_deleted_node_shape(self, temp_store: GraphStore):
        temp_store.add_node("c1", "test concept", NodeType.CONCEPT)
        result = services.prune_graph(temp_store, percent=0.1)
        node = result["deleted_nodes"][0]
        assert set(node.keys()) == {"id", "document", "type"}

    def test_summary_mentions_workspace(self, temp_store: GraphStore):
        temp_store.add_node("c1", "concept", NodeType.CONCEPT)
        result = services.prune_graph(temp_store, percent=0.1, workspace_id="myspace")
        assert "myspace" in result["summary"]
        assert "1" in result["summary"]


# ---------------------------------------------------------------------------
# report_graph
# ---------------------------------------------------------------------------


class TestReportGraph:
    """Tests for report_graph JSON contract (matches MCP report)."""

    def test_empty_graph_returns_zeros(self, temp_store: GraphStore):
        result = services.report_graph(temp_store)
        summary = result["summary"]
        assert summary["total_nodes"] == 0
        assert summary["total_edges"] == 0
        assert summary["concepts"] == 0
        assert summary["entities"] == 0
        assert summary["tags"] == 0
        assert result["top_nodes"] == []

    def test_summary_keys(self, temp_store: GraphStore):
        result = services.report_graph(temp_store)
        assert set(result.keys()) == {"summary", "top_nodes"}
        assert set(result["summary"].keys()) == {
            "workspace", "total_nodes", "total_edges",
            "concepts", "entities", "tags", "avg_connections",
        }

    def test_top_node_shape(self, temp_store: GraphStore):
        temp_store.add_node("c1", "concept", NodeType.CONCEPT)
        result = services.report_graph(temp_store)
        node = result["top_nodes"][0]
        assert set(node.keys()) == {
            "id", "document", "type", "importance_score",
            "connection_count", "edges", "tags",
        }
        assert node["type"] == "concept"

    def test_top_nodes_capped_at_five(self, temp_store: GraphStore):
        for i in range(8):
            temp_store.add_node(f"c{i}", f"Concept {i}", NodeType.CONCEPT)
        result = services.report_graph(temp_store)
        assert len(result["top_nodes"]) == 5

    def test_summary_counts_correctly(self, temp_store: GraphStore):
        temp_store.add_node("c1", "c1", NodeType.CONCEPT)
        temp_store.add_node("c2", "c2", NodeType.CONCEPT)
        temp_store.add_node("t1", "t1", NodeType.TAG)
        temp_store.add_node("e1", "e1", NodeType.ENTITY)
        result = services.report_graph(temp_store)
        assert result["summary"]["total_nodes"] == 4
        assert result["summary"]["concepts"] == 2
        assert result["summary"]["tags"] == 1
        assert result["summary"]["entities"] == 1


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health_check JSON structure (matches MCP health contract)."""

    def test_top_level_keys(self, temp_store: GraphStore):
        with patch(
            "mind_map.processor.processing_llm.check_ollama_available",
            return_value=False,
        ), patch(
            "mind_map.rag.llm_status.get_llm_status",
            return_value={"processing_llm": {"status": "offline", "provider": "x", "model": "y"}},
        ):
            result = services.health_check(temp_store, workspace_id="default")
        assert set(result.keys()) == {"status", "checks", "timestamp", "workspace"}
        assert result["workspace"] == "default"
        assert isinstance(result["timestamp"], float)
        assert result["status"] in ("healthy", "degraded", "unhealthy")

    def test_checks_sections(self, temp_store: GraphStore):
        with patch(
            "mind_map.processor.processing_llm.check_ollama_available",
            return_value=False,
        ), patch(
            "mind_map.rag.llm_status.get_llm_status",
            return_value={"processing_llm": {"status": "offline", "provider": "x", "model": "y"}},
        ):
            result = services.health_check(temp_store, workspace_id="default")
        checks = result["checks"]
        assert "ollama_connection" in checks
        assert "chromadb_connection" in checks
        assert "sqlite_connection" in checks
        assert "processing_llm" in checks
        assert "integration_tests" in checks
        # integration subtests
        integ = checks["integration_tests"]
        assert "similarity_search" in integ
        assert "memo_ingestion" in integ
        assert "data_persistence" in integ

    def test_integration_tests_clean_up(self, temp_store: GraphStore):
        before = temp_store.collection.count()
        with patch(
            "mind_map.processor.processing_llm.check_ollama_available",
            return_value=False,
        ), patch(
            "mind_map.rag.llm_status.get_llm_status",
            return_value={"processing_llm": {"status": "offline", "provider": "x", "model": "y"}},
        ):
            services.health_check(temp_store, workspace_id="default")
        after = temp_store.collection.count()
        assert after == before

    def test_workspace_in_result(self, temp_store: GraphStore):
        with patch(
            "mind_map.processor.processing_llm.check_ollama_available",
            return_value=False,
        ), patch(
            "mind_map.rag.llm_status.get_llm_status",
            return_value={"processing_llm": {"status": "offline", "provider": "x", "model": "y"}},
        ):
            result = services.health_check(temp_store, workspace_id="alice")
        assert result["workspace"] == "alice"

    def test_uses_ingest_memo_alias_for_memo_integration(self, temp_store: GraphStore):
        """The memo_ingestion integration test calls services.ingest_memo.
        If a test patches `mind_map.mcp.server.ingest_memo`, the patch must
        take effect. We verify the alias path is honored."""
        with patch(
            "mind_map.processor.processing_llm.check_ollama_available",
            return_value=False,
        ), patch(
            "mind_map.rag.llm_status.get_llm_status",
            return_value={"processing_llm": {"status": "offline", "provider": "x", "model": "y"}},
        ), patch(
            "mind_map.app.services.ingest_memo",
            return_value=(True, "ok", ["n1"]),
        ) as mock_alias:
            result = services.health_check(temp_store, workspace_id="default")
        mock_alias.assert_called()
        assert result["checks"]["integration_tests"]["memo_ingestion"]["status"] == "pass"


# ---------------------------------------------------------------------------
# ingest_memo back-compat alias
# ---------------------------------------------------------------------------


class TestIngestMemoAlias:
    """The ingest_memo alias must point to ingest_memo_internal for
    back-compat with existing MCP test patches."""

    def test_alias_points_to_internal(self):
        from mind_map.app.pipeline import ingest_memo_internal

        assert services.ingest_memo is ingest_memo_internal
