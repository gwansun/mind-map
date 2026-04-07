"""CLI tests for the retrieve command."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from mind_map.app.cli.main import app
from mind_map.core.schemas import Edge, NodeType
from mind_map.rag.graph_store import GraphStore


runner = CliRunner()


def build_store(tmp_path: Path) -> GraphStore:
    """Create and initialize a temporary graph store."""
    store = GraphStore(tmp_path)
    store.initialize()
    return store


def seed_retrieve_fixture(store: GraphStore) -> None:
    """Seed a tiny graph suitable for retrieve command tests."""
    store.add_node(
        node_id="node_anchor",
        document="Anchor concept about retrieval",
        node_type=NodeType.CONCEPT,
    )
    store.add_node(
        node_id="node_tag_1",
        document="#TagOne",
        node_type=NodeType.TAG,
    )
    store.add_node(
        node_id="node_entity_1",
        document="Entity One",
        node_type=NodeType.ENTITY,
    )
    store.add_node(
        node_id="node_entity_2",
        document="Entity Two",
        node_type=NodeType.ENTITY,
    )

    store.add_edge(Edge(source="node_anchor", target="node_tag_1", weight=3.0, relation_type="tagged_as"))
    store.add_edge(Edge(source="node_anchor", target="node_entity_1", weight=2.0, relation_type="mentions"))
    store.add_edge(Edge(source="node_anchor", target="node_entity_2", weight=1.0, relation_type="related_to"))


class TestRetrieveCli:
    """Tests for the retrieve CLI command."""

    def test_retrieve_shows_context_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_store(Path(tmpdir))
            seed_retrieve_fixture(store)

            result = runner.invoke(
                app,
                ["retrieve", "Anchor concept about retrieval", "--data-dir", tmpdir, "--n-results", "1"],
            )

            assert result.exit_code == 0
            assert "### Relevant Context from Mind Map:" in result.stdout
            assert "- [concept]" in result.stdout
            assert "related [tag] via tagged_as: #TagOne" in result.stdout
            assert "related [entity] via mentions: Entity One" in result.stdout

    def test_retrieve_no_context_suppresses_related_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_store(Path(tmpdir))
            seed_retrieve_fixture(store)

            result = runner.invoke(
                app,
                [
                    "retrieve",
                    "Anchor concept about retrieval",
                    "--data-dir",
                    tmpdir,
                    "--n-results",
                    "1",
                    "--no-context",
                ],
            )

            assert result.exit_code == 0
            assert "### Relevant Context from Mind Map:" in result.stdout
            assert "- [concept]" in result.stdout
            assert "related [" not in result.stdout

    def test_retrieve_respects_max_context_per_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_store(Path(tmpdir))
            seed_retrieve_fixture(store)

            result = runner.invoke(
                app,
                [
                    "retrieve",
                    "Anchor concept about retrieval",
                    "--data-dir",
                    tmpdir,
                    "--n-results",
                    "1",
                    "--max-context-per-node",
                    "2",
                ],
            )

            assert result.exit_code == 0
            related_lines = [line for line in result.stdout.splitlines() if "related [" in line]
            assert len(related_lines) == 2
            # Sorted by weight desc, so the 3.0 and 2.0 edges should appear, but not the 1.0 edge.
            assert any("related [tag] via tagged_as: #TagOne" in line for line in related_lines)
            assert any("related [entity] via mentions: Entity One" in line for line in related_lines)
            assert all("Entity Two" not in line for line in related_lines)
