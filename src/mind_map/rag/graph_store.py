"""Graph storage wrapper combining ChromaDB (nodes) and SQLite (edges)."""

import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import chromadb

from mind_map.core.schemas import DeleteNodeResult, Edge, GraphNode, NodeMetadata, NodeType


class GraphStore:
    """Hybrid storage layer for the knowledge graph.

    Uses ChromaDB for vector storage of nodes and SQLite for edge registry.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.chroma_path = data_dir / "chroma"
        self.sqlite_path = data_dir / "edges.db"
        self._chroma_client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._sqlite_conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Initialize both ChromaDB and SQLite storage."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._init_chroma()
        self._init_sqlite()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB persistent client and collection."""
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = self._chroma_client.get_or_create_collection(
            name="mind_map_nodes",
            metadata={"hnsw:space": "cosine"},
        )

    def _init_sqlite(self) -> None:
        """Initialize SQLite database for edge registry."""
        self._sqlite_conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                relation_type TEXT DEFAULT 'related_context',
                created_at REAL NOT NULL,
                UNIQUE(source, target, relation_type)
            )
        """)
        self._sqlite_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)"
        )
        self._sqlite_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)"
        )
        self._sqlite_conn.commit()

    def _serialize_metadata(self, metadata: NodeMetadata) -> dict[str, str | int | float | bool]:
        """Serialize NodeMetadata to ChromaDB-compatible dict."""
        data = metadata.model_dump()
        result: dict[str, str | int | float | bool] = {}
        for key, value in data.items():
            if value is None:
                continue
            if hasattr(value, "value"):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @property
    def collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection, initializing if needed."""
        if self._collection is None:
            self.initialize()
        assert self._collection is not None
        return self._collection

    @property
    def sqlite(self) -> sqlite3.Connection:
        """Get the SQLite connection, initializing if needed."""
        if self._sqlite_conn is None:
            self.initialize()
        assert self._sqlite_conn is not None
        return self._sqlite_conn

    def add_node(
        self,
        node_id: str,
        document: str,
        node_type: NodeType,
        embedding: list[float] | None = None,
        source_id: str | None = None,
    ) -> GraphNode:
        """Add a node to the knowledge graph."""
        now = time.time()
        metadata = NodeMetadata(
            type=node_type,
            created_at=now,
            last_interaction=now,
            connection_count=0,
            importance_score=0.0,
            original_source_id=source_id,
        )

        self.collection.add(
            ids=[node_id],
            documents=[document],
            metadatas=[self._serialize_metadata(metadata)],
            embeddings=[embedding] if embedding else None,
        )

        return GraphNode(id=node_id, document=document, metadata=metadata, embedding=embedding)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph and update connection counts."""
        now = time.time()
        self.sqlite.execute(
            """
            INSERT OR REPLACE INTO edges (source, target, weight, relation_type, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (edge.source, edge.target, edge.weight, edge.relation_type, now),
        )
        self.sqlite.commit()
        self._update_connection_count(edge.source)
        self._update_connection_count(edge.target)

    def _update_connection_count(self, node_id: str) -> None:
        """Update the connection count for a node."""
        cursor = self.sqlite.execute(
            "SELECT COUNT(*) FROM edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        )
        count = cursor.fetchone()[0]

        result = self.collection.get(ids=[node_id], include=["metadatas"])
        if result["metadatas"]:
            metadata = result["metadatas"][0]
            metadata["connection_count"] = count
            self.collection.update(ids=[node_id], metadatas=[metadata])

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a node by ID."""
        try:
            result = self.collection.get(
                ids=[node_id], include=["documents", "metadatas", "embeddings"]
            )
        except Exception:
            return None
        if not result["ids"]:
            return None

        metadata_dict = result["metadatas"][0] if result["metadatas"] else None
        metadata = (
            NodeMetadata(**metadata_dict)
            if metadata_dict
            else NodeMetadata(type=NodeType.CONCEPT, created_at=0, last_interaction=0)
        )
        embeddings = result.get("embeddings")
        embedding = None
        if embeddings is not None and len(embeddings) > 0:
            embedding = list(embeddings[0]) if embeddings[0] is not None else None

        return GraphNode(
            id=result["ids"][0],
            document=(result["documents"][0] if result.get("documents") and result["documents"][0] is not None else ""),
            metadata=metadata,
            embedding=embedding,
        )

    def get_edges(self, node_id: str) -> list[Edge]:
        """Get all edges connected to a node."""
        query = """
            SELECT source, target, weight, relation_type
            FROM edges WHERE source = ? OR target = ?
        """
        cursor = self.sqlite.execute(query, (node_id, node_id))
        return [
            Edge(source=row[0], target=row[1], weight=row[2], relation_type=row[3])
            for row in cursor.fetchall()
        ]

    def get_first_hop_neighbors(self, node_ids: list[str]) -> list[GraphNode]:
        """Return first-hop entity/tag neighbors for the given nodes."""
        if not node_ids:
            return []

        node_ids_set = set(node_ids)
        neighbor_ids: set[str] = set()
        for node_id in node_ids:
            for edge in self.get_edges(node_id):
                neighbor_id = edge.target if edge.source == node_id else edge.source
                if neighbor_id in node_ids_set:
                    continue
                neighbor = self.get_node(neighbor_id)
                if neighbor and neighbor.metadata.type in (NodeType.ENTITY, NodeType.TAG):
                    neighbor_ids.add(neighbor_id)

        neighbors: list[GraphNode] = []
        for neighbor_id in sorted(neighbor_ids):
            node = self.get_node(neighbor_id)
            if node is not None:
                neighbors.append(node)
        return neighbors

    def get_connected_context(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[GraphNode, Edge]]]:
        """Get direct neighbors for each node in node_ids."""
        if not node_ids:
            return {}

        context: dict[str, list[tuple[GraphNode, Edge]]] = {nid: [] for nid in node_ids}
        node_ids_set = set(node_ids)

        for node_id in node_ids:
            edges = self.get_edges(node_id)
            for edge in edges:
                neighbor_id = edge.target if edge.source == node_id else edge.source
                if neighbor_id in node_ids_set:
                    continue
                neighbor_node = self.get_node(neighbor_id)
                if neighbor_node:
                    context[node_id].append((neighbor_node, edge))

            context[node_id].sort(
                key=lambda item: (
                    -item[1].weight,
                    -(item[0].metadata.importance_score or 0.0),
                    item[1].relation_type,
                    item[0].document,
                    item[0].id,
                )
            )

        return context

    def get_subgraph(self, node_id: str, depth: int = 2) -> tuple[list[GraphNode], list[Edge]]:
        """Traverse the graph from a node to a specified depth."""
        visited_nodes: set[str] = set()
        edges: list[Edge] = []
        frontier = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)
                node_edges = self.get_edges(nid)
                for edge in node_edges:
                    if edge not in edges:
                        edges.append(edge)
                    neighbor = edge.target if edge.source == nid else edge.source
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)
            frontier = next_frontier

        nodes = [self.get_node(nid) for nid in visited_nodes]
        return [n for n in nodes if n is not None], edges

    def query_similar(
        self, query: str | list[float], n_results: int = 10, max_distance: float = 0.5
    ) -> list[GraphNode]:
        """Query nodes by text or embedding similarity."""
        if isinstance(query, str):
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_embeddings=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        if not results["ids"] or not results["ids"][0]:
            return []

        nodes: list[GraphNode] = []
        for i, node_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            if distance > max_distance:
                continue

            metadata_dict = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else None
            metadata = (
                NodeMetadata(**metadata_dict)
                if metadata_dict
                else NodeMetadata(type=NodeType.CONCEPT, created_at=0, last_interaction=0)
            )
            metadata.importance_score = max(0.0, 1 - distance)

            node = GraphNode(
                id=node_id,
                document=(results["documents"][0][i] if results.get("documents") and results["documents"][0][i] is not None else ""),
                metadata=metadata,
            )
            nodes.append(node)

        return nodes

    def get_relation_factors(
        self, anchor_id: str, candidate_ids: list[str]
    ) -> dict[str, float]:
        """Calculate relation factor for each candidate relative to an anchor node."""
        if not candidate_ids:
            return {}

        cursor = self.sqlite.execute(
            "SELECT COUNT(*) FROM edges WHERE source = ? OR target = ?",
            (anchor_id, anchor_id),
        )
        total_edges = cursor.fetchone()[0]
        if total_edges == 0:
            return {cid: 0.0 for cid in candidate_ids}

        factors: dict[str, float] = {}
        for cid in candidate_ids:
            if cid == anchor_id:
                factors[cid] = 1.0
                continue
            cursor = self.sqlite.execute(
                """SELECT COUNT(*) FROM edges
                   WHERE (source = ? AND target = ?)
                      OR (source = ? AND target = ?)""",
                (anchor_id, cid, cid, anchor_id),
            )
            shared = cursor.fetchone()[0]
            factors[cid] = shared / total_edges

        return factors

    def enrich_context_nodes(self, nodes: list[GraphNode]) -> list[GraphNode]:
        """Set relation_factor on each node and re-sort by combined score."""
        if len(nodes) <= 1:
            for n in nodes:
                n.relation_factor = 1.0
            return nodes

        anchor = nodes[0]
        candidate_ids = [n.id for n in nodes]
        factors = self.get_relation_factors(anchor.id, candidate_ids)

        for node in nodes:
            node.relation_factor = factors.get(node.id, 0.0)

        nodes.sort(
            key=lambda n: n.metadata.importance_score * (1 + (n.relation_factor or 0.0)),
            reverse=True,
        )

        return nodes

    def calculate_importance(
        self, node_id: str, lambda_decay: float = 0.05, time_unit_days: float = 1.0
    ) -> float:
        """Calculate importance score S = (C_node / C_max) * e^(-lambda * delta_t)."""
        node = self.get_node(node_id)
        if not node:
            return 0.0

        cursor = self.sqlite.execute("""
            SELECT MAX(cnt) FROM (
                SELECT COUNT(*) as cnt FROM (
                    SELECT source AS node_id FROM edges
                    UNION ALL
                    SELECT target AS node_id FROM edges
                )
                GROUP BY node_id
            )
        """)
        c_max = cursor.fetchone()[0] or 1
        c_node = node.metadata.connection_count

        age_seconds = time.time() - node.metadata.last_interaction
        age_days = age_seconds / (86400 * time_unit_days)

        return (c_node / c_max) * math.exp(-lambda_decay * age_days)

    def update_importance_scores(self) -> None:
        """Recalculate and persist importance scores for all nodes."""
        result = self.collection.get(include=["metadatas"])
        if not result["ids"]:
            return

        for node_id, metadata in zip(result["ids"], result["metadatas"]):
            if metadata is None:
                continue
            metadata["importance_score"] = self.calculate_importance(node_id)
            self.collection.update(ids=[node_id], metadatas=[metadata])

    def delete_node(self, node_id: str) -> DeleteNodeResult:
        """Delete a node and, for concept deletes, its first-layer tag neighbors."""
        node = self.get_node(node_id)
        if node is None:
            return DeleteNodeResult(deleted_node_id=node_id, deleted_tag_ids=[], deleted_edges_count=0)

        deleted_tag_ids: list[str] = []
        nodes_to_delete: list[str] = [node_id]

        if node.metadata.type == NodeType.CONCEPT:
            for neighbor_node, _edge in self.get_connected_context([node_id]).get(node_id, []):
                if neighbor_node.metadata.type == NodeType.TAG:
                    deleted_tag_ids.append(neighbor_node.id)
                    nodes_to_delete.append(neighbor_node.id)

        deleted_edges_count = 0
        for nid in nodes_to_delete:
            deleted_edges_count += self._delete_edges_for_node(nid)

        self.collection.delete(ids=nodes_to_delete)

        return DeleteNodeResult(
            deleted_node_id=node_id,
            deleted_tag_ids=deleted_tag_ids,
            deleted_edges_count=deleted_edges_count,
        )

    def _delete_edges_for_node(self, node_id: str) -> int:
        """Delete all edges connected to a node and update surviving neighbors' counts."""
        edges = self.get_edges(node_id)
        neighbors_to_update: set[str] = set()
        for edge in edges:
            neighbor = edge.target if edge.source == node_id else edge.source
            neighbors_to_update.add(neighbor)

        cursor = self.sqlite.execute(
            "DELETE FROM edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        )
        self.sqlite.commit()
        deleted_count = cursor.rowcount if cursor.rowcount != -1 else len(edges)

        for neighbor_id in neighbors_to_update:
            if self.get_node(neighbor_id) is not None:
                self._update_connection_count(neighbor_id)

        return deleted_count

    def delete_edges_for_node(self, node_id: str) -> int:
        """Public wrapper to delete all edges connected to a node."""
        return self._delete_edges_for_node(node_id)

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics about the graph."""
        result = self.collection.get(include=["metadatas"])
        total_nodes = len(result["ids"])

        concept_nodes = 0
        tag_nodes = 0
        entity_nodes = 0
        for metadata in result["metadatas"]:
            if not metadata:
                continue
            node_type = metadata.get("type")
            if node_type == NodeType.CONCEPT.value:
                concept_nodes += 1
            elif node_type == NodeType.TAG.value:
                tag_nodes += 1
            elif node_type == NodeType.ENTITY.value:
                entity_nodes += 1

        cursor = self.sqlite.execute("SELECT COUNT(*) FROM edges")
        total_edges = cursor.fetchone()[0]
        avg_connections = (2 * total_edges / total_nodes) if total_nodes > 0 else 0.0

        return {
            "total_nodes": total_nodes,
            "concept_nodes": concept_nodes,
            "tag_nodes": tag_nodes,
            "entity_nodes": entity_nodes,
            "total_edges": total_edges,
            "avg_connections": avg_connections,
        }
