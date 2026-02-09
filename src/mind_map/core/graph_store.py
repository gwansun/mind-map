"""Graph storage wrapper combining ChromaDB (nodes) and SQLite (edges)."""

import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import chromadb

from mind_map.models.schemas import Edge, GraphNode, NodeMetadata, NodeType


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
        self._sqlite_conn = sqlite3.connect(str(self.sqlite_path))
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
                continue  # ChromaDB doesn't accept None
            if hasattr(value, "value"):  # Handle enums
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
        result = self.collection.get(
            ids=[node_id], include=["documents", "metadatas", "embeddings"]
        )
        if not result["ids"]:
            return None

        metadata = (
            NodeMetadata(**result["metadatas"][0])
            if result["metadatas"]
            else NodeMetadata(type=NodeType.CONCEPT, created_at=0, last_interaction=0)
        )
        # Handle embeddings carefully - ChromaDB returns numpy arrays
        embeddings = result.get("embeddings")
        embedding = None
        if embeddings is not None and len(embeddings) > 0:
            embedding = list(embeddings[0]) if embeddings[0] is not None else None

        return GraphNode(
            id=result["ids"][0],
            document=result["documents"][0] if result["documents"] else "",
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
        self, query: str | list[float], n_results: int = 10
    ) -> list[GraphNode]:
        """Query nodes by text or embedding similarity.

        Args:
            query: Either a text string or embedding vector
            n_results: Maximum number of results to return

        Returns:
            List of GraphNode objects sorted by relevance
        """
        if isinstance(query, str):
            # Text-based query using ChromaDB's built-in embedding
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        else:
            # Embedding-based query
            results = self.collection.query(
                query_embeddings=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        if not results["ids"] or not results["ids"][0]:
            return []

        nodes: list[GraphNode] = []
        for i, node_id in enumerate(results["ids"][0]):
            metadata = (
                NodeMetadata(**results["metadatas"][0][i])
                if results["metadatas"]
                else NodeMetadata(type=NodeType.CONCEPT, created_at=0, last_interaction=0)
            )
            # Calculate importance score from distance
            distance = results["distances"][0][i] if results["distances"] else 0.0
            metadata.importance_score = 1 - distance  # Convert distance to similarity

            node = GraphNode(
                id=node_id,
                document=results["documents"][0][i] if results["documents"] else "",
                metadata=metadata,
            )
            nodes.append(node)

        return nodes

    def get_relation_factors(
        self, anchor_id: str, candidate_ids: list[str]
    ) -> dict[str, float]:
        """Calculate relation factor for each candidate relative to an anchor node.

        relation_factor(anchor → candidate) = edges_between(anchor, candidate) / total_edges(anchor)

        Args:
            anchor_id: The primary/anchor node ID (typically the most relevant query result)
            candidate_ids: List of candidate node IDs to score

        Returns:
            Dict mapping candidate_id → relation_factor (0.0 to 1.0)
        """
        if not candidate_ids:
            return {}

        # Get total edge count for anchor
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
        """Set relation_factor on each node and re-sort by combined score.

        Uses the first node as the anchor (highest similarity). Combined score:
            importance * (1 + relation_factor)

        Nodes with more edges to the anchor get boosted (up to 2x).
        Nodes with zero relation factor keep their original importance.

        Args:
            nodes: List of GraphNode objects (first node is treated as anchor)

        Returns:
            The same list with relation_factor set and re-sorted by combined score
        """
        if len(nodes) <= 1:
            for n in nodes:
                n.relation_factor = 1.0
            return nodes

        anchor = nodes[0]
        candidate_ids = [n.id for n in nodes]
        factors = self.get_relation_factors(anchor.id, candidate_ids)

        for node in nodes:
            node.relation_factor = factors.get(node.id, 0.0)

        # Re-sort by combined score: importance * (1 + relation_factor)
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

        # Get max connections across all nodes
        cursor = self.sqlite.execute("""
            SELECT MAX(cnt) FROM (
                SELECT COUNT(*) as cnt FROM edges GROUP BY source
                UNION ALL
                SELECT COUNT(*) as cnt FROM edges GROUP BY target
            )
        """)
        c_max = cursor.fetchone()[0] or 1

        c_node = node.metadata.connection_count
        delta_t = (time.time() - node.metadata.last_interaction) / (86400 * time_unit_days)

        score = (c_node / c_max) * math.exp(-lambda_decay * delta_t)
        return score

    def update_interaction(self, node_id: str) -> None:
        """Update the last_interaction timestamp for a node."""
        result = self.collection.get(ids=[node_id], include=["metadatas"])
        if result["metadatas"]:
            metadata = result["metadatas"][0]
            metadata["last_interaction"] = time.time()
            self.collection.update(ids=[node_id], metadatas=[metadata])

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        node_count = self.collection.count()
        cursor = self.sqlite.execute("SELECT COUNT(*) FROM edges")
        edge_count = cursor.fetchone()[0]

        result = self.collection.get(include=["metadatas"])
        metadatas = result["metadatas"] or []
        tag_count = sum(1 for m in metadatas if m.get("type") == NodeType.TAG.value)
        entity_count = sum(1 for m in metadatas if m.get("type") == NodeType.ENTITY.value)
        concept_count = node_count - tag_count - entity_count

        avg_connections = 0.0
        if node_count > 0:
            avg_connections = (edge_count * 2) / node_count

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "tag_nodes": tag_count,
            "entity_nodes": entity_count,
            "concept_nodes": concept_count,
            "avg_connections": round(avg_connections, 2),
        }
