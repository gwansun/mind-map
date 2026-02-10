import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add src to path to ensure imports work correctly
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from fastmcp import FastMCP
from mind_map.app.pipeline import ingest_memo
from mind_map.core.schemas import NodeMetadata, NodeType
from mind_map.rag.graph_store import GraphStore
from mind_map.processor.processing_llm import get_processing_llm

# Initialize FastMCP
mcp = FastMCP("MindMap")

# Setup project paths
# Default to project root 'data' folder, but allow override via environment variable
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_DIR = Path(os.getenv("MIND_MAP_DATA_DIR", str(PROJECT_ROOT / "data")))

# Dictionary to manage multiple stores
# workspace_id -> GraphStore instance
stores: dict[str, GraphStore] = {}

def get_store(workspace_id: Optional[str] = None) -> GraphStore:
    """Get or initialize a GraphStore for a specific workspace."""
    # Use default if no workspace_id provided
    ws_id = workspace_id or "default"
    
    if ws_id not in stores:
        # If workspace_id is provided, create a subfolder under DEFAULT_DATA_DIR
        if ws_id == "default":
            path = DEFAULT_DATA_DIR
        else:
            path = DEFAULT_DATA_DIR / "workspaces" / ws_id
            
        store = GraphStore(path)
        store.initialize()
        stores[ws_id] = store
        
    return stores[ws_id]

@mcp.tool()
def mind_map_retrieve(query: str, n_results: int = 5, workspace_id: Optional[str] = None) -> str:
    """Retrieve relevant context from the knowledge graph based on a query.
    
    Args:
        query: The search query or question.
        n_results: Number of relevant snippets to return (default 5).
        workspace_id: Unique identifier for the person or workspace (e.g., friend's name).
    """
    try:
        current_store = get_store(workspace_id)
        # 1. Similarity search
        nodes = current_store.query_similar(query, n_results=n_results)
        if not nodes:
            return f"No relevant information found in the knowledge graph ({workspace_id or 'default'})."
        
        # 2. Enrich with relation factors and re-sort
        nodes = current_store.enrich_context_nodes(nodes)
        
        # 3. Format results
        output = [f"### Relevant Context from Mind Map ({workspace_id or 'default'}):"]
        for node in nodes:
            score = node.metadata.importance_score * (1 + (node.relation_factor or 0))
            output.append(f"- [{node.metadata.type.value}] (Relevance: {score:.2f}): {node.document}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

@mcp.tool()
def mind_map_memo(text: str, workspace_id: Optional[str] = None) -> str:
    """Ingest new information or a Q&A pair into the knowledge graph.
    
    Args:
        text: The text content to store.
        workspace_id: Unique identifier for the person or workspace (e.g., friend's name).
    """
    try:
        current_store = get_store(workspace_id)
        llm = get_processing_llm()
        success, message, node_ids = ingest_memo(text=text, store=current_store, llm=llm)
        
        if success:
            return f"Successfully stored knowledge in '{workspace_id or 'default'}'. {message} (Total nodes created: {len(node_ids)})"
        else:
            return f"Information was not stored: {message}"
    except Exception as e:
        return f"Error storing knowledge: {str(e)}"

@mcp.tool()
def mind_map_stats(workspace_id: Optional[str] = None) -> str:
    """Get statistics about the current state of the knowledge graph for a specific workspace."""
    try:
        current_store = get_store(workspace_id)
        stats = current_store.get_stats()
        return (
            f"### Knowledge Graph Statistics ({workspace_id or 'default'}):\n"
            f"- Total Nodes: {stats['total_nodes']}\n"
            f"- Total Edges: {stats['total_edges']}\n"
            f"- Concepts: {stats['concept_nodes']}\n"
            f"- Entities: {stats['entity_nodes']}\n"
            f"- Tags: {stats['tag_nodes']}\n"
            f"- Avg Connections: {stats['avg_connections']}"
        )
    except Exception as e:
        return f"Error getting stats: {str(e)}"

@mcp.tool()
def mind_map_report(workspace_id: Optional[str] = None) -> str:
    """Generate a JSON report of the knowledge graph with summary stats and top nodes.

    Returns a JSON object containing:
    - summary: total nodes, edges, concepts, entities, tags
    - top_nodes: top 5 highest-importance nodes with their edges and connected tags

    Args:
        workspace_id: Unique identifier for the person or workspace (e.g., friend's name).
    """
    try:
        current_store = get_store(workspace_id)
        stats = current_store.get_stats()

        # Build summary
        summary: dict[str, Any] = {
            "workspace": workspace_id or "default",
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"],
            "concepts": stats["concept_nodes"],
            "entities": stats["entity_nodes"],
            "tags": stats["tag_nodes"],
            "avg_connections": stats["avg_connections"],
        }

        # Get all nodes with metadatas and documents
        all_data = current_store.collection.get(include=["metadatas", "documents"])
        if not all_data["ids"]:
            return json.dumps({"summary": summary, "top_nodes": []}, indent=2)

        # Calculate importance for each node and pair with its data
        scored: list[tuple[float, int]] = []
        for i, node_id in enumerate(all_data["ids"]):
            importance = current_store.calculate_importance(node_id)
            scored.append((importance, i))

        # Sort descending by importance, take top 5
        scored.sort(key=lambda x: x[0], reverse=True)
        top_5 = scored[:5]

        top_nodes: list[dict[str, Any]] = []
        for importance, i in top_5:
            node_id = all_data["ids"][i]
            doc = all_data["documents"][i] if all_data["documents"] else ""
            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}

            # Get edges for this node
            edges = current_store.get_edges(node_id)
            edge_list = [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "relation_type": e.relation_type,
                }
                for e in edges
            ]

            # Find connected tags: look at edge neighbors and filter for tag type
            connected_tag_ids: set[str] = set()
            for e in edges:
                neighbor_id = e.target if e.source == node_id else e.source
                connected_tag_ids.add(neighbor_id)

            tags: list[str] = []
            if connected_tag_ids:
                neighbors = current_store.collection.get(
                    ids=list(connected_tag_ids), include=["metadatas", "documents"]
                )
                for j, nid in enumerate(neighbors["ids"]):
                    n_meta = neighbors["metadatas"][j] if neighbors["metadatas"] else {}
                    if n_meta.get("type") == NodeType.TAG.value:
                        n_doc = neighbors["documents"][j] if neighbors["documents"] else ""
                        tags.append(n_doc)

            top_nodes.append({
                "id": node_id,
                "document": doc,
                "type": meta.get("type", "unknown"),
                "importance_score": round(importance, 4),
                "connection_count": meta.get("connection_count", 0),
                "edges": edge_list,
                "tags": tags,
            })

        return json.dumps({"summary": summary, "top_nodes": top_nodes}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def mind_map_prune(workspace_id: Optional[str] = None) -> str:
    """Prune the least important 10% of nodes from the knowledge graph.

    Only concept and entity nodes are direct prune candidates (sorted by importance
    score ascending). Tags are removed only if all their edges connect exclusively
    to nodes in the prune set.

    Returns a JSON report with deleted_nodes, deleted_tags, deleted_edges_count,
    and a human-readable summary.

    Args:
        workspace_id: Unique identifier for the person or workspace (e.g., friend's name).
    """
    try:
        current_store = get_store(workspace_id)

        # 1. Get all nodes
        all_data = current_store.collection.get(include=["metadatas", "documents"])
        if not all_data["ids"]:
            return json.dumps({
                "deleted_nodes": [],
                "deleted_tags": [],
                "deleted_edges_count": 0,
                "summary": "Graph is empty, nothing to prune.",
            }, indent=2)

        # 2. Separate candidates (concept/entity) from tags
        candidates: list[tuple[float, int]] = []  # (importance, index)
        tag_indices: list[int] = []
        for i, node_id in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
            node_type = meta.get("type", "concept")
            if node_type == NodeType.TAG.value:
                tag_indices.append(i)
            else:
                importance = current_store.calculate_importance(node_id)
                candidates.append((importance, i))

        if not candidates:
            return json.dumps({
                "deleted_nodes": [],
                "deleted_tags": [],
                "deleted_edges_count": 0,
                "summary": "No concept or entity nodes to prune.",
            }, indent=2)

        # 3. Sort ascending by importance, take bottom 10% (at least 1)
        candidates.sort(key=lambda x: x[0])
        prune_count = max(1, math.floor(len(candidates) * 0.1))
        prune_targets = candidates[:prune_count]
        prune_indices = {idx for _, idx in prune_targets}
        prune_ids = {all_data["ids"][idx] for _, idx in prune_targets}

        # 4. Collect neighbor tag IDs for prune targets
        tag_neighbor_ids: set[str] = set()
        for node_id in prune_ids:
            edges = current_store.get_edges(node_id)
            for edge in edges:
                neighbor_id = edge.target if edge.source == node_id else edge.source
                # Check if neighbor is a tag
                if neighbor_id not in prune_ids:
                    tag_neighbor_ids.add(neighbor_id)

        # 5. Determine which tags to remove: a tag is removed only if ALL its edges
        #    connect to nodes in the prune set
        tags_to_remove: set[str] = set()
        for tag_id in tag_neighbor_ids:
            # Verify it's actually a tag node
            tag_node = current_store.get_node(tag_id)
            if not tag_node or tag_node.metadata.type != NodeType.TAG:
                continue
            tag_edges = current_store.get_edges(tag_id)
            if not tag_edges:
                continue
            all_connected_to_prune = all(
                (e.target if e.source == tag_id else e.source) in prune_ids
                for e in tag_edges
            )
            if all_connected_to_prune:
                tags_to_remove.add(tag_id)

        # 6. Delete edges for all prune targets
        total_deleted_edges = 0
        for node_id in prune_ids:
            total_deleted_edges += current_store.delete_edges_for_node(node_id)

        # 7. Delete edges for single-connected tags (may have remaining edges)
        for tag_id in tags_to_remove:
            total_deleted_edges += current_store.delete_edges_for_node(tag_id)

        # 8. Build report data before deleting nodes
        deleted_nodes_info: list[dict[str, Any]] = []
        for _, idx in prune_targets:
            node_id = all_data["ids"][idx]
            doc = all_data["documents"][idx] if all_data["documents"] else ""
            meta = all_data["metadatas"][idx] if all_data["metadatas"] else {}
            deleted_nodes_info.append({
                "id": node_id,
                "document": doc,
                "type": meta.get("type", "unknown"),
            })

        deleted_tags_info: list[dict[str, str]] = []
        for tag_id in tags_to_remove:
            tag_node = current_store.get_node(tag_id)
            if tag_node:
                deleted_tags_info.append({
                    "id": tag_id,
                    "document": tag_node.document,
                })

        # 9. Delete nodes from ChromaDB
        for node_id in prune_ids:
            current_store.delete_node(node_id)
        for tag_id in tags_to_remove:
            current_store.delete_node(tag_id)

        summary = (
            f"Pruned {len(prune_ids)} node(s) and {len(tags_to_remove)} tag(s) "
            f"from workspace '{workspace_id or 'default'}'. "
            f"Removed {total_deleted_edges} edge(s)."
        )

        return json.dumps({
            "deleted_nodes": deleted_nodes_info,
            "deleted_tags": deleted_tags_info,
            "deleted_edges_count": total_deleted_edges,
            "summary": summary,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
