import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path to ensure imports work correctly
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from fastmcp import FastMCP
from mind_map.app.pipeline import ingest_memo
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

if __name__ == "__main__":
    mcp.run()
