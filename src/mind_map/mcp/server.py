import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path to ensure imports work correctly
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from fastmcp import FastMCP
from mind_map.agents.pipeline import ingest_memo
from mind_map.core.graph_store import GraphStore
from mind_map.core.processing_llm import get_processing_llm

# Initialize FastMCP
mcp = FastMCP("MindMap")

# Setup project paths
# Default to project root 'data' folder
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Initialize Store
store = GraphStore(DATA_DIR)

@mcp.tool()
def mind_map_retrieve(query: str, n_results: int = 5) -> str:
    """Retrieve relevant context from the knowledge graph based on a query.
    
    This tool performs a semantic similarity search and boosts results based on 
    graph connectivity and importance.
    
    Args:
        query: The search query or question.
        n_results: Number of relevant snippets to return (default 5).
    """
    try:
        # 1. Similarity search
        nodes = store.query_similar(query, n_results=n_results)
        if not nodes:
            return "No relevant information found in the knowledge graph."
        
        # 2. Enrich with relation factors and re-sort
        nodes = store.enrich_context_nodes(nodes)
        
        # 3. Format results
        output = ["### Relevant Context from Mind Map:"]
        for node in nodes:
            # combined score = importance * (1 + relation_factor)
            score = node.metadata.importance_score * (1 + (node.relation_factor or 0))
            output.append(f"- [{node.metadata.type.value}] (Relevance: {score:.2f}): {node.document}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

@mcp.tool()
def mind_map_memo(text: str) -> str:
    """Ingest new information or a Q&A pair into the knowledge graph.
    
    Use this to 'remember' important facts, decisions, or the result of a conversation.
    
    Args:
        text: The text content to store. Usually formatted as 'Q: question A: answer' for conversations.
    """
    try:
        llm = get_processing_llm()
        success, message, node_ids = ingest_memo(text=text, store=store, llm=llm)
        
        if success:
            return f"Successfully stored knowledge. {message} (Total nodes created: {len(node_ids)})"
        else:
            return f"Information was not stored: {message}"
    except Exception as e:
        return f"Error storing knowledge: {str(e)}"

@mcp.tool()
def mind_map_stats() -> str:
    """Get statistics about the current state of the knowledge graph."""
    try:
        stats = store.get_stats()
        return (
            f"### Knowledge Graph Statistics:\n"
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
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mcp.run()
