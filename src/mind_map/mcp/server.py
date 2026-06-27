import json
import sys
import time  # noqa: F401 — re-exported for test patchability (test_error_json_is_valid)
from pathlib import Path
from typing import Any, Optional

# Add src to path to ensure imports work correctly
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from fastmcp import FastMCP

# Back-compat alias for older tests/callers that still patch `ingest_memo`.
# Re-exported from services so the path remains `mind_map.mcp.server.ingest_memo`.
from mind_map.app.services import ingest_memo  # noqa: F401

from mind_map.app.services import (
    ask_question as _ask_question,
    format_stats_text as _format_stats_text,
    graph_stats as _graph_stats,
    health_check as _health_check,
    memo_ingest as _memo_ingest,
    prune_graph as _prune_graph,
    report_graph as _report_graph,
    resolve_store as _resolve_store,
    retrieve_context as _retrieve_context,
)
from mind_map.core.config import get_data_dir
from mind_map.core.schemas import Edge, NodeType
from mind_map.rag.graph_store import GraphStore
from mind_map.rag.llm_status import get_llm_status

# Re-exported for test patchability. Tests patch these at the `mind_map.mcp.server`
# module level (e.g., `patch("mind_map.mcp.server.check_ollama_available", ...)`).
# The service layer uses its own lazy imports, so these names are not called
# during normal operation — they exist solely to preserve test contract.
from mind_map.processor.processing_llm import (  # noqa: F401
    check_ollama_available,
    get_available_models,
    get_processing_llm,
    get_selected_model,
)

# Initialize FastMCP
mcp = FastMCP("MindMap")

# Setup project paths
DEFAULT_DATA_DIR = get_data_dir()

# Dictionary to manage multiple stores
# workspace_id -> GraphStore instance
stores: dict[str, GraphStore] = {}

def get_store(workspace_id: Optional[str] = None) -> GraphStore:
    """Get or initialize a GraphStore for a specific workspace.

    Test surface: tests patch `mind_map.mcp.server.get_store` and the
    `stores` dict directly. Behavior preserved verbatim.
    """
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
def mind_map_retrieve(
    query: str,
    n_results: int = 5,
    show_context: bool = True,
    max_context_per_node: int = 3,
    data_dir: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Retrieve relevant context from the knowledge graph based on a query.

    Args:
        query: The search query or question.
        n_results: Number of relevant snippets to return (default 5).
        show_context: Include connected nodes for each result (default True).
        max_context_per_node: Maximum neighbors per result (default 3, 0=unlimited).
        data_dir: Optional path to a custom data directory (CLI parity).
        workspace_id: Unique identifier for the person or workspace.
    """
    try:
        if data_dir is not None:
            store, _ = _resolve_store(data_dir=data_dir)
        else:
            store = get_store(workspace_id)

        lines = _retrieve_context(
            query,
            store,
            n_results=n_results,
            show_context=show_context,
            max_context_per_node=max_context_per_node,
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

@mcp.tool()
def mind_map_memo(
    text: str,
    source: Optional[str] = None,
    local: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Ingest new information or a Q&A pair into the knowledge graph.

    Args:
        text: The text content to store.
        source: Optional source identifier (CLI parity).
        local: Use a local OpenAI-compatible model. ``""`` = auto-resolve
            first model, or pass an explicit model id. CLI parity — adds the
            ability to choose a local model target (new capability for MCP).
        data_dir: Optional path to a custom data directory (CLI parity).
        workspace_id: Unique identifier for the person or workspace.

    Note:
        Per CLI parity, requires ``MINIMAX_API_KEY`` to be set unless
        ``local`` is provided. Raises an explicit error otherwise — does
        NOT silently fall back to a cloud-auto LLM.
    """
    try:
        if data_dir is not None:
            store, _ = _resolve_store(data_dir=data_dir)
        else:
            store = get_store(workspace_id)

        success, message, node_ids = _memo_ingest(
            text,
            store,
            local=local,
            source=source,
        )
        if success:
            return (
                f"Successfully stored knowledge in '{workspace_id or 'default'}'. "
                f"{message} (Total nodes created: {len(node_ids)})"
            )
        return f"Information was not stored: {message}"
    except ValueError as e:
        # MINIMAX_API_KEY not set + no local
        return f"Configuration error: {e}. Set MINIMAX_API_KEY or pass `local`."
    except Exception as e:
        return f"Error storing knowledge: {str(e)}"


@mcp.tool()
def mind_map_ask(
    query: str,
    depth: int = 2,
    n_results: int = 5,
    data_dir: Optional[str] = None,
    model: Optional[str] = None,
    back_feed: bool = False,
    workspace_id: Optional[str] = None,
) -> str:
    """Query the knowledge graph with a RAG-enhanced LLM response.

    Args:
        query: The question to ask.
        depth: Graph traversal depth (unused, kept for CLI parity).
        n_results: Number of context nodes to retrieve (default 5).
        data_dir: Optional path to a custom data directory (CLI parity).
        model: Specific processing model to use for Q&A back-feed.
        back_feed: If True, write the Q&A pair back to the graph (CLI parity).
            If False (default), this is a pure read — no graph writes.
        workspace_id: Unique identifier for the person or workspace.

    Note:
        MCP default is ``back_feed=False`` to avoid surprising callers with
        writes. CLI ask always back-feeds; pass ``back_feed=True`` here to
        opt into CLI parity behavior.
    """
    try:
        if data_dir is not None:
            store, _ = _resolve_store(data_dir=data_dir)
        else:
            store = get_store(workspace_id)

        result = _ask_question(
            query,
            store,
            depth=depth,
            model=model,
            back_feed=back_feed,
        )
        # Return the response as plain text (no JSON wrapping). Caller can
        # inspect context_nodes via retrieve if they want structured data.
        return result["response"]
    except Exception as e:
        return f"Error answering question: {str(e)}"


@mcp.tool()
def mind_map_stats(
    data_dir: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Get statistics about the current state of the knowledge graph.

    Args:
        data_dir: Optional path to a custom data directory (CLI parity).
        workspace_id: Unique identifier for the person or workspace.
    """
    try:
        if data_dir is not None:
            store, _ = _resolve_store(data_dir=data_dir)
        else:
            store = get_store(workspace_id)

        stats = _graph_stats(store)
        return _format_stats_text(stats, workspace_id=workspace_id or "default")
    except Exception as e:
        return f"Error getting stats: {str(e)}"

@mcp.tool()
def mind_map_report(
    data_dir: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Generate a JSON report of the knowledge graph with summary stats and top nodes.

    Returns a JSON object containing:
    - summary: total nodes, edges, concepts, entities, tags
    - top_nodes: top 5 highest-importance nodes with their edges and connected tags

    Args:
        data_dir: Optional path to a custom data directory (CLI parity).
        workspace_id: Unique identifier for the person or workspace.
    """
    try:
        if data_dir is not None:
            store, _ = _resolve_store(data_dir=data_dir)
        else:
            store = get_store(workspace_id)

        report = _report_graph(store)
        # Override workspace with the actual requested one
        report["summary"]["workspace"] = workspace_id or "default"
        return json.dumps(report, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def mind_map_prune(
    percent: float = 0.1,
    data_dir: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Prune the least important nodes from the knowledge graph.

    Only concept and entity nodes are direct prune candidates (sorted by importance
    score ascending). Tags are removed only if all their edges connect exclusively
    to nodes in the prune set.

    Returns a JSON report with deleted_nodes, deleted_tags, deleted_edges_count,
    and a human-readable summary.

    Args:
        percent: Percentage of nodes to prune (0.0-1.0, default 0.1). CLI parity.
        data_dir: Optional path to a custom data directory (CLI parity).
        workspace_id: Unique identifier for the person or workspace.
    """
    try:
        if data_dir is not None:
            store, _ = _resolve_store(data_dir=data_dir)
        else:
            store = get_store(workspace_id)

        result = _prune_graph(
            store,
            percent=percent,
            workspace_id=workspace_id or "default",
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def mind_map_health(
    data_dir: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Run a comprehensive health check on the Mind Map system.

    Checks Ollama connectivity, database connections (ChromaDB + SQLite),
    processing LLM availability, and runs integration tests (similarity search,
    memo ingestion, data persistence).

    Args:
        data_dir: Optional path to a custom data directory (CLI parity).
        workspace_id: Unique identifier for the person or workspace.

    Note:
        ``get_store(workspace_id)`` is passed to the service as a ``store_getter``
        so the service can resolve the store lazily inside per-section try/except
        blocks. When the patch raises, each dependent check (chromadb, sqlite,
        integration tests) is marked failed individually instead of bubbling up
        to the wrapper's outer try/except — this preserves the test contract
        where each section reports its own status.
    """
    try:
        ws = workspace_id or "default"
        # Pass store=None; the service will lazily call store_getter(workspace_id)
        # inside each section's try/except. This preserves the OLD architecture's
        # per-section failure semantics for test patches on get_store.
        if data_dir is not None:
            # When data_dir is provided, eagerly resolve via _resolve_store so we
            # still get a populated checks dict for store-dependent sections even
            # if the user's store_getter is patched to raise (test isolation).
            try:
                store, _ = _resolve_store(data_dir=data_dir)
            except Exception:
                store = None
            result = _health_check(store, workspace_id=ws)
        else:
            result = _health_check(None, workspace_id=ws, store_getter=get_store)
        # Override timestamp with the wrapper's own call so tests can patch
        # mind_map.mcp.server.time to simulate a timestamp failure. The service
        # also computes a timestamp, but this one wins for the response shape.
        result["timestamp"] = time.time()
        return json.dumps(result, indent=2)
    except Exception as e:
        # Critical failure (e.g., patched time.time raises): return minimal
        # unhealthy response. Per-section failures (get_store, ingest_memo) are
        # captured by the service's own try/except blocks.
        return json.dumps({"status": "unhealthy", "error": str(e)})


if __name__ == "__main__":
    mcp.run()
