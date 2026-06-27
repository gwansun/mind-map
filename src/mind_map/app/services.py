"""Shared business-logic layer for CLI commands and MCP tools.

Pure functions that operate on a GraphStore. No Typer, no Rich, no FastMCP.
Both ``mind_map.app.cli.main`` (Typer commands) and ``mind_map.mcp.server``
(FastMCP tools) wrap these to format output for their respective surfaces.

Goal: one source of truth so CLI and MCP cannot drift.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from mind_map.app.pipeline import ingest_memo_cli, ingest_memo_internal
from mind_map.core.config import DEFAULT_DATA_DIR
from mind_map.core.schemas import Edge, NodeType
from mind_map.processor.cli_executor import (
    CLIExecutionError,
    LocalTarget,
    MemoTarget,
    MiniMaxTarget,
    resolve_local_model,
)
from mind_map.rag.graph_store import GraphStore

# Back-compat re-export for tests that patch ``mind_map.mcp.server.ingest_memo``.
# The MCP health-check integration test (test_mcp_health.py line 322) patches
# this name; if it disappears, the patch silently no-ops and integration-test
# assertions fail. Keep this alias stable.
ingest_memo = ingest_memo_internal


# ---------------------------------------------------------------------------
# Store resolution
# ---------------------------------------------------------------------------


def resolve_store(
    *,
    data_dir: Path | str | None = None,
    workspace_id: str | None = None,
) -> tuple[GraphStore, Path]:
    """Resolve data dir + workspace, initialize + return store.

    Precedence:
        1. ``data_dir`` provided → use it directly (workspace_id ignored)
        2. ``workspace_id`` provided → ``DEFAULT_DATA_DIR/workspaces/<id>/``
        3. neither → ``DEFAULT_DATA_DIR`` (default workspace)

    The workspace subfolder layout is MCP-only. CLI commands pass ``data_dir``
    and never set ``workspace_id``.

    Returns:
        ``(store, effective_data_dir)``. The store is already initialized.
    """
    if data_dir is not None:
        path = Path(data_dir).expanduser().resolve()
    elif workspace_id is not None:
        path = (DEFAULT_DATA_DIR / "workspaces" / workspace_id).expanduser().resolve()
    else:
        path = DEFAULT_DATA_DIR.expanduser().resolve()

    store = GraphStore(path)
    store.initialize()
    return store, path


# ---------------------------------------------------------------------------
# Memo target resolution
# ---------------------------------------------------------------------------


def parse_memo_target(*, local: str | None, api_key: str | None) -> MemoTarget:
    """Resolve a ``MemoTarget`` from a CLI/MCP ``local`` flag and an optional key.

    - ``local`` not None (including ``""``): build ``LocalTarget`` via
      ``resolve_local_model``. ``local=""`` triggers auto-resolution (queries
      ``/models`` endpoint for the first model id).
    - ``local`` None and ``api_key`` provided: build ``MiniMaxTarget``.
    - Both None / missing: raises ``ValueError`` — caller should check first.

    Raises:
        ``CLIExecutionError`` on local-model resolution failure.
        ``ValueError`` if both ``local`` and ``api_key`` are missing.
    """
    if local is not None:
        # Lazy-import so tests patching
        # `mind_map.processor.cli_executor.resolve_local_model` take effect.
        from mind_map.processor.cli_executor import resolve_local_model as _resolve_local_model

        model_name = _resolve_local_model(model=local or None)
        return LocalTarget(model=model_name)
    if api_key:
        return MiniMaxTarget(api_key=api_key)
    raise ValueError("Either `local` or `api_key` must be supplied")


# ---------------------------------------------------------------------------
# Memo ingestion
# ---------------------------------------------------------------------------


def memo_ingest(
    text: str,
    store: GraphStore,
    *,
    local: str | None = None,
    source: str | None = None,
) -> tuple[bool, str, list[str]]:
    """Ingest a memo through the direct MiniMax / local pipeline.

    Mirrors ``mind-map memo`` exactly. Uses ``ingest_memo_cli`` (the direct
    MiniMax or local OpenAI-compatible path), NOT ``ingest_memo_internal``
    (legacy LangChain path) — this closes the silent MCP behavioral drift.

    Args:
        text: Memo text to ingest.
        store: Initialized GraphStore.
        local: Model id (``""`` = auto-resolve first model) for LocalTarget.
        source: Optional source identifier passed through to ingestion.

    Returns:
        ``(success, message, node_ids)``.

    Raises:
        ``ValueError`` if ``local`` is None and ``MINIMAX_API_KEY`` env unset.
        ``CLIExecutionError`` on local-model resolution failure.
        ``RuntimeError`` on ``ingest_memo_cli`` failure with details.

    Note:
        Caller is responsible for the ``data_dir.exists()`` check BEFORE
        calling this — preserves CLI exit-code-1 behavior.
    """
    if local is None:
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY not set. Set it in your environment or pass "
                "`local` for local mode."
            )
        target: MemoTarget = MiniMaxTarget(api_key=api_key)
    else:
        # Lazy-import resolve_local_model so tests patching either
        # `mind_map.processor.cli_executor.resolve_local_model` (legacy
        # CLI test) or `mind_map.app.services.resolve_local_model` (new
        # services test) both work — the lookup happens at call time.
        from mind_map.processor.cli_executor import resolve_local_model as _resolve_local_model

        # `local=""` triggers auto-resolve via resolve_local_model
        model_name = _resolve_local_model(model=local or None)
        target = LocalTarget(model=model_name)

    # Lazy-import so tests patching `mind_map.app.pipeline.ingest_memo_cli`
    # (the original call site in CLI ask/memo) keep working. Otherwise the
    # patch bypasses our wrapper and our service never sees the mock.
    from mind_map.app.pipeline import ingest_memo_cli as _ingest_memo_cli

    success, message, node_ids = _ingest_memo_cli(
        text,
        store,
        target=target,
        source_id=source,
    )
    return success, message, node_ids


# ---------------------------------------------------------------------------
# Retrieve (vector search + neighbor expansion)
# ---------------------------------------------------------------------------


def retrieve_context(
    query: str,
    store: GraphStore,
    *,
    n_results: int = 5,
    show_context: bool = True,
    max_context_per_node: int = 3,
) -> list[str]:
    """Format retrieval results as CLI does.

    Returns a list of lines (caller joins with newline). Mirrors the CLI
    ``retrieve`` output format exactly:

    - Header: ``### Relevant Context from Mind Map:``
    - Per result: ``- [<type>] (Relevance: <score>): <doc>``
    - Per neighbor (if ``show_context``): ``  └─ related [<type>] via <relation>: <doc>``

    Empty result returns a single line: ``No relevant information found in the knowledge graph.``

    Note:
        Caller is responsible for the ``data_dir.exists()`` check BEFORE
        calling this — CLI prints that message and exits 0.
    """
    nodes = store.query_similar(query, n_results=n_results)
    if not nodes:
        return ["No relevant information found in the knowledge graph."]

    nodes = store.enrich_context_nodes(nodes)

    connected_context: dict[str, list] = {}
    if show_context:
        node_ids = [n.id for n in nodes]
        connected_context = store.get_connected_context(node_ids)

    lines = ["### Relevant Context from Mind Map:"]
    for node in nodes:
        score = node.metadata.importance_score * (1 + (node.relation_factor or 0))
        lines.append(f"- [{node.metadata.type.value}] (Relevance: {score:.2f}): {node.document}")

        if show_context and connected_context.get(node.id):
            neighbors = connected_context[node.id]
            if max_context_per_node > 0:
                neighbors = neighbors[:max_context_per_node]
            for neighbor_node, edge in neighbors:
                lines.append(
                    f"  └─ related [{neighbor_node.metadata.type.value}] "
                    f"via {edge.relation_type}: {neighbor_node.document}"
                )
    return lines


# ---------------------------------------------------------------------------
# Ask (RAG with optional back-feed)
# ---------------------------------------------------------------------------


def ask_question(
    query: str,
    store: GraphStore,
    *,
    depth: int = 2,                  # unused, kept for CLI parity
    model: str | None = None,        # processing LLM override for back_feed path
    back_feed: bool = False,         # CLI ask always back-feeds; MCP defaults off
) -> dict[str, Any]:
    """Run a RAG query against the graph, optionally back-feeding the Q&A.

    When ``back_feed=False`` (MCP default): pure read — retrieve + LLM answer,
    no writes. ``update_interaction``, ``ingest_memo_internal``, and edge
    linking are all skipped.

    When ``back_feed=True`` (CLI parity opt-in): write — also ingests the Q&A
    pair back into the graph and links it to context nodes.

    Returns dict with keys: ``response``, ``context_nodes``, ``qa_node_ids``,
    ``status``. ``status`` is one of ``{"answered", "no_llm", "no_context"}``.

    When no reasoning LLM is available, ``response`` contains a textual fallback
    showing raw context nodes (string form of CLI fallback path).
    """
    # Lazy imports to avoid pulling reasoning-LLM dependencies on import
    from mind_map.rag.reasoning_llm import get_reasoning_llm
    from mind_map.rag.response_generator import ResponseGenerator

    nodes = store.query_similar(query, n_results=5)
    if nodes:
        nodes = store.enrich_context_nodes(nodes)

    llm = get_reasoning_llm()
    if not llm:
        if nodes:
            bullets = "\n".join(f"- {n.document[:200]}..." for n in nodes)
            response = (
                "Reasoning LLM not available. Raw context:\n" + bullets
            )
            status = "no_llm"
        else:
            response = "Reasoning LLM not available and no relevant context found."
            status = "no_llm"
        return {
            "response": response,
            "context_nodes": nodes,
            "qa_node_ids": [],
            "status": status,
        }

    generator = ResponseGenerator(llm)
    response = generator.generate_sync(query, nodes)

    qa_node_ids: list[str] = []
    if back_feed:
        # Refresh interaction timestamps
        # NOTE: GraphStore.update_interaction is referenced by CLI ask but is
        # not implemented in GraphStore. Preserved verbatim for CLI parity —
        # the AttributeError is caught here so the back-feed path doesn't
        # blow up. pyright: ignore[reportAttributeAccessIssue]
        for node in nodes:
            try:
                store.update_interaction(node.id)  # type: ignore[attr-defined]
            except AttributeError:
                pass

        # Extract Q&A back into the graph via internal ingestion path
        from mind_map.processor.processing_llm import get_processing_llm

        processing_llm = get_processing_llm(model_name=model)
        qa_text = f"Q: {query}\nA: {response}"
        success, _msg, qa_node_ids = ingest_memo_internal(
            text=qa_text,
            store=store,
            llm=processing_llm,
            source_id=f"qa_{query[:50]}",
        )

        # Link Q&A nodes to context nodes
        if success and qa_node_ids and nodes:
            qa_concept_id = qa_node_ids[0]
            for context_node in nodes:
                store.add_edge(Edge(
                    source=qa_concept_id,
                    target=context_node.id,
                    relation_type="derived_from",
                ))

    return {
        "response": response,
        "context_nodes": nodes,
        "qa_node_ids": qa_node_ids,
        "status": "answered",
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def graph_stats(store: GraphStore) -> dict[str, Any]:
    """Return normalized stats dict from the store.

    Returns the dict from ``store.get_stats()`` with the keys callers expect:
    ``total_nodes``, ``total_edges``, ``concept_nodes``, ``entity_nodes``,
    ``tag_nodes``, ``avg_connections``.
    """
    return store.get_stats()


def format_stats_text(
    stats: dict[str, Any], *, workspace_id: str = "default"
) -> str:
    """Format a stats dict as the MCP ``mind_map_stats`` text block.

    This exact format is locked by test contracts:
    ``### Knowledge Graph Statistics (<workspace>):`` header + 6 metric lines.
    """
    return (
        f"### Knowledge Graph Statistics ({workspace_id}):\n"
        f"- Total Nodes: {stats['total_nodes']}\n"
        f"- Total Edges: {stats['total_edges']}\n"
        f"- Concepts: {stats['concept_nodes']}\n"
        f"- Entities: {stats['entity_nodes']}\n"
        f"- Tags: {stats['tag_nodes']}\n"
        f"- Avg Connections: {stats['avg_connections']}"
    )


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


def prune_graph(
    store: GraphStore,
    *,
    percent: float = 0.1,
    workspace_id: str = "default",
) -> dict[str, Any]:
    """Prune low-importance nodes from the graph.

    Returns a dict matching the MCP ``mind_map_prune`` JSON contract exactly:
    ``deleted_nodes``, ``deleted_tags``, ``deleted_edges_count``, ``summary``.

    Algorithm (identical to current MCP ``mind_map_prune`` body):
        1. Collect all nodes, separate concept/entity candidates from tags
        2. Sort candidates ascending by importance
        3. Take bottom ``percent`` (clamped to >= 1)
        4. Collect tag-neighbor IDs of prune targets
        5. A tag is removed only if ALL its edges connect to prune targets
        6. Delete edges for prune targets + tags-to-remove
        7. Delete nodes from ChromaDB

    ``summary`` includes the workspace name and counts (test contract).
    """
    all_data = store.collection.get(include=["metadatas", "documents"])
    if not all_data["ids"]:
        return {
            "deleted_nodes": [],
            "deleted_tags": [],
            "deleted_edges_count": 0,
            "summary": f"Graph is empty in workspace '{workspace_id}', nothing to prune.",
        }

    candidates: list[tuple[float, int]] = []
    tag_indices: list[int] = []
    for i, node_id in enumerate(all_data["ids"]):
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        node_type = meta.get("type", "concept")
        if node_type == NodeType.TAG.value:
            tag_indices.append(i)
        else:
            importance = store.calculate_importance(node_id)
            candidates.append((importance, i))

    if not candidates:
        return {
            "deleted_nodes": [],
            "deleted_tags": [],
            "deleted_edges_count": 0,
            "summary": f"No concept or entity nodes to prune in '{workspace_id}'.",
        }

    candidates.sort(key=lambda x: x[0])
    prune_count = max(1, math.floor(len(candidates) * percent))
    prune_targets = candidates[:prune_count]
    prune_ids = {all_data["ids"][idx] for _, idx in prune_targets}

    # Collect neighbor tag IDs of prune targets
    tag_neighbor_ids: set[str] = set()
    for node_id in prune_ids:
        edges = store.get_edges(node_id)
        for edge in edges:
            neighbor_id = edge.target if edge.source == node_id else edge.source
            if neighbor_id not in prune_ids:
                tag_neighbor_ids.add(neighbor_id)

    # A tag is removed only if ALL its edges connect to prune targets
    tags_to_remove: set[str] = set()
    for tag_id in tag_neighbor_ids:
        tag_node = store.get_node(tag_id)
        if not tag_node or tag_node.metadata.type != NodeType.TAG:
            continue
        tag_edges = store.get_edges(tag_id)
        if not tag_edges:
            continue
        all_connected_to_prune = all(
            (e.target if e.source == tag_id else e.source) in prune_ids
            for e in tag_edges
        )
        if all_connected_to_prune:
            tags_to_remove.add(tag_id)

    # Delete edges for prune targets + tags-to-remove
    total_deleted_edges = 0
    for node_id in prune_ids:
        total_deleted_edges += store.delete_edges_for_node(node_id)
    for tag_id in tags_to_remove:
        total_deleted_edges += store.delete_edges_for_node(tag_id)

    # Build deleted_nodes info before deleting
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
        tag_node = store.get_node(tag_id)
        if tag_node:
            deleted_tags_info.append({"id": tag_id, "document": tag_node.document})

    # Delete nodes
    for node_id in prune_ids:
        store.delete_node(node_id)
    for tag_id in tags_to_remove:
        store.delete_node(tag_id)

    summary = (
        f"Pruned {len(prune_ids)} node(s) and {len(tags_to_remove)} tag(s) "
        f"from workspace '{workspace_id}'. "
        f"Removed {total_deleted_edges} edge(s)."
    )
    return {
        "deleted_nodes": deleted_nodes_info,
        "deleted_tags": deleted_tags_info,
        "deleted_edges_count": total_deleted_edges,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report_graph(store: GraphStore) -> dict[str, Any]:
    """Generate a JSON report of the graph: summary stats + top 5 nodes.

    Returns dict matching the MCP ``mind_map_report`` JSON contract exactly:
    ``summary`` (counts + workspace + avg_connections) and ``top_nodes``
    (top 5 by importance, each with edges + connected tags).
    """
    stats = store.get_stats()

    summary: dict[str, Any] = {
        "workspace": "default",
        "total_nodes": stats["total_nodes"],
        "total_edges": stats["total_edges"],
        "concepts": stats["concept_nodes"],
        "entities": stats["entity_nodes"],
        "tags": stats["tag_nodes"],
        "avg_connections": stats["avg_connections"],
    }

    all_data = store.collection.get(include=["metadatas", "documents"])
    if not all_data["ids"]:
        return {"summary": summary, "top_nodes": []}

    scored: list[tuple[float, int]] = []
    for i, node_id in enumerate(all_data["ids"]):
        importance = store.calculate_importance(node_id)
        scored.append((importance, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_5 = scored[:5]

    top_nodes: list[dict[str, Any]] = []
    for importance, i in top_5:
        node_id = all_data["ids"][i]
        doc = all_data["documents"][i] if all_data["documents"] else ""
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}

        edges = store.get_edges(node_id)
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
            neighbors = store.collection.get(
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

    return {"summary": summary, "top_nodes": top_nodes}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def health_check(
    store: GraphStore | None,
    *,
    workspace_id: str,
    store_getter: Callable[[str], GraphStore] | None = None,
) -> dict[str, Any]:
    """Run a comprehensive health check on the Mind Map system.

    Returns the full health-check dict matching the MCP ``mind_map_health``
    JSON contract. Lazy-imports Ollama/LLM modules to avoid pulling them
    when not needed.

    When ``store_getter`` is provided, sections that need the store will call
    ``store_getter(workspace_id)`` lazily inside their per-section try/except.
    This preserves the OLD architecture's per-section failure semantics: when
    the store cannot be resolved, only the dependent checks are marked failed
    while the rest of the report is still populated.

    Uses the module-level ``ingest_memo`` alias (which points to
    ``ingest_memo_internal``) for the memo-ingestion integration test, so that
    tests patching ``mind_map.mcp.server.ingest_memo`` continue to work after
    the MCP server delegates here.
    """
    # Lazy imports: health checks should not force Ollama/LLM deps at import
    import time as _time

    from mind_map.processor.processing_llm import (
        check_ollama_available,
        get_available_models,
        get_selected_model,
    )
    from mind_map.rag.llm_status import get_llm_status

    def _resolve_store() -> GraphStore | None:
        """Resolve the store lazily, returning None if store_getter raises.

        Falls back to the pre-resolved ``store`` arg if no getter is provided.
        """
        if store_getter is not None:
            try:
                return store_getter(workspace_id)
            except Exception as e:
                _store_resolution_error = e
                return None
        return store

    checks: dict[str, Any] = {}

    # 1. Ollama connection
    try:
        ollama_up = check_ollama_available()
        if ollama_up:
            model = get_selected_model()
            available = get_available_models()
            model_found = model in available
            checks["ollama_connection"] = {
                "status": "pass" if model_found else "fail",
                "model": model,
                "details": (
                    "Model available" if model_found
                    else f"Model '{model}' not found in {available}"
                ),
            }
        else:
            checks["ollama_connection"] = {
                "status": "fail",
                "model": None,
                "details": "Ollama server not running",
            }
    except Exception as e:
        checks["ollama_connection"] = {
            "status": "fail",
            "model": None,
            "details": str(e),
        }

    # 2. ChromaDB connection
    chroma_store: GraphStore | None = None
    try:
        chroma_store = _resolve_store()
    except Exception as e:
        checks["chromadb_connection"] = {
            "status": "fail",
            "node_count": 0,
            "details": str(e),
        }
    if "chromadb_connection" not in checks:
        if chroma_store is None:
            checks["chromadb_connection"] = {
                "status": "fail",
                "node_count": 0,
                "details": "GraphStore unavailable",
            }
        else:
            try:
                node_count = chroma_store.collection.count()
                checks["chromadb_connection"] = {
                    "status": "pass",
                    "node_count": node_count,
                    "details": f"{node_count} nodes in collection",
                }
            except Exception as e:
                checks["chromadb_connection"] = {
                    "status": "fail",
                    "node_count": 0,
                    "details": str(e),
                }

    # 3. SQLite connection
    sqlite_store: GraphStore | None = None
    try:
        sqlite_store = _resolve_store()
    except Exception as e:
        checks["sqlite_connection"] = {
            "status": "fail",
            "edge_count": 0,
            "details": str(e),
        }
    if "sqlite_connection" not in checks:
        if sqlite_store is None:
            checks["sqlite_connection"] = {
                "status": "fail",
                "edge_count": 0,
                "details": "GraphStore unavailable",
            }
        else:
            try:
                cursor = sqlite_store.sqlite.execute("SELECT COUNT(*) FROM edges")
                edge_count = cursor.fetchone()[0]
                checks["sqlite_connection"] = {
                    "status": "pass",
                    "edge_count": edge_count,
                    "details": f"{edge_count} edges in database",
                }
            except Exception as e:
                checks["sqlite_connection"] = {
                    "status": "fail",
                    "edge_count": 0,
                    "details": str(e),
                }
    # 4. Processing LLM status
    try:
        llm_status = get_llm_status()
        proc = llm_status.get("processing_llm", {})
        checks["processing_llm"] = {
            "status": "available" if proc.get("status") == "online" else "unavailable",
            "provider": proc.get("provider", "unknown"),
            "model": proc.get("model", "unknown"),
        }
    except Exception:
        checks["processing_llm"] = {
            "status": "unavailable",
            "provider": "unknown",
            "model": "unknown",
        }

    # 5-7. Integration tests
    integration: dict[str, Any] = {}

    # Resolve the store lazily. If store_getter raises (or pre-resolved store
    # is None), mark all 3 integration sections as failed with the same error.
    store_resolution_error: str | None = None
    try:
        int_store = _resolve_store()
    except Exception as e:
        int_store = None
        store_resolution_error = str(e)

    if int_store is None:
        msg = store_resolution_error or "GraphStore unavailable"
        integration["similarity_search"] = {"status": "fail", "details": msg}
        integration["memo_ingestion"] = {
            "status": "fail",
            "nodes_created": 0,
            "details": msg,
        }
        integration["data_persistence"] = {"status": "fail", "details": msg}
    else:
        # 5. Similarity search
        test_id = f"_health_check_{uuid.uuid4().hex[:12]}"
        try:
            int_store.add_node(test_id, "health check similarity test node", NodeType.CONCEPT)
            results = int_store.query_similar("health check similarity test node", n_results=1)
            found = any(r.id == test_id for r in results)
            int_store.delete_node(test_id)
            integration["similarity_search"] = {
                "status": "pass" if found else "fail",
                "details": (
                    "Node inserted, queried, and cleaned up" if found
                    else "Query did not return test node"
                ),
            }
        except Exception as e:
            try:
                int_store.delete_node(test_id)
            except Exception:
                pass
            integration["similarity_search"] = {
                "status": "fail",
                "details": str(e),
            }

        # 6. Memo ingestion (heuristic only, no LLM cost)
        try:
            test_text = "Health check memo ingestion test for Python programming concepts"
            success, message, node_ids = ingest_memo(
                text=test_text, store=int_store, llm=None
            )
            for nid in node_ids:
                int_store.delete_edges_for_node(nid)
                int_store.delete_node(nid)
            integration["memo_ingestion"] = {
                "status": "pass" if success else "fail",
                "nodes_created": len(node_ids),
                "details": message,
            }
        except Exception as e:
            integration["memo_ingestion"] = {
                "status": "fail",
                "nodes_created": 0,
                "details": str(e),
            }

        # 7. Data persistence
        n1 = f"_health_check_{uuid.uuid4().hex[:12]}"
        n2 = f"_health_check_{uuid.uuid4().hex[:12]}"
        try:
            int_store.add_node(n1, "persistence test node A", NodeType.CONCEPT)
            int_store.add_node(n2, "persistence test node B", NodeType.CONCEPT)

            read_node = int_store.get_node(n1)
            if read_node is None:
                raise RuntimeError("Failed to read back node after insert")

            int_store.add_edge(Edge(source=n1, target=n2, relation_type="test_relation"))
            edges = int_store.get_edges(n1)
            edge_found = any(
                (e.source == n1 and e.target == n2) or (e.source == n2 and e.target == n1)
                for e in edges
            )
            if not edge_found:
                raise RuntimeError("Failed to read back edge after insert")

            int_store.delete_edges_for_node(n1)
            int_store.delete_edges_for_node(n2)
            int_store.delete_node(n1)
            int_store.delete_node(n2)

            integration["data_persistence"] = {
                "status": "pass",
                "details": "Node and edge write/read/delete cycle successful",
            }
        except Exception as e:
            try:
                int_store.delete_edges_for_node(n1)
                int_store.delete_edges_for_node(n2)
                int_store.delete_node(n1)
                int_store.delete_node(n2)
            except Exception:
                pass
            integration["data_persistence"] = {
                "status": "fail",
                "details": str(e),
            }

    checks["integration_tests"] = integration

    # Overall status
    db_ok = (
        checks.get("chromadb_connection", {}).get("status") == "pass"
        and checks.get("sqlite_connection", {}).get("status") == "pass"
    )
    integration_ok = all(t.get("status") == "pass" for t in integration.values())
    llm_ok = checks.get("processing_llm", {}).get("status") == "available"
    ollama_ok = checks.get("ollama_connection", {}).get("status") == "pass"

    if db_ok and integration_ok and llm_ok and ollama_ok:
        status = "healthy"
    elif db_ok and integration_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "checks": checks,
        "timestamp": _time.time(),
        "workspace": workspace_id,
    }
