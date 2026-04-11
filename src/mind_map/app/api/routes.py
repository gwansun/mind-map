"""FastAPI routes for Mind Map API."""

from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mind_map.core.config import get_data_dir
from mind_map.rag.graph_store import GraphStore

load_dotenv()

app = FastAPI(
    title="Mind Map API",
    description="Knowledge Graph-based Mind Map with RAG-enhanced responses",
    version="0.1.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """Request body for the ask endpoint."""

    query: str
    depth: int = 2


class AskResponse(BaseModel):
    """Response body for the ask endpoint."""

    query: str
    response: str
    context_nodes: list[str]


class GraphResponse(BaseModel):
    """Response body for graph data."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class MemoRequest(BaseModel):
    """Request body for the memo endpoint."""

    text: str
    source: str | None = None


def get_store() -> GraphStore:
    """Get or initialize the graph store."""
    data_dir = get_data_dir()
    store = GraphStore(data_dir)
    if not data_dir.exists():
        store.initialize()
    return store


@app.get("/")
async def root() -> dict[str, str]:
    """API root endpoint."""
    return {"message": "Mind Map API", "version": "0.1.0"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint with LLM status and graph statistics."""
    from mind_map.rag.llm_status import get_llm_status

    store = get_store()
    stats = store.get_stats()
    llm_status = get_llm_status()

    return {
        "status": "healthy",
        "llm_status": llm_status,
        "graph_stats": {
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"]
        }
    }


@app.get("/graph", response_model=GraphResponse)
async def get_graph() -> GraphResponse:
    """Fetch the full knowledge graph for visualization."""
    store = get_store()
    stats = store.get_stats()

    if stats["total_nodes"] == 0:
        return GraphResponse(nodes=[], edges=[])

    # Get all nodes
    result = store.collection.get(include=["documents", "metadatas"])
    nodes = []
    for i, node_id in enumerate(result["ids"]):
        nodes.append({
            "id": node_id,
            "document": result["documents"][i] if result["documents"] else "",
            "metadata": result["metadatas"][i] if result["metadatas"] else {},
        })

    # Get all edges
    cursor = store.sqlite.execute(
        "SELECT source, target, weight, relation_type FROM edges"
    )
    edges = [
        {"source": row[0], "target": row[1], "weight": row[2], "relation_type": row[3]}
        for row in cursor.fetchall()
    ]

    return GraphResponse(nodes=nodes, edges=edges)


@app.get("/node/{node_id:path}")
async def get_node(node_id: str) -> dict[str, Any]:
    """Get detailed view of a specific node."""
    store = get_store()
    node = store.get_node(node_id)

    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    edges = store.get_edges(node_id)
    importance = store.calculate_importance(node_id)

    return {
        "node": node.model_dump(),
        "edges": [e.model_dump() for e in edges],
        "importance_score": importance,
    }


class DeleteNodeResponse(BaseModel):
    """Response body for node deletion.

    For concept deletes, also returns the list of tag node IDs that were
    cascade-deleted. For non-concept deletes, deleted_tag_ids is empty.
    """

    node_id: str
    deleted_tag_ids: list[str] = []
    deleted_edges_count: int


@app.delete("/node/{node_id:path}", response_model=DeleteNodeResponse)
async def delete_node(node_id: str) -> DeleteNodeResponse:
    """Delete a node and all its connected edges.

    For concept nodes: also deletes all directly-connected first-layer neighbor
    tags, even if those tags are shared with other nodes. All edges attached to
    deleted tags are removed as well.

    For non-concept nodes (tag/entity): deletes only the node and its own edges,
    preserving existing behavior.

    Raises 404 if the primary node is not found.
    """
    store = get_store()
    node = store.get_node(node_id)

    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    result = store.delete_node(node_id)

    return DeleteNodeResponse(
        node_id=result.deleted_node_id,
        deleted_tag_ids=result.deleted_tag_ids,
        deleted_edges_count=result.deleted_edges_count,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Query the knowledge graph with RAG-enhanced response.

    Flow:
    1. Search for relevant context nodes in the knowledge graph
    2. Generate response using reasoning LLM (with or without context)
    3. Process Q&A pair through LLM-B pipeline to extract and store knowledge
    4. Link new nodes to context nodes if any existed
    """
    from mind_map.app.pipeline import ingest_memo
    from mind_map.core.schemas import Edge
    from mind_map.processor.processing_llm import get_processing_llm
    from mind_map.rag.reasoning_llm import get_reasoning_llm
    from mind_map.rag.response_generator import ResponseGenerator

    store = get_store()

    # Step 1: Retrieve similar nodes (may be empty for new topics)
    context_nodes = store.query_similar(request.query, n_results=5)

    # Enrich with relation factors and re-sort by combined score
    context_nodes = store.enrich_context_nodes(context_nodes)

    context_node_ids = [node.id for node in context_nodes]

    # Step 2: Get reasoning LLM for response generation
    llm = get_reasoning_llm()
    if not llm:
        # Cannot generate response without reasoning LLM
        if context_nodes:
            context_text = "\n\n".join([node.document for node in context_nodes])
            return AskResponse(
                query=request.query,
                response=f"Reasoning LLM not configured. Raw context:\n\n{context_text}",
                context_nodes=context_node_ids,
            )
        return AskResponse(
            query=request.query,
            response="Reasoning LLM not configured. Please set up Claude CLI or configure an API key.",
            context_nodes=[],
        )

    # Step 3: Generate response using ResponseGenerator
    # Pass context_nodes (may be empty list for new topics)
    generator = ResponseGenerator(llm)
    response = await generator.generate(request.query, context_nodes)

    # Update importance scores for context nodes that were used
    for node in context_nodes:
        store.update_interaction(node.id)

    # Step 4: Process Q&A pair through LLM(B) pipeline and add to knowledge graph
    # This happens regardless of whether context existed - grows the KG
    qa_text = f"Q: {request.query}\nA: {response}"

    processing_llm = get_processing_llm()
    success, message, qa_node_ids = ingest_memo(
        text=qa_text,
        store=store,
        llm=processing_llm,
        source_id=f"qa_{request.query[:50]}",
    )

    # Step 5: Link Q&A nodes to context nodes if any existed
    if success and qa_node_ids and context_nodes:
        qa_concept_id = qa_node_ids[0]
        for context_node in context_nodes:
            edge = Edge(
                source=qa_concept_id,
                target=context_node.id,
                weight=1.0,
                relation_type="derived_from",
            )
            store.add_edge(edge)

    return AskResponse(
        query=request.query,
        response=response,
        context_nodes=context_node_ids,
    )


@app.post("/memo")
async def add_memo(request: MemoRequest) -> dict[str, Any]:
    """Ingest a memo into the knowledge graph via the LangGraph pipeline."""
    from mind_map.app.pipeline import ingest_memo
    from mind_map.processor.processing_llm import get_processing_llm

    store = get_store()
    store.initialize()

    processing_llm = get_processing_llm()

    success, message, node_ids = ingest_memo(
        text=request.text,
        store=store,
        llm=processing_llm,
        source_id=request.source,
    )

    return {
        "status": "success" if success else "skipped",
        "message": message,
        "node_ids": node_ids,
    }


@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Get knowledge graph statistics."""
    store = get_store()
    return store.get_stats()
