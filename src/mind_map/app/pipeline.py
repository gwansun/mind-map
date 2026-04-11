"""LangGraph pipeline for memo ingestion with retrieval-augmented extraction."""

import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph

from mind_map.core.schemas import (
    Edge,
    ExtractionResult,
    ExistingLink,
    FilterDecision,
    GraphNode,
    NodeType,
)
from mind_map.rag.graph_store import GraphStore


@dataclass
class PipelineState:
    """State passed through the LangGraph pipeline."""

    raw_text: str
    source_id: str | None = None
    filter_decision: FilterDecision | None = None
    retrieved_nodes: list[GraphNode] = field(default_factory=list)
    extraction: ExtractionResult | None = None
    node_ids: list[str] = field(default_factory=list)
    error: str | None = None


# How many existing nodes to retrieve for grounding extraction
_RETRIEVAL_TOP_K = 5


def create_filter_node(llm: Any | None = None):
    """Create a filter node that decides keep/discard."""

    def filter_node(state: PipelineState) -> dict[str, Any]:
        text = state.raw_text.strip()

        # Simple heuristic filtering if no LLM available
        if llm is None:
            # Filter out very short or trivial messages
            trivial_patterns = [
                "hello", "hi", "thanks", "thank you", "ok", "okay",
                "yes", "no", "sure", "bye", "goodbye"
            ]
            lower_text = text.lower()

            if len(text) < 10:
                return {
                    "filter_decision": FilterDecision(
                        action="discard",
                        reason="Text too short (< 10 characters)",
                        summary=None
                    )
                }

            if lower_text in trivial_patterns:
                return {
                    "filter_decision": FilterDecision(
                        action="discard",
                        reason="Trivial/greeting message",
                        summary=None
                    )
                }

            return {
                "filter_decision": FilterDecision(
                    action="keep",
                    reason="Content appears substantive",
                    summary=text
                )
            }

        # Use LLM for filtering, fall back to heuristic on failure
        from mind_map.processor.filter_agent import FilterAgent
        agent = FilterAgent(llm)
        try:
            decision = agent.evaluate_sync(text)
            return {"filter_decision": decision}
        except Exception:
            # LLM failed — fall back to heuristic (keep substantive content)
            return {
                "filter_decision": FilterDecision(
                    action="keep",
                    reason="LLM filter unavailable, kept by heuristic fallback",
                    summary=text,
                )
            }

    return filter_node


def create_retrieval_node(store: GraphStore):
    """Create a retrieval node that finds relevant existing nodes before extraction.

    This enables the extraction model to form grounded links to existing records.
    """

    def retrieval_node(state: PipelineState) -> dict[str, Any]:
        if state.filter_decision is None or state.filter_decision.action == "discard":
            return {}

        text = state.filter_decision.summary or state.raw_text

        # Query the graph for similar nodes, keeping the threshold conservative
        retrieved = store.query_similar(text, n_results=_RETRIEVAL_TOP_K, max_distance=0.5)

        return {"retrieved_nodes": retrieved}

    return retrieval_node


def create_extraction_node(llm: Any | None = None):
    """Create an extraction node that extracts entities, tags, and existing links.

    Uses retrieval context from the pipeline state when available.
    """

    def extraction_node(state: PipelineState) -> dict[str, Any]:
        if state.filter_decision is None or state.filter_decision.action == "discard":
            return {}

        text = state.filter_decision.summary or state.raw_text
        retrieved_nodes = state.retrieved_nodes

        # Use KnowledgeProcessor with retrieval context
        if llm is not None:
            from mind_map.processor.knowledge_processor import KnowledgeProcessor
            processor = KnowledgeProcessor(llm)
            try:
                result = processor.extract_with_context(text, retrieved_nodes)
                return {"extraction": result}
            except Exception:
                pass

        # Fall back to heuristic extraction
        return _heuristic_extraction(text)

    return extraction_node


def _heuristic_extraction(text: str) -> dict[str, Any]:
    """Simple heuristic extraction as fallback."""
    import re

    hashtags = re.findall(r"#(\w+)", text)
    tags = [f"#{tag}" for tag in hashtags]

    words = text.split()
    entities = [
        w for w in words
        if len(w) > 2 and w[0].isupper() and not w.isupper()
    ]
    entities = list(set(entities))[:5]

    if not tags:
        keywords = ["python", "api", "database", "test", "config", "auth"]
        for kw in keywords:
            if kw in text.lower():
                tags.append(f"#{kw.capitalize()}")

    return {
        "extraction": ExtractionResult(
            summary=text[:200] if len(text) > 200 else text,
            tags=tags[:5],
            entities=entities,
            relationships=[],
            existing_links=[],
        )
    }


def create_storage_node(store: GraphStore):
    """Create a storage node that persists to GraphStore.

    Handles:
    - Main concept node creation
    - Tag nodes and tagged_as edges
    - Entity nodes and mentions edges (including standalone entities)
    - Relationship edges between entities
    - Existing link edges to retrieved nodes (with ID validation)
    - Safe link deduplication
    """

    def storage_node(state: PipelineState) -> dict[str, Any]:
        if state.extraction is None:
            return {}

        node_ids: list[str] = []
        extraction = state.extraction

        # Build a set of valid retrieved node IDs for filtering existing_links
        retrieved_ids: set[str] = {n.id for n in state.retrieved_nodes}

        # ---- Main concept node ----
        concept_id = str(uuid.uuid4())
        store.add_node(
            node_id=concept_id,
            document=extraction.summary,
            node_type=NodeType.CONCEPT,
            source_id=state.source_id,
        )
        node_ids.append(concept_id)

        # ---- Tag nodes and edges ----
        for tag in extraction.tags:
            tag_id = f"tag_{tag.lower().replace('#', '').replace(' ', '_')}"

            if store.get_node(tag_id) is None:
                store.add_node(
                    node_id=tag_id,
                    document=tag,
                    node_type=NodeType.TAG,
                )
            node_ids.append(tag_id)

            store.add_edge(Edge(
                source=concept_id,
                target=tag_id,
                relation_type="tagged_as",
            ))

        # ---- Entity nodes and edges ----
        # Track which entities we've already linked to avoid duplicate mentions edges
        linked_entity_ids: set[str] = set()

        for source_name, relation, target_name in extraction.relationships:
            source_eid = f"entity_{source_name.lower().replace(' ', '_')}"
            target_eid = f"entity_{target_name.lower().replace(' ', '_')}"

            for eid, ename in [(source_eid, source_name), (target_eid, target_name)]:
                if store.get_node(eid) is None:
                    store.add_node(
                        node_id=eid,
                        document=ename,
                        node_type=NodeType.ENTITY,
                    )
                    node_ids.append(eid)
                else:
                    # Entity already exists — still track it for mentions edge
                    pass

                if eid not in linked_entity_ids:
                    store.add_edge(Edge(
                        source=concept_id,
                        target=eid,
                        relation_type="mentions",
                    ))
                    linked_entity_ids.add(eid)

            # Relationship edge between the two entities
            store.add_edge(Edge(
                source=source_eid,
                target=target_eid,
                relation_type=relation,
            ))

        # ---- Standalone entities (not in any relationship) ----
        # Persist them even if they don't appear in relationships
        relationship_entity_ids: set[str] = set()
        for source_name, relation, target_name in extraction.relationships:
            relationship_entity_ids.add(
                f"entity_{source_name.lower().replace(' ', '_')}"
            )
            relationship_entity_ids.add(
                f"entity_{target_name.lower().replace(' ', '_')}"
            )

        for entity_name in extraction.entities:
            eid = f"entity_{entity_name.lower().replace(' ', '_')}"
            if eid in relationship_entity_ids:
                continue  # Already handled above
            if eid in linked_entity_ids:
                continue  # Already linked via a previous iteration

            if store.get_node(eid) is None:
                store.add_node(
                    node_id=eid,
                    document=entity_name,
                    node_type=NodeType.ENTITY,
                )
                node_ids.append(eid)

            store.add_edge(Edge(
                source=concept_id,
                target=eid,
                relation_type="mentions",
            ))
            linked_entity_ids.add(eid)

        # ---- Existing link edges (to retrieved nodes) ----
        # Only persist links to IDs that were actually in the retrieval context
        seen_existing_links: set[tuple[str, str, str]] = set()
        for link in extraction.existing_links:
            if link.target_id not in retrieved_ids:
                # ID not in retrieval context — skip to prevent hallucinated links
                continue

            relation_type = link.relation_type or "related_context"
            key = (concept_id, link.target_id, relation_type)
            if key in seen_existing_links:
                continue

            store.add_edge(Edge(
                source=concept_id,
                target=link.target_id,
                relation_type=relation_type,
            ))
            seen_existing_links.add(key)

        return {"node_ids": node_ids}

    return storage_node


def should_continue(state: PipelineState) -> str:
    """Determine if pipeline should continue after filtering."""
    if state.error:
        return "end"
    if state.filter_decision and state.filter_decision.action == "discard":
        return "end"
    return "retrieve"


def should_extract(state: PipelineState) -> str:
    """Decide whether to proceed to extraction or end."""
    if state.error:
        return "end"
    return "extract"


def build_ingestion_pipeline(
    store: GraphStore,
    llm: Any | None = None,
) -> StateGraph:
    """Build the LangGraph pipeline for memo ingestion.

    Pipeline flow:
        filter -> retrieve -> extract -> store -> end

    Args:
        store: GraphStore instance for persistence
        llm: Optional LangChain LLM for intelligent processing

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("filter", create_filter_node(llm))
    workflow.add_node("retrieve", create_retrieval_node(store))
    workflow.add_node("extract", create_extraction_node(llm))
    workflow.add_node("store", create_storage_node(store))

    # Set entry point
    workflow.set_entry_point("filter")

    # Conditional after filter
    workflow.add_conditional_edges(
        "filter",
        should_continue,
        {
            "retrieve": "retrieve",
            "end": END,
        }
    )

    # retrieve -> extract
    workflow.add_conditional_edges(
        "retrieve",
        should_extract,
        {
            "extract": "extract",
            "end": END,
        }
    )

    # extract -> store -> end
    workflow.add_edge("extract", "store")
    workflow.add_edge("store", END)

    return workflow.compile()


def ingest_memo(
    text: str,
    store: GraphStore,
    llm: Any | None = None,
    source_id: str | None = None,
) -> tuple[bool, str, list[str]]:
    """Ingest a memo through the pipeline.

    Args:
        text: Raw text to ingest
        store: GraphStore instance
        llm: Optional LangChain LLM
        source_id: Optional source identifier

    Returns:
        Tuple of (success, message, node_ids)
    """
    pipeline = build_ingestion_pipeline(store, llm)

    initial_state = PipelineState(
        raw_text=text,
        source_id=source_id,
    )

    # Run the pipeline
    final_state = pipeline.invoke(initial_state)

    if final_state.get("error"):
        return False, final_state["error"], []

    filter_decision = final_state.get("filter_decision")
    if filter_decision and filter_decision.action == "discard":
        return False, f"Discarded: {filter_decision.reason}", []

    node_ids = final_state.get("node_ids", [])
    return True, f"Created {len(node_ids)} nodes", node_ids
