"""LangGraph pipeline for memo ingestion with retrieval-first duplicate detection."""

import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph

from mind_map.core.schemas import (
    Edge,
    ExtractionResult,
    FilterDecision,
    GraphNode,
    NodeType,
    RetrievalContext,
)
from mind_map.rag.graph_store import GraphStore


@dataclass
class PipelineState:
    """State passed through the LangGraph pipeline."""

    raw_text: str
    source_id: str | None = None
    retrieval: RetrievalContext | None = None
    filter_decision: FilterDecision | None = None
    extraction: ExtractionResult | None = None
    node_ids: list[str] = field(default_factory=list)
    error: str | None = None


_RETRIEVAL_TOP_K = 5


def _heuristic_filter(text: str, retrieved_concepts: list[GraphNode]) -> FilterDecision:
    """Simple non-LLM novelty filter."""
    trivial_patterns = [
        "hello", "hi", "thanks", "thank you", "ok", "okay",
        "yes", "no", "sure", "bye", "goodbye"
    ]
    lower_text = text.lower().strip()

    if len(text.strip()) < 10:
        return FilterDecision(action="discard", reason="Text too short (< 10 characters)", summary=None)

    if lower_text in trivial_patterns:
        return FilterDecision(action="discard", reason="Trivial/greeting message", summary=None)

    normalized = " ".join(text.lower().split())
    for concept in retrieved_concepts:
        if normalized and normalized == " ".join(concept.document.lower().split()):
            return FilterDecision(
                action="duplicate",
                reason=f"Matches existing concept {concept.id}",
                summary=None,
            )

    return FilterDecision(action="new", reason="Content appears substantive and new", summary=text)


def create_retrieval_node(store: GraphStore):
    """Retrieve similar concepts and their first-hop entity/tag neighbors."""

    def retrieval_node(state: PipelineState) -> dict[str, Any]:
        text = state.raw_text.strip()
        concepts = [
            node for node in store.query_similar(text, n_results=_RETRIEVAL_TOP_K, max_distance=0.5)
            if node.metadata.type == NodeType.CONCEPT
        ]
        neighbors = store.get_first_hop_neighbors([node.id for node in concepts])
        entities = [node for node in neighbors if node.metadata.type == NodeType.ENTITY]
        tags = [node for node in neighbors if node.metadata.type == NodeType.TAG]
        return {"retrieval": RetrievalContext(concepts=concepts, entities=entities, tags=tags)}

    return retrieval_node


def create_filter_node(filter_llm: Any | None = None):
    """Create a filter node that decides discard/duplicate/new.

    Filter evaluation order:
        1. phi3.5 via FilterAgentWithFallback (LangChain) — if filter_llm provided
        2. OpenClaw MiniMax CLI — via FilterAgentWithFallback fallback
        3. Heuristic filter (final fallback)
    """
    from mind_map.processor.filter_agent import FilterAgentWithFallback

    agent_with_fallback = FilterAgentWithFallback(filter_llm) if filter_llm else None

    def filter_node(state: PipelineState) -> dict[str, Any]:
        text = state.raw_text.strip()
        retrieved_concepts = state.retrieval.concepts if state.retrieval else []

        # Always check heuristic first — discard/trivial always returns discard
        heuristic_decision = _heuristic_filter(text, retrieved_concepts)
        if heuristic_decision.action == "discard":
            return {"filter_decision": heuristic_decision}

        # Try LLM chain: phi3.5 -> MiniMax CLI -> heuristic
        if agent_with_fallback is not None:
            try:
                decision = agent_with_fallback.evaluate_sync(text, retrieved_concepts)
                return {"filter_decision": decision}
            except RuntimeError:
                # Both phi3.5 and MiniMax failed — fall through to heuristic
                pass

        return {"filter_decision": heuristic_decision}

    return filter_node


def create_extraction_node(llm: Any | None = None):
    """Create an extraction node that extracts only for new memos."""

    def extraction_node(state: PipelineState) -> dict[str, Any]:
        if state.filter_decision is None or state.filter_decision.action != "new":
            return {}

        text = state.filter_decision.summary or state.raw_text
        references = []
        if state.retrieval is not None:
            references = [*state.retrieval.entities, *state.retrieval.tags]

        if llm is not None:
            from mind_map.processor.knowledge_processor import KnowledgeProcessor
            processor = KnowledgeProcessor(llm)
            try:
                result = processor.extract_with_references(text, references)
                return {"extraction": result}
            except Exception:
                pass

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
        )
    }


def create_storage_node(store: GraphStore):
    """Persist new memo extraction results to GraphStore."""

    def storage_node(state: PipelineState) -> dict[str, Any]:
        if state.extraction is None or state.filter_decision is None or state.filter_decision.action != "new":
            return {}

        node_ids: list[str] = []
        extraction = state.extraction

        concept_id = str(uuid.uuid4())
        store.add_node(
            node_id=concept_id,
            document=extraction.summary,
            node_type=NodeType.CONCEPT,
            source_id=state.source_id,
        )
        node_ids.append(concept_id)

        for tag in extraction.tags:
            tag_id = f"tag_{tag.lower().replace('#', '').replace(' ', '_')}"
            if store.get_node(tag_id) is None:
                store.add_node(
                    node_id=tag_id,
                    document=tag,
                    node_type=NodeType.TAG,
                )
            node_ids.append(tag_id)
            store.add_edge(Edge(source=concept_id, target=tag_id, relation_type="tagged_as"))

        linked_entity_ids: set[str] = set()

        for source_name, relation, target_name in extraction.relationships:
            source_eid = f"entity_{source_name.lower().replace(' ', '_')}"
            target_eid = f"entity_{target_name.lower().replace(' ', '_')}"

            for eid, ename in [(source_eid, source_name), (target_eid, target_name)]:
                if store.get_node(eid) is None:
                    store.add_node(node_id=eid, document=ename, node_type=NodeType.ENTITY)
                    node_ids.append(eid)
                if eid not in linked_entity_ids:
                    store.add_edge(Edge(source=concept_id, target=eid, relation_type="mentions"))
                    linked_entity_ids.add(eid)

            store.add_edge(Edge(source=source_eid, target=target_eid, relation_type=relation))

        relationship_entity_ids: set[str] = set()
        for source_name, _, target_name in extraction.relationships:
            relationship_entity_ids.add(f"entity_{source_name.lower().replace(' ', '_')}")
            relationship_entity_ids.add(f"entity_{target_name.lower().replace(' ', '_')}")

        for entity_name in extraction.entities:
            eid = f"entity_{entity_name.lower().replace(' ', '_')}"
            if eid in relationship_entity_ids or eid in linked_entity_ids:
                continue
            if store.get_node(eid) is None:
                store.add_node(node_id=eid, document=entity_name, node_type=NodeType.ENTITY)
                node_ids.append(eid)
            store.add_edge(Edge(source=concept_id, target=eid, relation_type="mentions"))
            linked_entity_ids.add(eid)

        if state.retrieval is not None:
            for concept in state.retrieval.concepts:
                store.add_edge(Edge(source=concept_id, target=concept.id, relation_type="related_context"))

        return {"node_ids": node_ids}

    return storage_node


def should_continue_after_filter(state: PipelineState) -> str:
    """Determine whether to continue to extraction or end."""
    if state.error:
        return "end"
    if state.filter_decision is None:
        return "end"
    if state.filter_decision.action == "new":
        return "extract"
    return "end"


def build_ingestion_pipeline(
    store: GraphStore,
    llm: Any | None = None,
    filter_llm: Any | None = None,
) -> StateGraph:
    """Build the LangGraph pipeline for memo ingestion.

    Pipeline flow:
        retrieve -> filter -> extract -> store -> end

    Args:
        store: GraphStore instance for retrieval and storage
        llm: General processing LLM (used for extraction, not filter)
        filter_llm: Ollama LLM for filter step (phi3.5 via LangChain).
                    If None, filter falls back to MiniMax CLI then heuristic.
    """
    workflow = StateGraph(PipelineState)

    workflow.add_node("retrieve", create_retrieval_node(store))
    workflow.add_node("filter", create_filter_node(filter_llm))
    workflow.add_node("extract", create_extraction_node(llm))
    workflow.add_node("store", create_storage_node(store))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "filter")
    workflow.add_conditional_edges(
        "filter",
        should_continue_after_filter,
        {
            "extract": "extract",
            "end": END,
        }
    )
    workflow.add_edge("extract", "store")
    workflow.add_edge("store", END)

    return workflow.compile()


def ingest_memo(
    text: str,
    store: GraphStore,
    llm: Any | None = None,
    source_id: str | None = None,
    filter_llm: Any | None = None,
) -> tuple[bool, str, list[str]]:
    """Ingest a memo through the pipeline.

    Args:
        text: Raw memo text to process
        store: GraphStore instance
        llm: General processing LLM for extraction step
        source_id: Optional source identifier for the memo
        filter_llm: Ollama LLM for filter step (phi3.5). If None,
                     uses MiniMax CLI fallback then heuristic.
    """
    pipeline = build_ingestion_pipeline(store, llm, filter_llm=filter_llm)

    initial_state = PipelineState(raw_text=text, source_id=source_id)
    final_state = pipeline.invoke(initial_state)

    if final_state.get("error"):
        return False, final_state["error"], []

    filter_decision = final_state.get("filter_decision")
    if filter_decision:
        if filter_decision.action == "discard":
            return False, f"Discarded: {filter_decision.reason}", []
        if filter_decision.action == "duplicate":
            return False, f"Skipped duplicate: {filter_decision.reason}", []

    node_ids = final_state.get("node_ids", [])
    return True, f"Created {len(node_ids)} nodes", node_ids
