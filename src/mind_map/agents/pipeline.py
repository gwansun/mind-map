"""LangGraph pipeline for memo ingestion."""

import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph

from mind_map.core.graph_store import GraphStore
from mind_map.models.schemas import Edge, ExtractionResult, FilterDecision, NodeType


@dataclass
class PipelineState:
    """State passed through the LangGraph pipeline."""

    raw_text: str
    source_id: str | None = None
    filter_decision: FilterDecision | None = None
    extraction: ExtractionResult | None = None
    node_ids: list[str] = field(default_factory=list)
    error: str | None = None


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

        # Use LLM for filtering
        from mind_map.agents.filter_agent import FilterAgent
        agent = FilterAgent(llm)
        try:
            decision = agent.evaluate_sync(text)
            return {"filter_decision": decision}
        except Exception as e:
            return {"error": f"Filter agent error: {e}"}

    return filter_node


def create_extraction_node(llm: Any | None = None):
    """Create an extraction node that extracts entities and tags."""

    def extraction_node(state: PipelineState) -> dict[str, Any]:
        if state.filter_decision is None or state.filter_decision.action == "discard":
            return {}

        text = state.filter_decision.summary or state.raw_text

        # Simple heuristic extraction if no LLM available
        if llm is None:
            # Extract hashtags as tags
            import re
            hashtags = re.findall(r"#(\w+)", text)
            tags = [f"#{tag}" for tag in hashtags]

            # Extract capitalized words as potential entities
            words = text.split()
            entities = [
                w for w in words
                if len(w) > 2 and w[0].isupper() and not w.isupper()
            ]
            entities = list(set(entities))[:5]  # Limit to 5

            # If no tags found, generate based on keywords
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
                    relationships=[]
                )
            }

        # Use LLM for extraction
        from mind_map.agents.knowledge_processor import KnowledgeProcessor
        processor = KnowledgeProcessor(llm)
        try:
            result = processor.extract_sync(text)
            return {"extraction": result}
        except Exception as e:
            return {"error": f"Extraction error: {e}"}

    return extraction_node


def create_storage_node(store: GraphStore):
    """Create a storage node that persists to GraphStore."""

    def storage_node(state: PipelineState) -> dict[str, Any]:
        if state.extraction is None:
            return {}

        node_ids = []
        extraction = state.extraction

        # Create main concept node
        concept_id = str(uuid.uuid4())
        store.add_node(
            node_id=concept_id,
            document=extraction.summary,
            node_type=NodeType.CONCEPT,
            source_id=state.source_id,
        )
        node_ids.append(concept_id)

        # Create tag nodes and edges
        for tag in extraction.tags:
            # Check if tag already exists by querying
            tag_id = f"tag_{tag.lower().replace('#', '').replace(' ', '_')}"

            # Try to get existing tag node
            existing = store.get_node(tag_id)
            if existing is None:
                store.add_node(
                    node_id=tag_id,
                    document=tag,
                    node_type=NodeType.TAG,
                )
            node_ids.append(tag_id)

            # Create edge from concept to tag
            store.add_edge(Edge(
                source=concept_id,
                target=tag_id,
                relation_type="tagged_as",
            ))

        # Create entity nodes and relationship edges
        linked_entities: set[str] = set()
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
                if eid not in linked_entities:
                    store.add_edge(Edge(
                        source=concept_id,
                        target=eid,
                        relation_type="mentions",
                    ))
                    linked_entities.add(eid)
                    node_ids.append(eid)

            # Relationship edge between the two entities
            store.add_edge(Edge(
                source=source_eid,
                target=target_eid,
                relation_type=relation,
            ))

        return {"node_ids": node_ids}

    return storage_node


def should_continue(state: PipelineState) -> str:
    """Determine if pipeline should continue after filtering."""
    if state.error:
        return "end"
    if state.filter_decision and state.filter_decision.action == "discard":
        return "end"
    return "extract"


def build_ingestion_pipeline(store: GraphStore, llm: Any | None = None) -> StateGraph:
    """Build the LangGraph pipeline for memo ingestion.

    Args:
        store: GraphStore instance for persistence
        llm: Optional LangChain LLM for intelligent processing

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("filter", create_filter_node(llm))
    workflow.add_node("extract", create_extraction_node(llm))
    workflow.add_node("store", create_storage_node(store))

    # Set entry point
    workflow.set_entry_point("filter")

    # Add conditional edge after filter
    workflow.add_conditional_edges(
        "filter",
        should_continue,
        {
            "extract": "extract",
            "end": END,
        }
    )

    # Linear flow: extract -> store -> end
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
