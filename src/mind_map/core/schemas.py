"""Pydantic schemas for Mind Map data structures."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    """Type of node in the knowledge graph."""

    CONCEPT = "concept"
    TAG = "tag"
    ENTITY = "entity"


class NodeMetadata(BaseModel):
    """Metadata stored with each graph node in ChromaDB."""

    type: NodeType = Field(description="Node type: concept or tag")
    created_at: float = Field(description="Unix timestamp of creation")
    last_interaction: float = Field(description="Unix timestamp of last access/update")
    connection_count: int = Field(default=0, description="Number of edges connected to this node")
    importance_score: float = Field(default=0.0, description="Cached importance score S")
    original_source_id: str | None = Field(
        default=None, description="Reference to raw interaction source"
    )


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(description="Unique identifier (UUID)")
    document: str = Field(description="Summarized text or tag name")
    metadata: NodeMetadata = Field(description="Graph attributes and metrics")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    relation_factor: float | None = Field(
        default=None, description="Context-dependent edge density score (computed at query time)"
    )


class Edge(BaseModel):
    """An edge/relationship between two nodes."""

    source: str = Field(description="Source node UUID")
    target: str = Field(description="Target node UUID")
    weight: float = Field(default=1.0, description="Edge weight")
    relation_type: str = Field(
        default="related_context", description="Type: tagged_as, related_context, etc."
    )


class FilterDecision(BaseModel):
    """LLM(B) filter agent decision on incoming data."""

    action: Literal["keep", "discard"] = Field(description="Whether to keep or discard the data")
    reason: str = Field(description="Explanation for the decision")
    summary: str | None = Field(
        default=None, description="Summarized content if action is keep"
    )


class ExistingLink(BaseModel):
    """A link from a new concept to an existing graph node, drawn from retrieval context."""

    target_id: str = Field(
        description="ID of an existing graph node (must come from the retrieval context provided during extraction)"
    )
    relation_type: str = Field(
        default="related_context",
        description="Type of relation to the existing node",
    )


class ExtractionResult(BaseModel):
    """Result from knowledge extraction."""

    summary: str = Field(description="Summarized content")
    tags: list[str] = Field(default_factory=list, description="Extracted tags")
    entities: list[str] = Field(default_factory=list, description="Extracted entities")
    relationships: list[tuple[str, str, str]] = Field(
        default_factory=list, description="(source, relation, target) tuples"
    )
    existing_links: list[ExistingLink] = Field(
        default_factory=list,
        description="Links from this concept to existing retrieved nodes (IDs must come from retrieval context)",
    )

    @field_validator("relationships", mode="before")
    @classmethod
    def filter_relationships(cls, v: list) -> list:
        """Drop any relationship the LLM returned that isn't exactly [source, relation, target]."""
        if not isinstance(v, list):
            return []
        return [r for r in v if isinstance(r, (list, tuple)) and len(r) == 3]


class QueryResult(BaseModel):
    """Result from a knowledge graph query."""

    nodes: list[GraphNode] = Field(default_factory=list, description="Retrieved nodes")
    context: str = Field(description="Synthesized context from nodes")
    response: str | None = Field(default=None, description="LLM-generated response")


class DeleteNodeResult(BaseModel):
    """Result of a delete operation on a node.

    For concept deletes: also deletes first-layer neighbor tags and their edges.
    For non-concept deletes: only deletes the node and its own edges.
    """

    deleted_node_id: str = Field(description="ID of the primary deleted node")
    deleted_tag_ids: list[str] = Field(
        default_factory=list,
        description="IDs of tag nodes that were also deleted (concept deletes only)",
    )
    deleted_edges_count: int = Field(description="Total number of edges deleted across all nodes")
