"""Response Generator - LLM(A) for synthesizing answers from retrieved context."""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from mind_map.models.schemas import GraphNode

# Prompt for when we have context from the knowledge graph
RESPONSE_WITH_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent assistant with access to a knowledge graph.
Use the provided context to answer the user's question accurately and helpfully.

Context from knowledge graph:
{context}

Guidelines:
- Base your answer primarily on the provided context
- You may supplement with your general knowledge if the context is incomplete
- Be concise but thorough
- Reference specific pieces of context when relevant"""),
    ("human", "{query}"),
])

# Prompt for when we have no context (new topic for the knowledge graph)
RESPONSE_NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent assistant helping to build a knowledge graph.
The user is asking about a topic that is new to the knowledge graph.

Answer the user's question using your knowledge. Your response will be processed
and added to the knowledge graph for future reference.

Guidelines:
- Provide a helpful, accurate answer
- Be concise but thorough
- Structure your response clearly"""),
    ("human", "{query}"),
])


class ResponseGenerator:
    """Agent that generates responses using retrieved knowledge graph context."""

    def __init__(self, llm: Any) -> None:
        """Initialize with a LangChain-compatible LLM.

        Args:
            llm: LangChain LLM instance (e.g., ChatOpenAI, ChatAnthropic)
        """
        self.llm = llm
        self.chain_with_context = RESPONSE_WITH_CONTEXT_PROMPT | llm
        self.chain_no_context = RESPONSE_NO_CONTEXT_PROMPT | llm

    def _format_context(self, nodes: list[GraphNode]) -> str:
        """Format retrieved nodes into context string.

        Args:
            nodes: List of GraphNode objects

        Returns:
            Formatted context string
        """
        if not nodes:
            return ""

        context_parts = []
        for i, node in enumerate(nodes, 1):
            importance = node.metadata.importance_score
            node_type = node.metadata.type.value
            context_parts.append(
                f"[{i}] ({node_type}, importance: {importance:.2f})\n{node.document}"
            )

        return "\n\n".join(context_parts)

    async def generate(self, query: str, nodes: list[GraphNode]) -> str:
        """Generate a response based on query and retrieved nodes.

        Args:
            query: User's question
            nodes: Retrieved nodes from knowledge graph (may be empty)

        Returns:
            Generated response string
        """
        if nodes:
            # Use RAG with context
            context = self._format_context(nodes)
            result = await self.chain_with_context.ainvoke({"context": context, "query": query})
        else:
            # No context - answer directly (new topic for KG)
            result = await self.chain_no_context.ainvoke({"query": query})
        return result.content

    def generate_sync(self, query: str, nodes: list[GraphNode]) -> str:
        """Synchronous version of generate.

        Args:
            query: User's question
            nodes: Retrieved nodes from knowledge graph (may be empty)

        Returns:
            Generated response string
        """
        if nodes:
            # Use RAG with context
            context = self._format_context(nodes)
            result = self.chain_with_context.invoke({"context": context, "query": query})
        else:
            # No context - answer directly (new topic for KG)
            result = self.chain_no_context.invoke({"query": query})
        return result.content
