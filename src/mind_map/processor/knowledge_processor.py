"""Knowledge Processor - LLM(B) for entity extraction and summarization.

Supports a primary/fallback extraction chain:
1. Primary: OpenClaw MiniMax agent subprocess for structured extraction with retrieval context
2. Fallback: Ollama phi-3.5 via LangChain
3. Final fallback: heuristic extraction

The primary extractor receives both the new memo text AND relevant existing nodes
from the graph, enabling it to form grounded links to existing records.
"""

import json
import logging
import shutil
import subprocess
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mind_map.core.schemas import ExtractionResult, ExistingLink, GraphNode

logger = logging.getLogger(__name__)

_OPENCLAW_TIMEOUT_SECONDS = 40.0

# Base extraction prompt (no retrieval context)
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledge extraction system for a knowledge graph.
Your job is to analyze text and extract structured information.

Extract:
1. **Summary**: A concise summary of the key information (1-2 sentences)
2. **Tags**: Relevant topic tags (e.g., #Python, #Authentication, #Database)
3. **Entities**: Named entities, concepts, or technical terms mentioned
4. **Relationships**: Connections between entities as (source, relation, target) tuples

Respond ONLY with a JSON object in this exact format:
{{
  "summary": "concise summary",
  "tags": ["#Tag1", "#Tag2"],
  "entities": ["Entity1", "Entity2"],
  "relationships": [["Entity1", "uses", "Entity2"], ["Entity2", "part_of", "Entity3"]],
  "existing_links": []
}}"""),
    ("human", "Extract knowledge from this text:\n\n{text}"),
])

# Retrieval-augmented extraction prompt - includes existing nodes from the graph
_RETRIEVAL_CONTEXT_TEMPLATE = """You are a knowledge extraction system for a knowledge graph.
Your job is to analyze new text AND relate it to existing knowledge in the graph.

**Existing graph nodes (for grounding new links):**
{existing_context}

**New text to extract:**
{text}

Extract from the new text:
1. **Summary**: A concise summary of the key information (1-2 sentences)
2. **Tags**: Relevant topic tags (e.g., #Python, #Authentication, #Database)
3. **Entities**: Named entities, concepts, or technical terms mentioned
4. **Relationships**: Connections between entities as (source, relation, target) tuples
5. **Existing Links**: If the new text mentions or relates to the existing nodes above,
   create links using their exact IDs. Use relation types like "extends",
   "contradicts", "references", "related_to", "uses", "opposite_of".
   Only use IDs that appear in the existing context above.

Respond ONLY with a JSON object in this exact format:
{{
  "summary": "concise summary",
  "tags": ["#Tag1", "#Tag2"],
  "entities": ["Entity1", "Entity2"],
  "relationships": [["Entity1", "uses", "Entity2"]],
  "existing_links": [
    {{"target_id": "<id from existing context>", "relation_type": "extends"}}
  ]
}}

Only include existing_links entries where the target_id is from the provided context above.
If no connections to existing nodes are found, use an empty existing_links array."""


def _build_retrieval_context(retrieved_nodes: list[GraphNode]) -> str:
    """Build a human-readable context string from retrieved nodes."""
    if not retrieved_nodes:
        return "(No existing nodes found)"

    lines = []
    for node in retrieved_nodes:
        type_label = node.metadata.type.value
        lines.append(f"- [{type_label}] ID={node.id}: {node.document[:200]}")
    return "\n".join(lines)


def _extract_json_from_text(content: str) -> dict[str, Any] | None:
    """Extract a JSON object from text, tolerating wrapper prose."""
    import re

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

    return None


def _call_openclaw_minimax_extraction(
    prompt: str,
    timeout: float = _OPENCLAW_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Call OpenClaw MiniMax agent for structured extraction.

    This uses the local OpenClaw CLI as the primary extraction path so the
    memo-ingestion stack matches the requested architecture.
    """
    if shutil.which("openclaw") is None:
        logger.debug("openclaw CLI not found, skipping OpenClaw MiniMax extraction")
        return None

    command = [
        "openclaw",
        "agent",
        "--agents",
        "minimax",
        "--message",
        prompt,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("OpenClaw MiniMax extraction failed: %s", result.stderr[:200])
            return None

        content = (result.stdout or "").strip()
        if not content:
            logger.warning("OpenClaw MiniMax returned empty content")
            return None

        return _extract_json_from_text(content)
    except subprocess.TimeoutExpired:
        logger.warning("OpenClaw MiniMax extraction timed out")
        return None
    except Exception as e:
        logger.warning("OpenClaw MiniMax extraction failed: %s", e)
        return None


def _call_ollama_extraction(
    text: str,
    llm: Any,
    parser: JsonOutputParser,
    chain: Any,
) -> ExtractionResult:
    """Call Ollama LLM for extraction (fallback path)."""
    result = chain.invoke({"text": text})
    return ExtractionResult(**result)


class KnowledgeProcessor:
    """Agent that extracts structured knowledge from text.

    Uses a primary/fallback chain:
    1. Primary: MiniMax API (if credentials available)
    2. Fallback: Ollama via LangChain
    """

    def __init__(self, llm: Any | None = None) -> None:
        """Initialize with an optional LangChain-compatible Ollama LLM.

        The LLM is only used as fallback when the primary MiniMax path is unavailable.

        Args:
            llm: LangChain LLM instance (e.g., ChatOllama) for fallback extraction
        """
        self._llm = llm
        self._parser = JsonOutputParser(pydantic_object=ExtractionResult)
        self._chain = EXTRACTION_PROMPT | llm | self._parser if llm else None

    @property
    def _has_fallback(self) -> bool:
        return self._llm is not None and self._chain is not None

    def _parse_extraction_result(self, raw: dict[str, Any]) -> ExtractionResult:
        """Parse and validate a raw dict into an ExtractionResult.

        Silently drops invalid fields rather than raising.
        """
        summary = raw.get("summary", "")
        tags = raw.get("tags", [])
        entities = raw.get("entities", [])
        relationships = raw.get("relationships", [])
        existing_links_raw = raw.get("existing_links", [])

        # Parse existing_links, filtering out any with invalid structure
        existing_links: list[ExistingLink] = []
        for link in existing_links_raw:
            if isinstance(link, dict) and "target_id" in link:
                try:
                    existing_links.append(ExistingLink(**link))
                except Exception:
                    pass  # Skip invalid links

        return ExtractionResult(
            summary=summary,
            tags=tags if isinstance(tags, list) else [],
            entities=entities if isinstance(entities, list) else [],
            relationships=relationships,
            existing_links=existing_links,
        )

    def extract_with_context(
        self,
        text: str,
        retrieved_nodes: list[GraphNode],
    ) -> ExtractionResult:
        """Extract structured knowledge with retrieval-augmented context.

        This is the primary extraction entry point used by the pipeline.
        It attempts MiniMax first, then falls back to Ollama.

        Args:
            text: The new memo text to extract from
            retrieved_nodes: Existing graph nodes retrieved via similarity search

        Returns:
            ExtractionResult with summary, tags, entities, relationships, and existing_links
        """
        # Try OpenClaw MiniMax primary extractor
        minimax_result = self._try_openclaw_minimax_primary(text, retrieved_nodes)
        if minimax_result is not None:
            return minimax_result

        # Fall back to Ollama
        if self._has_fallback:
            return self._extract_fallback_with_context(text, retrieved_nodes)

        # Last resort: heuristic extraction
        return self._heuristic_extraction(text)

    def _try_openclaw_minimax_primary(
        self,
        text: str,
        retrieved_nodes: list[GraphNode],
    ) -> ExtractionResult | None:
        """Try primary OpenClaw MiniMax extraction with retrieval context."""
        context_str = _build_retrieval_context(retrieved_nodes)
        prompt = _RETRIEVAL_CONTEXT_TEMPLATE.format(
            existing_context=context_str,
            text=text,
        )

        raw = _call_openclaw_minimax_extraction(prompt, timeout=_OPENCLAW_TIMEOUT_SECONDS)
        if raw is None:
            logger.debug("OpenClaw MiniMax primary extraction returned None")
            return None

        return self._parse_extraction_result(raw)

    def _extract_fallback_with_context(
        self,
        text: str,
        retrieved_nodes: list[GraphNode],
    ) -> ExtractionResult:
        """Fallback extraction using Ollama with retrieval context.

        Since the base chain doesn't support retrieval context directly,
        we build a custom prompt here.
        """
        context_str = _build_retrieval_context(retrieved_nodes)
        prompt_text = _RETRIEVAL_CONTEXT_TEMPLATE.format(
            existing_context=context_str,
            text=text,
        )

        try:
            # Invoke the LLM directly with the retrieval-augmented prompt
            response = self._llm.invoke(prompt_text)
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse JSON from the response
            raw = self._extract_json_from_response(content)
            if raw:
                return self._parse_extraction_result(raw)
        except Exception as e:
            logger.warning(f"Fallback extraction failed: {e}")

        return self._heuristic_extraction(text)

    def _extract_json_from_response(self, content: str) -> dict[str, Any] | None:
        """Extract JSON object from LLM response text."""
        return _extract_json_from_text(content)

    def _heuristic_extraction(self, text: str) -> ExtractionResult:
        """Simple heuristic extraction as last resort."""
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

        return ExtractionResult(
            summary=text[:200] if len(text) > 200 else text,
            tags=tags[:5],
            entities=entities,
            relationships=[],
            existing_links=[],
        )

    # ---- Legacy sync/async interface (no retrieval context) ----

    def extract_sync(self, text: str) -> ExtractionResult:
        """Synchronous extraction without retrieval context (legacy path).

        Uses primary/fallback chain but does not include retrieval context.
        """
        # Try OpenClaw MiniMax without context first
        raw = _call_openclaw_minimax_extraction(
            _RETRIEVAL_CONTEXT_TEMPLATE.format(
                existing_context="(No existing nodes found)",
                text=text,
            )
        )
        if raw is not None:
            return self._parse_extraction_result(raw)

        # Fall back to Ollama chain
        if self._has_fallback:
            try:
                result = self._chain.invoke({"text": text})
                return ExtractionResult(**result.model_dump())
            except Exception:
                pass

        return self._heuristic_extraction(text)

    async def extract(self, text: str) -> ExtractionResult:
        """Async extraction (delegates to sync version)."""
        return self.extract_sync(text)
