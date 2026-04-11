"""Knowledge Processor - LLM(B) for entity extraction and summarization.

Supports a primary/fallback extraction chain:
1. Primary: OpenClaw MiniMax agent subprocess for structured extraction with retrieval context
2. Fallback: Ollama phi3.5 via LangChain
3. Final fallback: heuristic extraction
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from mind_map.core.schemas import ExistingLink, ExtractionResult
from mind_map.rag.graph_store import GraphNode

logger = __import__("logging").getLogger(__name__)

_OPENCLAW_TIMEOUT_SECONDS = 60.0

# Base extraction prompt (no retrieval context)
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """JSON-ONLY MODE: Respond with ONLY a raw JSON object. No prose. No markdown. No explanation. No greeting.

Required keys:
- summary (string, 1-2 sentences)
- tags (array of #hashtag strings)
- entities (array of entity name strings)
- relationships (array of [subject, predicate, object] arrays)
- existing_links (array of {target_id, relation_type} objects)

Example valid response:
{"summary":"Test","tags":["#test"],"entities":["X"],"relationships":[],"existing_links":[]}"""),
    ("human", "Extract: {text}"),
])

# Retrieval-augmented extraction prompt
_RETRIEVAL_CONTEXT_TEMPLATE = """EXTRACT JSON from the NEW text below. Respond with ONLY raw JSON. No text before or after.

EXISTING GRAPH NODES (for existing_links only - use their IDs):
{existing_context}

NEW TEXT TO EXTRACT:
{text}

Required keys: summary, tags, entities, relationships, existing_links
Only link to existing nodes using IDs from the list above.
Example: {{"summary":"...","tags":["#tag"],"entities":["X"],"relationships":[],"existing_links":[{{"target_id":"abc123","relation_type":"related_to"}}]}}"""


def _build_retrieval_context(retrieved_nodes: list[GraphNode]) -> str:
    """Build a compact context string from retrieved nodes."""
    if not retrieved_nodes:
        return "(none)"
    lines = []
    for node in retrieved_nodes[:10]:  # limit to 10 nodes
        type_label = node.metadata.type.value
        snippet = node.document[:150].replace("\n", " ")
        lines.append(f"[{type_label}] id={node.id}: {snippet}")
    return "\n".join(lines)


def _extract_json_from_text(content: str) -> dict[str, Any] | None:
    """Extract a JSON object from text, tolerating wrapper prose."""
    content = content.strip()
    # Try direct JSON parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try finding JSON object with regex (handles single-level nesting)
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _call_openclaw_minimax_extraction(
    prompt: str,
    timeout: float = _OPENCLAW_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Call OpenClaw MiniMax agent for structured extraction.

    Uses the local OpenClaw CLI as primary extraction path.
    """
    if shutil.which("openclaw") is None:
        logger.debug("openclaw CLI not found, skipping MiniMax extraction")
        return None

    command = [
        "openclaw",
        "agent",
        "--agent",
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
            logger.warning("OpenClaw MiniMax extraction failed (rc=%s): %s", result.returncode, result.stderr[:200])
            return None

        content = (result.stdout or "").strip()
        if not content:
            logger.warning("OpenClaw MiniMax returned empty content")
            return None

        return _extract_json_from_text(content)
    except subprocess.TimeoutExpired:
        logger.warning("OpenClaw MiniMax extraction timed out after %ds", timeout)
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
    1. Primary: MiniMax via OpenClaw CLI
    2. Fallback: Ollama via LangChain
    """

    def __init__(self, llm: Any | None = None) -> None:
        """Initialize with an optional LangChain-compatible Ollama LLM.

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

        Primary extraction uses MiniMax via OpenClaw CLI.
        Falls back to Ollama if available, then to heuristic extraction.

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
        """Fallback extraction using Ollama with retrieval context."""
        context_str = _build_retrieval_context(retrieved_nodes)
        prompt_text = _RETRIEVAL_CONTEXT_TEMPLATE.format(
            existing_context=context_str,
            text=text,
        )
        return _call_ollama_extraction(
            text=prompt_text,
            llm=self._llm,
            parser=self._parser,
            chain=self._chain,
        )

    def _heuristic_extraction(self, text: str) -> ExtractionResult:
        """Rule-based fallback when no LLM is available."""
        import re

        # Extract hashtags
        tag_pattern = re.compile(r"#(\w+)")
        tags = [f"#{m.group(1).lower()}" for m in tag_pattern.finditer(text)]
        tags = list(dict.fromkeys(tags))  # deduplicate preserving order

        # Extract capitalized "entities" (simple heuristic)
        entity_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
        entities = list(dict.fromkeys(m.group(0) for m in entity_pattern.finditer(text)))

        return ExtractionResult(
            summary=text[:300] if len(text) > 300 else text,
            tags=tags,
            entities=entities,
            relationships=[],
            existing_links=[],
        )
