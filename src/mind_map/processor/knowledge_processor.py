"""Knowledge Processor - LLM(B) for entity extraction and summarization.

Supports a primary/fallback extraction chain:
1. Primary: OpenClaw MiniMax agent subprocess for structured extraction
2. Fallback: Ollama phi3.5 via LangChain
3. Final fallback: heuristic extraction

Supports custom CLI template via constructor (used by --model option in CLI).
When cli_template is set, that exact command is used as primary and no fallback occurs.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mind_map.core.schemas import ExtractionResult, GraphNode

logger = __import__("logging").getLogger(__name__)

_OPENCLAW_TIMEOUT_SECONDS = 60.0

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """JSON-ONLY MODE: Respond with ONLY a raw JSON object. No prose. No markdown. No explanation.

Extract only from NEW TEXT.
REFERENCE ENTITIES/TAGS are optional grounding hints only.
Do not copy unsupported facts from references.
Do not treat references as newly introduced facts unless NEW TEXT supports them.

Required keys:
- summary (string, 1-2 sentences)
- tags (array of #hashtag strings)
- entities (array of entity name strings)
- relationships (array of [subject, predicate, object] arrays)

Example valid response:
{"summary":"Test","tags":["#test"],"entities":["X"],"relationships":[]}"""),
    ("human", "NEW TEXT:\n{text}\n\nREFERENCE ENTITIES/TAGS:\n{references}"),
])

_REFERENCE_CONTEXT_TEMPLATE = """EXTRACT JSON from the NEW text below. Respond with ONLY raw JSON. No text before or after.

NEW TEXT:
{text}

REFERENCE ENTITIES/TAGS (optional grounding hints only, not facts to copy):
{references}

Required keys: summary, tags, entities, relationships
Extract only what is supported by NEW TEXT."""


def _build_reference_context(reference_nodes: list[GraphNode]) -> str:
    """Build a compact reference string from entity/tag nodes."""
    if not reference_nodes:
        return "(none)"
    lines = []
    for node in reference_nodes[:15]:
        type_label = node.metadata.type.value
        snippet = node.document[:120].replace("\n", " ")
        lines.append(f"[{type_label}] {snippet}")
    return "\n".join(lines)


def _extract_json_from_text(content: str) -> dict[str, Any] | None:
    """Extract a JSON object from text, tolerating wrapper prose."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def _call_openclaw_minimax_extraction(
    prompt: str,
    timeout: float = _OPENCLAW_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Call OpenClaw MiniMax agent for structured extraction."""
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


def _call_custom_extraction(
    cli_template: str,
    prompt: str,
) -> dict[str, Any]:
    """Run a custom CLI extraction command.

    Raises CLIExecutionError on failure (caller should reject the memo).
    """
    from mind_map.processor.cli_executor import run_extraction_cli, CLIExecutionError
    return run_extraction_cli(cli_template, prompt)


def _call_ollama_extraction(
    text: str,
    references: str,
    parser: JsonOutputParser,
    chain: Any,
) -> ExtractionResult:
    """Call Ollama LLM for extraction (fallback path)."""
    result = chain.invoke({"text": text, "references": references})
    return ExtractionResult(**result)


class KnowledgeProcessor:
    """Agent that extracts structured knowledge from text."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        cli_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._cli_template = cli_template
        self._parser = JsonOutputParser(pydantic_object=ExtractionResult)
        self._chain = EXTRACTION_PROMPT | llm | self._parser if llm else None

    @property
    def _has_ollama(self) -> bool:
        return self._llm is not None and self._chain is not None

    def _parse_extraction_result(self, raw: dict[str, Any]) -> ExtractionResult:
        summary = raw.get("summary", "")
        tags = raw.get("tags", [])
        entities = raw.get("entities", [])
        relationships = raw.get("relationships", [])

        return ExtractionResult(
            summary=summary,
            tags=tags if isinstance(tags, list) else [],
            entities=entities if isinstance(entities, list) else [],
            relationships=relationships,
        )

    def extract_with_references(
        self,
        text: str,
        reference_nodes: list[GraphNode],
    ) -> ExtractionResult:
        """Extract structured knowledge from new text with optional entity/tag references.

        When cli_template is set: uses that command only; raises on failure.
        Otherwise: MiniMax → Ollama LangChain → heuristic.
        """
        references = _build_reference_context(reference_nodes)

        # Custom CLI path (no fallback)
        if self._cli_template is not None:
            prompt = _REFERENCE_CONTEXT_TEMPLATE.format(
                text=text,
                references=references,
            )
            raw = _call_custom_extraction(self._cli_template, prompt)
            return self._parse_extraction_result(raw)

        # Built-in MiniMax primary
        raw = _call_openclaw_minimax_extraction(
            _REFERENCE_CONTEXT_TEMPLATE.format(text=text, references=references),
            timeout=_OPENCLAW_TIMEOUT_SECONDS,
        )
        if raw is not None:
            return self._parse_extraction_result(raw)

        # Ollama LangChain fallback
        if self._has_ollama:
            return self._extract_fallback_with_references(text, references)

        return self._heuristic_extraction(text)

    def _extract_fallback_with_references(
        self,
        text: str,
        references: str,
    ) -> ExtractionResult:
        return _call_ollama_extraction(
            text=text,
            references=references,
            parser=self._parser,
            chain=self._chain,
        )

    def _heuristic_extraction(self, text: str) -> ExtractionResult:
        tag_pattern = re.compile(r"#(\w+)")
        tags = [f"#{m.group(1).lower()}" for m in tag_pattern.finditer(text)]
        tags = list(dict.fromkeys(tags))

        entity_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
        entities = list(dict.fromkeys(m.group(0) for m in entity_pattern.finditer(text)))

        return ExtractionResult(
            summary=text[:300] if len(text) > 300 else text,
            tags=tags,
            entities=entities,
            relationships=[],
        )
