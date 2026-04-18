"""Knowledge Processor - LLM(B) for entity extraction and summarization.

Supports explicit memo target execution for structured extraction.
The CLI must resolve the model path earlier and pass it in.
If the given target fails, extraction throws and the memo is rejected.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mind_map.core.schemas import ExtractionResult, GraphNode
from mind_map.processor.cli_executor import MemoTarget, build_cli_template

logger = __import__("logging").getLogger(__name__)

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


def _call_custom_extraction(
    cli_template: str,
    prompt: str,
) -> dict[str, Any]:
    """Run a custom CLI extraction command.

    Raises CLIExecutionError on failure (caller should reject the memo).
    """
    from mind_map.processor.cli_executor import run_extraction_cli, CLIExecutionError
    return run_extraction_cli(cli_template, prompt)


class KnowledgeProcessor:
    """Agent that extracts structured knowledge from text."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        target: MemoTarget | None = None,
    ) -> None:
        self._llm = llm
        self._target = target
        self._parser = JsonOutputParser(pydantic_object=ExtractionResult)
        self._chain = EXTRACTION_PROMPT | llm | self._parser if llm else None

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

        Uses the explicit target only; raises on failure.
        No internal provider loading or fallback is allowed on the memo CLI path.
        """
        references = _build_reference_context(reference_nodes)

        if self._target is None:
            raise ValueError("Memo target is required for extraction")

        prompt = _REFERENCE_CONTEXT_TEMPLATE.format(
            text=text,
            references=references,
        )
        raw = _call_custom_extraction(build_cli_template(self._target), prompt)
        return self._parse_extraction_result(raw)

