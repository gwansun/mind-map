"""Knowledge Processor - LLM(B) for entity extraction and summarization.

Supports explicit memo target execution for structured extraction.
The CLI must resolve the model path earlier and pass it in.
If the given target fails, extraction throws and the memo is rejected.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from mind_map.core.schemas import ExtractionResult, GraphNode
from mind_map.processor.cli_executor import MemoTarget, build_cli_template

logger = __import__("logging").getLogger(__name__)

_REFERENCE_CONTEXT_TEMPLATE = (
    """EXTRACT JSON from the NEW text below. Respond with ONLY raw JSON. No text before or after.

NEW TEXT:
{text}

REFERENCE ENTITIES/TAGS (optional grounding hints only, not facts to copy):
{references}

Required keys: summary, tags, entities, relationships
Extract only what is supported by NEW TEXT."""
)


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


def _parse_json_object(message: Any) -> dict[str, Any]:
    """Pull a JSON object out of a chat-model response.

    Tolerates fenced ```json blocks. Raises ValueError when the response is not
    a JSON object, so callers can decide how to degrade.
    """
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON was not an object")
    return parsed


def _call_custom_extraction(
    cli_template: str,
    prompt: str,
) -> dict[str, Any]:
    """Run a custom CLI extraction command.

    Raises CLIExecutionError on failure (caller should reject the memo).
    """
    from mind_map.processor.cli_executor import run_extraction_cli

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

    def _parse_extraction_result(
        self, raw: dict[str, Any], *, fallback_text: str
    ) -> ExtractionResult:
        summary = raw.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            summary = fallback_text.strip()

        tags = raw.get("tags", [])
        if isinstance(tags, list):
            tags = [tag for tag in tags if isinstance(tag, str) and tag.strip().startswith("#")]
        else:
            tags = []

        entities = raw.get("entities", [])
        if isinstance(entities, list):
            entities = [entity for entity in entities if isinstance(entity, str) and entity.strip()]
        else:
            entities = []

        relationships = raw.get("relationships", [])

        return ExtractionResult(
            summary=summary,
            tags=tags,
            entities=entities,
            relationships=relationships,
        )

    def _build_prompt(self, text: str, reference_nodes: list[GraphNode] | None) -> str:
        """Single source of the extraction prompt for the target and LLM paths."""
        return _REFERENCE_CONTEXT_TEMPLATE.format(
            text=text,
            references=_build_reference_context(reference_nodes or []),
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
        if self._target is None:
            raise ValueError("Memo target is required for extraction")

        prompt = self._build_prompt(text, reference_nodes)
        raw = _call_custom_extraction(build_cli_template(self._target), prompt)
        return self._parse_extraction_result(raw, fallback_text=text)

    def extract_with_llm(
        self,
        text: str,
        reference_nodes: list[GraphNode] | None = None,
    ) -> ExtractionResult:
        """Extract structured knowledge through the injected chat LLM.

        The internal (non-CLI) counterpart of ``extract_with_references``: same
        prompt and parser, but called through the LLM object rather than a memo
        target subprocess. Raises on failure; the caller decides how to degrade.
        """
        if self._llm is None:
            raise ValueError("LLM is required for LLM extraction")

        messages = [HumanMessage(content=self._build_prompt(text, reference_nodes))]
        return self._parse_extraction_result(
            _parse_json_object(self._llm.invoke(messages)),
            fallback_text=text,
        )

