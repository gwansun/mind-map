"""Filter Agent - LLM(B) for discard/duplicate/new decisions on incoming data."""

import json
import subprocess
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mind_map.core.schemas import FilterDecision, GraphNode

FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a memo novelty filter for a knowledge graph system.
Your job is to classify incoming text as one of:
- new
- duplicate
- discard

Rules:
1. discard -> trivial, greeting, too short, or no useful information
2. duplicate -> the memo is already represented by one of the retrieved concept candidates
3. new -> useful new memo that should be extracted and stored

Retrieved concepts are only for duplicate checking.
Do not perform extraction.

Respond ONLY with a JSON object:
{"action": "new"|"duplicate"|"discard", "reason": "...", "summary": "cleaned text or null"}"""),
    ("human", "NEW MEMO:\n{text}\n\nRETRIEVED CONCEPT CANDIDATES:\n{retrieved_concepts}"),
])

FILTER_MINIMAX_PROMPT_TEMPLATE = """You are a memo novelty filter for a knowledge graph system.
Your job is to classify incoming text as one of: new, duplicate, or discard.

Rules:
1. discard -> trivial, greeting, too short, or no useful information
2. duplicate -> the memo is already represented by one of the retrieved concept candidates
3. new -> useful new memo that should be extracted and stored

Retrieved concepts are only for duplicate checking. Do not perform extraction.

NEW MEMO:
{text}

RETRIEVED CONCEPT CANDIDATES:
{retrieved_concepts}

Respond ONLY with a JSON object with keys: action (one of: new, duplicate, discard), reason (string), summary (string or null).
"""


def _format_retrieved_concepts(concepts: list[GraphNode]) -> str:
    if not concepts:
        return "(none)"
    lines = []
    for concept in concepts[:5]:
        snippet = concept.document[:160].replace("\n", " ")
        lines.append(f"- id={concept.id}: {snippet}")
    return "\n".join(lines)


def _call_minimax_fallback(text: str, retrieved_concepts: list[GraphNode]) -> FilterDecision:
    """Call OpenClaw MiniMax agent as fallback for filter classification.

    Raises:
        RuntimeError: If the MiniMax call fails or returns unparseable output.
    """
    import re

    prompt = FILTER_MINIMAX_PROMPT_TEMPLATE.format(
        text=text,
        retrieved_concepts=_format_retrieved_concepts(retrieved_concepts),
    )

    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "minimax", "--message", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip() if result.stdout else result.stderr.strip()

        if not output:
            raise RuntimeError("MiniMax returned empty output")

        # Try to extract JSON from output (may be wrapped in markdown or extra text)
        json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            # Try parsing the whole output as JSON
            data = json.loads(output)

        return FilterDecision(
            action=data.get("action", "new"),
            reason=data.get("reason", "MiniMax classification"),
            summary=data.get("summary"),
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError("MiniMax call timed out after 60s")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"MiniMax returned invalid JSON: {e} — output: {output[:200]}")
    except Exception as e:
        raise RuntimeError(f"MiniMax call failed: {e}")


class FilterAgent:
    """Agent that filters incoming data for knowledge graph ingestion."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=FilterDecision)
        self.chain = FILTER_PROMPT | llm | self.parser

    async def evaluate(self, text: str, retrieved_concepts: list[GraphNode] | None = None) -> FilterDecision:
        result = await self.chain.ainvoke({
            "text": text,
            "retrieved_concepts": _format_retrieved_concepts(retrieved_concepts or []),
        })
        return FilterDecision(**result)

    def evaluate_sync(self, text: str, retrieved_concepts: list[GraphNode] | None = None) -> FilterDecision:
        result = self.chain.invoke({
            "text": text,
            "retrieved_concepts": _format_retrieved_concepts(retrieved_concepts or []),
        })
        return FilterDecision(**result)


class FilterAgentWithFallback:
    """Filter agent with phi3.5-first, MiniMax-fallback chain.

    Evaluation order:
        1. phi3.5 via LangChain (if filter_llm is provided and available)
        2. OpenClaw MiniMax CLI (openclaw agent --agent minimax)
        3. Raises RuntimeError so the pipeline falls back to heuristic
    """

    def __init__(self, filter_llm: Any) -> None:
        self.filter_llm = filter_llm
        self._primary_agent = FilterAgent(filter_llm) if filter_llm else None

    def evaluate_sync(self, text: str, retrieved_concepts: list[GraphNode] | None = None) -> FilterDecision:
        retrieved_concepts = retrieved_concepts or []

        # Step 1: Try primary phi3.5 via LangChain
        if self._primary_agent is not None:
            try:
                return self._primary_agent.evaluate_sync(text, retrieved_concepts)
            except Exception:
                pass  # Fall through to MiniMax fallback

        # Step 2: Try OpenClaw MiniMax CLI
        try:
            return _call_minimax_fallback(text, retrieved_concepts)
        except RuntimeError:
            pass  # Fall through to pipeline heuristic

        # Step 3: Raise to let pipeline use heuristic
        raise RuntimeError("Both phi3.5 and MiniMax filter backends failed")
