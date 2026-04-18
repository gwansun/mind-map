"""Filter Agent for discard/duplicate/new decisions on incoming data.

Supports explicit memo targets resolved by the CLI.
When a target is set, that exact model path is used; on failure the memo is rejected.
"""
from __future__ import annotations

from typing import Any

from mind_map.core.schemas import FilterDecision, GraphNode
from mind_map.processor.cli_executor import MemoTarget, build_cli_template

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


def _run_custom_filter(
    cli_template: str,
    text: str,
    retrieved_concepts: list[GraphNode],
) -> FilterDecision:
    """Run a custom CLI filter command.

    Raises CLIExecutionError on failure (caller should reject the memo).
    """
    from mind_map.processor.cli_executor import run_filter_cli, CLIExecutionError

    prompt = FILTER_MINIMAX_PROMPT_TEMPLATE.format(
        text=text,
        retrieved_concepts=_format_retrieved_concepts(retrieved_concepts),
    )

    try:
        data = run_filter_cli(cli_template, prompt)
        return FilterDecision(
            action=data.get("action", "new"),
            reason=data.get("reason", "CLI classification"),
            summary=data.get("summary"),
        )
    except CLIExecutionError:
        raise  # Re-raise so caller can handle (reject memo)


class FilterAgent:
    """Filter agent with explicit target execution and heuristic trivial discard.

    The caller must provide a resolved target. Internal provider loading is bypassed.
    If target execution fails, the caller should reject the memo.
    """

    def __init__(self, target: MemoTarget | None = None) -> None:
        self._target = target

    def evaluate_sync(
        self,
        text: str,
        retrieved_concepts: list[GraphNode] | None = None,
    ) -> FilterDecision:
        retrieved_concepts = retrieved_concepts or []

        # Always check heuristic first — discard/trivial always returns discard
        heuristic = self._heuristic_filter(text, retrieved_concepts)
        if heuristic.action == "discard":
            return heuristic

        if self._target is None:
            raise ValueError("Memo target is required for filter evaluation")

        decision = _run_custom_filter(build_cli_template(self._target), text, retrieved_concepts)
        return decision

    def _heuristic_filter(
        self,
        text: str,
        retrieved_concepts: list[GraphNode],
    ) -> FilterDecision:
        """Simple non-LLM novelty filter (final fallback)."""
        trivial_patterns = [
            "hello", "hi", "thanks", "thank you", "ok", "okay",
            "yes", "no", "sure", "bye", "goodbye",
        ]
        lower_text = text.lower().strip()

        if len(text.strip()) < 10:
            return FilterDecision(
                action="discard",
                reason="Text too short (< 10 characters)",
                summary=None,
            )

        if lower_text in trivial_patterns:
            return FilterDecision(
                action="discard",
                reason="Trivial/greeting message",
                summary=None,
            )

        normalized = " ".join(text.lower().split())
        for concept in retrieved_concepts:
            if normalized and normalized == " ".join(concept.document.lower().split()):
                return FilterDecision(
                    action="duplicate",
                    reason=f"Matches existing concept {concept.id}",
                    summary=None,
                )

        return FilterDecision(
            action="new",
            reason="Content appears substantive and new",
            summary=text,
        )
