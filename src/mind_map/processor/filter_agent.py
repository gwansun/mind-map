"""Filter Agent - LLM(B) for discard/duplicate/new decisions on incoming data.

Primary: OpenClaw MiniMax CLI (openclaw agent --agent minimax)
Fallback: heuristic

Supports custom CLI template via constructor (used by --model option in CLI).
When cli_template is set, that exact command is run; on failure the memo is rejected.
"""
from __future__ import annotations

import json
import re
import subprocess
import shutil
from typing import Any

from mind_map.core.schemas import FilterDecision, GraphNode

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

_FILTER_TIMEOUT_SECONDS = 60.0


def _format_retrieved_concepts(concepts: list[GraphNode]) -> str:
    if not concepts:
        return "(none)"
    lines = []
    for concept in concepts[:5]:
        snippet = concept.document[:160].replace("\n", " ")
        lines.append(f"- id={concept.id}: {snippet}")
    return "\n".join(lines)


def _call_minimax_filter(
    text: str,
    retrieved_concepts: list[GraphNode],
) -> FilterDecision | None:
    """Call OpenClaw MiniMax CLI for filter classification.

    Returns None on failure so caller can fall through to heuristic.
    """
    if shutil.which("openclaw") is None:
        return None

    prompt = FILTER_MINIMAX_PROMPT_TEMPLATE.format(
        text=text,
        retrieved_concepts=_format_retrieved_concepts(retrieved_concepts),
    )

    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "minimax", "--message", prompt],
            capture_output=True,
            text=True,
            timeout=_FILTER_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode != 0:
            return None

        output = (result.stdout or "").strip()
        if not output:
            return None

        json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(output)

        return FilterDecision(
            action=data.get("action", "new"),
            reason=data.get("reason", "MiniMax classification"),
            summary=data.get("summary"),
        )

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


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
    """Filter agent with MiniMax CLI primary and heuristic fallback.

    When cli_template is set, that exact command is used as primary.
    On CLI failure, the memo is rejected (caller should not continue).
    """

    def __init__(self, cli_template: str | None = None) -> None:
        self._cli_template = cli_template

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

        # Use custom CLI if provided
        if self._cli_template is not None:
            decision = _run_custom_filter(self._cli_template, text, retrieved_concepts)
            return decision

        # Use built-in MiniMax CLI
        decision = _call_minimax_filter(text, retrieved_concepts)
        if decision is not None:
            return decision

        # Fall through to heuristic
        return heuristic

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
