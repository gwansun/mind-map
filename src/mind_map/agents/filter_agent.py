"""Filter Agent - LLM(B) for keep/discard decisions on incoming data."""

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mind_map.models.schemas import FilterDecision

FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data quality filter for a knowledge graph system.
Your job is to evaluate incoming text and decide whether it should be kept or discarded.

Evaluate based on:
1. **Information Gain**: Does this provide new, valuable context?
2. **Relevance**: Is this structural (logic, decisions, facts) or trivial (greetings)?
3. **Clarity**: Is the content interpretable and specific enough to be useful?

Respond ONLY with a JSON object:
{{"action": "keep"|"discard", "reason": "...", "summary": "condensed or null"}}"""),
    ("human", "Evaluate this text:\n\n{text}"),
])


class FilterAgent:
    """Agent that filters incoming data for knowledge graph ingestion."""

    def __init__(self, llm: Any) -> None:
        """Initialize with a LangChain-compatible LLM.

        Args:
            llm: LangChain LLM instance (e.g., phi-3.5 with ollama)
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=FilterDecision)
        self.chain = FILTER_PROMPT | llm | self.parser

    async def evaluate(self, text: str) -> FilterDecision:
        """Evaluate text and return a keep/discard decision.

        Args:
            text: Raw text to evaluate

        Returns:
            FilterDecision with action, reason, and optional summary
        """
        result = await self.chain.ainvoke({"text": text})
        return FilterDecision(**result)

    def evaluate_sync(self, text: str) -> FilterDecision:
        """Synchronous version of evaluate.

        Args:
            text: Raw text to evaluate

        Returns:
            FilterDecision with action, reason, and optional summary
        """
        result = self.chain.invoke({"text": text})
        return FilterDecision(**result)
