"""Knowledge Processor - LLM(B) for entity extraction and summarization."""

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mind_map.models.schemas import ExtractionResult

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
  "relationships": [["Entity1", "uses", "Entity2"], ["Entity2", "part_of", "Entity3"]]
}}"""),
    ("human", "Extract knowledge from this text:\n\n{text}"),
])


class KnowledgeProcessor:
    """Agent that extracts structured knowledge from text."""

    def __init__(self, llm: Any) -> None:
        """Initialize with a LangChain-compatible LLM.

        Args:
            llm: LangChain LLM instance (e.g., ChatOllama, ChatOpenAI)
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ExtractionResult)
        self.chain = EXTRACTION_PROMPT | llm | self.parser

    async def extract(self, text: str) -> ExtractionResult:
        """Extract structured knowledge from text.

        Args:
            text: Text to process

        Returns:
            ExtractionResult with summary, tags, entities, and relationships
        """
        result = await self.chain.ainvoke({"text": text})
        return ExtractionResult(**result)

    def extract_sync(self, text: str) -> ExtractionResult:
        """Synchronous version of extract.

        Args:
            text: Text to process

        Returns:
            ExtractionResult with summary, tags, entities, and relationships
        """
        result = self.chain.invoke({"text": text})
        return ExtractionResult(**result)
