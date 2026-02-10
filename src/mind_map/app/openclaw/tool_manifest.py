"""OpenClaw tool manifest for Mind Map capabilities."""

from typing import Any


def get_tool_manifest() -> dict[str, Any]:
    """Return the OpenClaw tool manifest describing mind-map's capabilities.

    The manifest follows OpenClaw's tool registration schema, exposing
    ask, memo, graph, and stats as invocable tools.

    Returns:
        Dictionary conforming to OpenClaw tool manifest format.
    """
    return {
        "name": "mind_map",
        "description": "Knowledge Graph-based Mind Map with RAG-enhanced responses",
        "version": "0.1.0",
        "tools": [
            {
                "name": "ask",
                "description": "Query the knowledge graph and get a RAG-enhanced response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to ask",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memo",
                "description": "Add a memo to the knowledge graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memo text to ingest",
                        }
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "graph",
                "description": "Get the full knowledge graph (nodes and edges)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "stats",
                "description": "Get knowledge graph statistics",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ],
    }
