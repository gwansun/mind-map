# Mind Map

A knowledge graph-based context management system for persisting and growing AI context across all conversations.

## Overview

Mind Map is a personal knowledge graph that accumulates context from every conversation and uses it to generate increasingly personalized AI responses. It functions as a **context management system** that persists what matters to you over time — not just the current session, but your entire history of interactions.

**Key idea:** Instead of starting each conversation cold, the AI draws from your accumulated context graph, ranked by what matters most to you.

## What It Does

- **Ingest notes** — Add text snippets, ideas, or conversation fragments to the knowledge graph
- **Smart retrieval** — Query the graph to get contextually relevant, personalized responses
- **Importance scoring** — Nodes are ranked by connectivity and recency, surfacing your true preferences and interests
- **Context injection** — Retrieved context can be injected into agentic AI system prompts or used in standalone LLM-powered Q&A

## Architecture

### Knowledge Graph

Data is stored as a graph of **nodes** (concepts, entities, tags) connected by **edges** (relationships). Unlike keyword-based retrieval, this understands *how* ideas are connected.

### LangGraph Pipeline

Notes flow through an orchestrated pipeline:

```
FilterAgent → KnowledgeProcessor → GraphStore
```

- **FilterAgent** — LLM decides whether input is worth storing (avoids redundancy)
- **KnowledgeProcessor** — Extracts entities, tags, relationships, and summary
- **GraphStore** — Dual-writes to ChromaDB (vectors) + SQLite (edges)

### Importance Scoring

```
S = (C_node / C_max) * e^(-λ * Δt)
```

Nodes are scored by connectivity (`C`) balanced against time decay — frequently referenced, well-connected ideas stay relevant without crowding out new information.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph |
| LLM Processing | Gemini, Claude, OpenAI, Ollama |
| Vector Storage | ChromaDB |
| Graph Storage | SQLite |
| API | FastAPI |
| Frontend | Angular 18 + D3.js |
| MCP Server | FastMCP |

## Quick Start

```bash
# Install
poetry install

# Initialize
poetry run mind-map init

# Add a note
poetry run mind-map memo "Thinking about building a RAG system"

# Ask a question
poetry run mind-map ask "What am I working on?"

# Start API server
poetry run mind-map serve
```

## MCP Integration

Mind Map exposes tools via FastMCP for agentic workflows:

- `mind_map_retrieve` — Search with importance ranking
- `mind_map_memo` — Ingest through the pipeline
- `mind_map_stats` — View graph statistics
- `mind_map_prune` — Clean up low-importance nodes

See [CLAUDE.md](CLAUDE.md) for full documentation.
