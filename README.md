# Mind Map

A knowledge graph-based context management system for persisting and growing AI context across all conversations.

## Overview

Mind Map is a personal knowledge graph that accumulates context from every conversation and uses it to generate increasingly personalized AI responses. It functions as a **context management system** that persists what matters to you over time, not just the current session, but your entire history of interactions.

**Key idea:** Instead of starting each conversation cold, the AI draws from your accumulated context graph, ranked by what matters most to you.

## What It Does

- **Ingest notes** — Add text snippets, ideas, or conversation fragments to the knowledge graph
- **Smart retrieval** — Query the graph to get contextually relevant, personalized responses
- **Importance scoring** — Nodes are ranked by connectivity and recency, surfacing your true preferences and interests
- **Context-aware memo ingestion** — New memos are compared against existing graph records before extraction so the system can form grounded links to prior knowledge
- **Context injection** — Retrieved context can be injected into agentic AI system prompts or used in standalone LLM-powered Q&A

## Architecture

### Knowledge Graph

Data is stored as a graph of **nodes** (concepts, entities, tags) connected by **edges** (relationships). Unlike keyword-based retrieval, this understands *how* ideas are connected.

<img width="1200" height="878" alt="demo1" src="https://github.com/user-attachments/assets/0a940eff-4941-4c1c-8aa8-cd2e0e44a3fa" />

### LangGraph Pipeline

Notes flow through an orchestrated pipeline:

```
FilterAgent → Similarity Retrieval → KnowledgeProcessor → GraphStore
```

- **FilterAgent** — decides whether input is worth storing
- **Similarity Retrieval** — searches existing graph records before extraction
- **KnowledgeProcessor** — extracts summary, tags, entities, relationships, and links to existing retrieved nodes
- **GraphStore** — dual-writes to ChromaDB (vectors) + SQLite (edges)

### Memo Ingestion Workflow

When `mind-map memo "..."` runs, the backend now performs retrieval-augmented ingestion:

1. Filter the incoming memo
2. Run similarity search against existing graph nodes
3. Pass the new memo and retrieved node records, including their IDs, into the extraction model
4. Extract:
   - summary
   - tags
   - entities
   - entity-to-entity relationships
   - `existing_links` to retrieved nodes
5. Persist the new concept plus edges to tags, entities, and validated existing nodes

This fixes a prior limitation where memo ingestion only saw the new text and could not form grounded links to existing graph records.

### Extraction Fallback Chain

Memo extraction now uses this processor order:

1. **Primary** — OpenClaw MiniMax via local `openclaw agent --agents minimax`
2. **Fallback** — configured processing LLM, commonly Ollama `phi3.5`
3. **Final fallback** — heuristic extraction

Only node IDs supplied from retrieval context are allowed to become `existing_links`, which prevents arbitrary hallucinated graph links.

### Importance Scoring

```
S = (C_node / C_max) * e^(-λ * Δt)
```

Nodes are scored by connectivity (`C`) balanced against time decay, frequently referenced, well-connected ideas stay relevant without crowding out new information.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph |
| Primary memo extraction | OpenClaw agent with MiniMax |
| Fallback processing LLM | Ollama `phi3.5` or other configured processing model |
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

# Initialize a custom database location
poetry run mind-map init --data-dir /path/to/mind-map-data

# Add a note
poetry run mind-map memo "Thinking about building a RAG system"

# Ask a question
poetry run mind-map ask "What am I working on?"

# Start API server
poetry run mind-map serve

# Start API server against a custom database location
poetry run mind-map serve --data-dir /path/to/mind-map-data
```

## MCP Integration

Mind Map exposes tools via FastMCP for agentic workflows:

- `mind_map_retrieve` — Search with importance ranking
- `mind_map_memo` — Ingest through the retrieval-augmented pipeline
- `mind_map_stats` — View graph statistics
- `mind_map_prune` — Clean up low-importance nodes

See [CLAUDE.md](CLAUDE.md) for full documentation.
