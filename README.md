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

```text
retrieve -> filter -> extract -> store
```

- **retrieve** — searches existing graph records before filtering/extraction
- **filter** — decides whether input is worth storing
- **extract** — extracts summary, tags, entities, relationships, and grounded references
- **store** — dual-writes to ChromaDB (vectors) + SQLite (edges)

### Memo Ingestion Workflow

When `mind-map memo "..."` runs, the backend performs retrieval-augmented ingestion:

1. Retrieve similar concept nodes and first-hop entity/tag neighbors
2. Filter the incoming memo against that context
3. Pass the new memo and retrieved entity/tag references into the extraction model
4. Extract:
   - summary
   - tags
   - entities
   - entity-to-entity relationships
5. Persist the new concept plus edges to tags, entities, and related retrieved concepts

The strict memo CLI path now requires an explicit target and does not rely on implicit internal loading or fallback.

### Explicit Memo Model Paths

Memo ingestion requires an explicit model path at the CLI.

Supported modes:
1. **OpenClaw path** — `mind-map memo ... --openclaw [agent]`
2. **Local path** — `mind-map memo ... --local [model]`

Rules:
- exactly one of `--openclaw` or `--local` must be provided for the memo CLI
- if neither is provided, memo CLI ingestion fails early
- memo CLI ingestion bypasses implicit internal model loading
- if the selected path fails, memo CLI ingestion rejects with no fallback
- internal non-CLI ingestion paths still use a separate internal ingestion flow

Current defaults:
- `--openclaw` uses OpenClaw with default message `"info"`
- `--openclaw minimax` uses agent `minimax` with message `"info"`
- `--local` uses `http://127.0.0.1:11435/v1`
- `--local` without a model resolves the first model returned by `/v1/models`

### Importance Scoring

```text
S = (C_node / C_max) * e^(-λ * Δt)
```

Nodes are scored by connectivity (`C`) balanced against time decay, frequently referenced, well-connected ideas stay relevant without crowding out new information.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph |
| Primary memo extraction | OpenClaw agent or explicit local OpenAI-compatible endpoint |
| Legacy/internal processing LLM | Ollama `phi3.5` or other configured processing model |
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

# Add a note through OpenClaw
poetry run mind-map memo "Thinking about building a RAG system" --openclaw

# Add a note through local endpoint
poetry run mind-map memo "Thinking about building a RAG system" --local

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
- `mind_map_memo` — Ingest through the internal non-CLI memo ingestion path
- `mind_map_stats` — View graph statistics
- `mind_map_prune` — Clean up low-importance nodes
- `mind_map_report` — JSON report with summary and top nodes
- `mind_map_health` — system and integration health checks

See [CLAUDE.md](CLAUDE.md) for full documentation.
