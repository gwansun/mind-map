# Mind-Map — Project Documentation

> **GitHub:** https://github.com/gwansun/mind-map.git  
> **Upstream:** https://github.com/Gwanjin-Chun/mind-map.git  
> **Local path:** `/Users/gwansun/Desktop/projects/mind-map`

---

## Mind-map Purpose

Mind Map is a **knowledge graph-based context management system** that solves a fundamental problem with AI assistants: **lack of persistent memory across conversations**.

Instead of starting every session from scratch, Mind Map accumulates context from every conversation and uses it to generate increasingly personalized AI responses. It functions as a **context management system** that persists what matters over time, not just the current session, but your entire history of interactions.

### Mind-map, What Problem Does It Solve?

| Problem | Solution |
|---------|----------|
| AI forgets everything between sessions | Knowledge graph persists context indefinitely |
| Simple keyword search returns irrelevant results | Graph traversal finds clusters of related context |
| Low-value data pollutes the database | FilterAgent discards trivial content before storage |
| New information used to be stored without grounded links to existing knowledge | Retrieval-augmented memo ingestion compares new memos to existing graph records before extraction |
| New information crowds out important foundational facts | Importance scoring with time-decay balances recency and connectivity |
| No visualization of how ideas connect | Angular + D3.js interactive graph explorer |

---

## Mind-map Project Directory Structure

```text
mind-map/
├── src/mind_map/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # FastAPI endpoints
│   │   ├── cli/
│   │   │   └── main.py                # Typer CLI commands
│   │   └── pipeline.py                # LangGraph ingestion pipeline
│   ├── core/
│   │   ├── config.py                  # Configuration management
│   │   └── schemas.py                 # Pydantic data models
│   ├── processor/
│   │   ├── filter_agent.py            # LLM(B) keep/discard decisions
│   │   ├── knowledge_processor.py     # Retrieval-aware extraction logic
│   │   └── processing_llm.py          # LLM provider selection & fallback
│   ├── rag/
│   │   ├── graph_store.py             # ChromaDB + SQLite dual storage
│   │   ├── llm_status.py              # LLM availability checker
│   │   ├── reasoning_llm.py           # LLM(A) response generation
│   │   └── response_generator.py      # RAG-enhanced response synthesis
│   └── mcp/
│       └── server.py                  # FastMCP server for agentic workflows
├── frontend/
│   └── src/app/
│       ├── core/
│       │   └── api.service.ts         # Angular API client
│       ├── features/
│       │   ├── chat/                  # Chat UI components
│       │   ├── graph/                 # D3.js graph visualization
│       │   └── inspector/             # Node inspector panel
│       └── models/
│           └── graph.model.ts         # TypeScript interfaces
├── tests/                             # Pytest unit & integration tests
├── e2e/                               # Playwright end-to-end tests
├── config.yaml                        # Configuration file
├── pyproject.toml                     # Poetry project definition
├── README.md                          # Primary user-facing project README
├── README-PROJECT.md                  # Extended project documentation
└── CLAUDE.md                          # Developer and architecture reference
```

---

## Mind-map Core Algorithm

### 1. Importance Scoring (Time-Decay Formula)

Nodes are ranked by an importance score `S` that balances **connectivity** against **recency**:

```text
S = (C_node / C_max) * e^(-λ * Δt)
```

| Variable | Meaning |
|----------|---------|
| `C_node` | Number of edges connected to this node, counted bidirectionally |
| `C_max` | Maximum connection count of any node in the graph |
| `λ` | Decay constant (default 0.05) |
| `Δt` | Time elapsed since the node's last interaction |

**Effect:** Highly connected or recently used knowledge remains important, while isolated stale nodes decay naturally.

---

### 2. Retrieval-Augmented Memo Ingestion Pipeline

The memo workflow is no longer just `Filter → Extract → Store`.

It now performs retrieval before extraction so the extractor can form grounded relations to existing records.

```text
raw text
   │
   ▼
┌──────────────────┐
│  FilterAgent     │  decides keep/discard
└────────┬─────────┘
         │ keep
         ▼
┌──────────────────┐
│ Similarity       │  retrieves top relevant existing nodes
│ Retrieval        │  from the graph using ChromaDB
└────────┬─────────┘
         ▼
┌──────────────────┐
│ KnowledgeProcessor│ processes:
│  (extract)        │  • new memo text
│                   │  • retrieved existing nodes with IDs
│                   │
│                   │ extracts:
│                   │  • summary
│                   │  • tags
│                   │  • entities
│                   │  • relationships
│                   │  • existing_links
└────────┬─────────┘
         ▼
┌──────────────────┐
│  GraphStore      │  persists concept, tags, entities,
│  (persist)       │  relationships, and validated links
└──────────────────┘
```

### Why this change matters

Previously, the extractor only saw the new memo text. That meant it could summarize and extract entities, but it could not reliably connect new information to prior knowledge already stored in the graph.

Now the extractor receives relevant existing graph records, including their IDs, so it can create grounded `existing_links`.

Only IDs supplied through retrieval context are allowed to become stored links.

---

### 3. Extraction Chain and Fallback Logic

Memo extraction now uses a layered fallback strategy:

1. **Primary**: OpenClaw MiniMax
   - invoked through local OpenClaw CLI
   - designed to do the best structured extraction with retrieved context
2. **Fallback**: configured processing LLM
   - typically Ollama `phi3.5`
3. **Final fallback**: heuristic extraction
   - hashtags become tags
   - capitalized words become entities

This keeps memo ingestion resilient even if the preferred model path fails.

---

### 4. Existing Links and Entity Persistence

Two major backend fixes were added:

#### Existing links
The extraction schema now supports:

```json
{
  "existing_links": [
    {
      "target_id": "existing-node-id",
      "relation_type": "extends"
    }
  ]
}
```

Rules:
- only retrieved candidate IDs are allowed
- hallucinated IDs are dropped
- deduplication preserves different relation types to the same target

#### Standalone entities
Previously, extracted entities could be lost unless they also appeared inside relationship tuples.

Now standalone entities are always persisted and connected to the new concept with `mentions` edges.

---

### 5. Hybrid Storage (ChromaDB + SQLite)

| Storage | Technology | What it holds |
|---------|------------|---------------|
| Vector store | ChromaDB | Node content, embeddings, metadata |
| Edge registry | SQLite | Graph edges with source, target, weight, relation type |

Both stores are updated through `GraphStore`.

---

### 6. Node Types

| Type | Role | Example |
|------|------|---------|
| `concept` | Main summarized knowledge unit | "OAuth 2.0 allows delegated access" |
| `tag` | Topic labels | `#Authentication`, `#Security` |
| `entity` | Named things or technical terms | `Python`, `ChromaDB`, `Google Cloud` |

Common edge types:
- `tagged_as`
- `mentions`
- custom extracted relation types
- links from `existing_links`

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Orchestration | **LangGraph** | Memo ingestion pipeline |
| Memo extraction primary | **OpenClaw agent + MiniMax** | Retrieval-grounded extraction |
| Memo extraction fallback | **Ollama phi3.5** | Structured extraction fallback |
| General processing LLM | **Gemini / Claude / OpenAI / Ollama** | Filtering, extraction, summarization |
| Reasoning LLM | **OpenClaw Agent / Claude CLI / Cloud APIs** | RAG-enhanced answer synthesis |
| Vector DB | **ChromaDB** | Node embeddings and metadata |
| Graph DB | **SQLite** | Edge registry |
| API | **FastAPI** | REST endpoints |
| Frontend | **Angular 18 + D3.js** | Interactive visualization |
| MCP Server | **FastMCP** | OpenClaw agent tools |
| CLI packaging | **Typer + uv tool install** | CLI entrypoint and installation |
| Testing | **Pytest + Playwright** | Backend and E2E testing |

---

## Key Features

- **Retrieval-augmented memo ingestion** — new memos are linked against existing knowledge before extraction
- **Grounded graph linking** — only retrieved node IDs can become stored `existing_links`
- **Standalone entity persistence** — extracted entities are preserved even without relationship tuples
- **Importance-ranked retrieval** — context ranked by connectivity and recency
- **Multi-stage fallback behavior** — OpenClaw MiniMax → processing LLM → heuristic fallback
- **Graph visualization** — D3.js force-directed graph explorer
- **MCP integration** — tools for retrieve, memo, stats, health, report, and prune
- **Node deletion rules** — concept delete can cascade to first-layer tags
- **CLI install/update path** — managed via `uv tool install --reinstall`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API root |
| `GET` | `/health` | Health check + graph readiness |
| `GET` | `/graph` | Full graph for visualization |
| `GET` | `/node/{node_id}` | Single node details with connected edges |
| `GET` | `/stats` | Graph statistics |
| `POST` | `/ask` | RAG-enhanced query |
| `POST` | `/memo` | Ingest a note through the retrieval-augmented pipeline |
| `DELETE` | `/node/{node_id}` | Delete a node |

---

## CLI Reference

All commands can be run either as:

```bash
poetry run mind-map <command>
```

or, after installation:

```bash
mind-map <command>
```

### Main Commands

| Command | Description |
|---------|-------------|
| `mind-map init` | Initialize database and configuration |
| `mind-map init --data-dir /path` | Initialize at a custom database path |
| `mind-map memo "TEXT"` | Ingest a memo through the retrieval-augmented pipeline |
| `mind-map memo "TEXT" --no-llm` | Force heuristic-only processing |
| `mind-map ask "QUERY"` | Query the graph |
| `mind-map stats` | View graph statistics |
| `mind-map serve` | Start the FastAPI backend |
| `mind-map help` | Show help |

### Model Management

| Command | Description |
|---------|-------------|
| `mind-map model list` | List Ollama models |
| `mind-map model get` | Show current processing model |
| `mind-map model set <name>` | Set processing model |
| `mind-map model set <name> --persist` | Persist model to config |
| `mind-map model pull <name>` | Download an Ollama model |
| `mind-map model select` | Interactive model chooser |

### CLI Install / Refresh

The installed CLI on this machine is managed through `uv`.

Current installed path:

```bash
/Users/gwansun/.local/bin/mind-map
```

Refresh the installed CLI from local source:

```bash
uv tool install --reinstall /Users/gwansun/Desktop/projects/mind-map
```

---

## Configuration

### config.yaml

```yaml
processing_llm:
  provider: auto
  model: phi3.5
  temperature: 0.1
  auto_pull: false

reasoning_llm:
  provider: openclaw-agent
  model: main
  temperature: 0.7
  timeout: 120
```

### Similarity behavior

- query retrieval uses a conservative cosine `max_distance=0.5`
- memo-ingestion retrieval also uses a conservative threshold so weakly related nodes do not pollute extraction context

---

## OpenClaw Integration

### Database Path

When used inside the OpenClaw workspace, the active production-like data directory is:

```bash
/Users/gwansun/.openclaw/workspace/projects/mind-map/data
```

Examples:

```bash
mind-map stats -d /Users/gwansun/.openclaw/workspace/projects/mind-map/data
mind-map memo "note" -d /Users/gwansun/.openclaw/workspace/projects/mind-map/data
mind-map ask "query" -d /Users/gwansun/.openclaw/workspace/projects/mind-map/data
mind-map serve -d /Users/gwansun/.openclaw/workspace/projects/mind-map/data
```

### Backend restart path used in recent work

The backend was restarted against:

```bash
/Users/gwansun/.openclaw/workspace/projects/mind-map/data
```

### MCP Server Integration

Mind Map exposes tools through FastMCP for OpenClaw workflows, including:
- `mind_map_retrieve`
- `mind_map_memo`
- `mind_map_stats`
- `mind_map_report`
- `mind_map_prune`
- `mind_map_health`

---

## Summary of Recent Changes

Recent backend and docs changes now reflected in project documentation:

- added retrieval before memo extraction
- added `existing_links` schema
- validated links against retrieved node IDs only
- fixed standalone entity persistence
- changed memo extraction primary path to OpenClaw MiniMax
- kept processing LLM fallback, usually Ollama `phi3.5`
- restored conservative retrieval threshold
- refreshed installed CLI with `uv tool install --reinstall`
- updated `README.md`, `README-PROJECT.md`, and `CLAUDE.md`
