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

In code, the stages are:

1. `FilterAgent` (LLM-B) → keep/discard decision based on information gain
2. `Similarity Retrieval` → query similar existing nodes before extraction
3. `KnowledgeProcessor` (LLM-B) → extract summary, tags, entities, relationships, and links to existing retrieved nodes
4. `GraphStore` → dual-write to ChromaDB (vectors) + SQLite (edges)

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

Memo ingestion is retrieval-augmented rather than text-only:

- it first retrieves top similar existing nodes from the graph
- those retrieved nodes are passed to the extractor with their IDs and documents
- extractor output can include `existing_links`, which are validated against retrieved IDs before storage
- standalone extracted entities are persisted even if they do not appear in relationship tuples

The strict memo CLI path requires an explicit target and does not rely on implicit internal loading or fallback.

### Memo Extraction Chain

Memo ingestion requires an explicit model target. Resolution order:

1. **Explicit local target (`--local`)**
   - any OpenAI-compatible endpoint; base URL defaults to `http://127.0.0.1:11435/v1`
   - override endpoint with `MIND_MAP_LOCAL_BASE_URL`
   - optional auth: `Authorization: Bearer …` header sent only when `MIND_MAP_LOCAL_API_KEY` is set
   - `--local` without a model value resolves the first model returned by `/v1/models`
2. **Default cloud target (no flags)** — the CommandCode gateway
   - **primary**: when `COMMANDCODE_API_KEY` is set, uses the OpenAI-compatible LocalTarget transport at `https://api.commandcode.ai/provider/v1` (model `deepseek/deepseek-v4.1-flash`, overridable via `MIND_MAP_LLM_MODEL`; the legacy `MIND_MAP_DEEPSEEK_MODEL` alias is still honoured)
   - **MiniMax fallback**: `MINIMAX_API_KEY` + `MiniMaxTarget` (`api.minimax.io`, model `MiniMax-M2.5`) — legacy path, kept working
3. **Configured processing LLM fallback** — commonly Ollama `phi3.5`
4. **Heuristic fallback** — used when both model-backed paths fail

Exactly one of the CommandCode gateway (default) / MiniMax API (legacy fallback) / `--local` must provide a model target. If none of `COMMANDCODE_API_KEY`, `MINIMAX_API_KEY`, nor `--local` is provided, memo CLI ingestion fails early; if the selected path fails, it rejects with no fallback.

Internal non-CLI ingestion paths use a separate internal ingestion flow.

This structure improves grounded linking while keeping ingestion resilient.

### Prompting Notes for `knowledge_processor.py`

Extraction prompt characteristics:

- a short **JSON-only** prompt rather than a long explanatory one
- the base prompt explicitly forbids prose, markdown, explanation, and greetings
- the retrieval-context prompt says `EXTRACT JSON` and focuses on required keys + allowed existing IDs
- retrieval context is compacted:
  - max 10 retrieved nodes
  - each node snippet trimmed to 150 chars
  - empty context marker is `(none)`

The goal of the simplification is to elicit machine-parseable JSON instead of conversational wrapper text from the extraction model.

### MiniMax API Integration Notes

MiniMax is the legacy memo target and is kept working:

- called directly via the `minimax` Python module (HTTP API)
- `knowledge_processor.py` invokes it through `subprocess.run(...)`
- JSON is parsed first directly, then with a regex fallback to tolerate wrapper text

### Importance Scoring

```text
S = (C_node / C_max) * e^(-λ * Δt)
```

- `C_node` and `C_max` are both counted bidirectionally (source OR target)
- score is clamped to `[0.0, 1.0]`

Nodes are scored by connectivity (`C`) balanced against time decay — frequently referenced, well-connected ideas stay relevant without crowding out new information.

### Similarity Threshold

- `query_similar()` uses `max_distance=0.5` (cosine) to filter irrelevant results
- memo ingestion retrieval also uses a conservative threshold so only reasonably relevant existing nodes are passed into extraction

### Relation Factor

Context nodes are weighted by edge density to the most query-relevant node. Combined score: `importance * (1 + relation_factor)` where `relation_factor = edges_between(anchor, node) / total_edges(anchor)`.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph |
| Memo extraction (CLI + MCP) | CommandCode gateway (OpenAI-compatible) via the resolved memo target |
| Memo/processing fallback | Ollama `phi3.5` or another configured processing model; MiniMax is the legacy cloud fallback |
| Vector Storage | ChromaDB |
| Graph Storage | SQLite |
| API | FastAPI |
| Frontend | Angular 18 + D3.js |
| MCP Server | FastMCP |

## LLM Configuration

| Role | Provider | Default Model | Purpose |
|------|----------|---------------|---------|
| Processing (general LLM-B) | Cloud APIs (auto) / Ollama fallback | gemini-2.0-flash | Filtering, extraction, summarization |
| Memo extraction | CommandCode gateway (default) / MiniMax (legacy) / `--local` | deepseek/deepseek-v4.1-flash | Retrieval-grounded memo ingestion |
| Memo extraction fallback | Ollama / configured processing model | phi3.5 | Structured extraction fallback |
| Reasoning (LLM-A) | CommandCode gateway / fallbacks | deepseek/deepseek-v4.1-flash | Response generation |

**Processing (general LLM-B)**: cloud-first with validated fallback to Ollama

- provider priority (`auto`): Gemini → Anthropic → OpenAI → Ollama
- each cloud provider is validated with a test API call before use; if validation fails, the next provider is tried
- cloud models: `gemini-2.0-flash`, `claude-haiku-4-5-20250901`, `gpt-4o-mini`
- Ollama recommended: `phi3.5`, `phi3`, `llama3.2`, `mistral`, `gemma2:2b`, `qwen2.5:3b`
- config: `processing_llm.provider` in `config.yaml` (`auto`|`gemini`|`anthropic`|`openai`|`ollama`)
- auto-pull (Ollama): disabled by default

**Reasoning (LLM-A)**: DeepSeek family via the CommandCode gateway (default) with fallbacks

- priority: CommandCode gateway → Claude CLI → Gemini → Anthropic Claude → OpenAI GPT
- default: `deepseek` (`deepseek/deepseek-v4.1-flash`; override via `MIND_MAP_LLM_MODEL`)
- transport is `requests`; the gateway edge accepts its default User-Agent

## Build & Run

```bash
# Install dependencies
poetry install

# Run CLI
poetry run mind-map help              # Show comprehensive help
poetry run mind-map init              # Initialize database
poetry run mind-map init --with-ollama  # Initialize with Ollama model
poetry run mind-map init --data-dir /path/to/db  # Initialize a custom database path
poetry run mind-map memo "text"       # Ingest a note (retrieval-augmented extraction)
poetry run mind-map memo "text" --no-llm  # Ingest with heuristic only
poetry run mind-map memo "text" --data-dir /path/to/db  # Use a custom database path
poetry run mind-map ask "query"       # Query the knowledge graph
poetry run mind-map ask "query" --data-dir /path/to/db  # Query a custom database path
poetry run mind-map stats             # View graph statistics
poetry run mind-map stats --data-dir /path/to/db  # View stats for a custom database path
poetry run mind-map serve             # Start FastAPI server
poetry run mind-map serve --data-dir /path/to/db  # Start API server for a custom database path

# Model Management
poetry run mind-map model list        # List available Ollama models
poetry run mind-map model get         # Show current processing model
poetry run mind-map model set phi3.5  # Set processing model
poetry run mind-map model set phi3.5 --persist  # Set and save to config
poetry run mind-map model pull mistral  # Download a model
poetry run mind-map model select      # Interactive model selection

# Development (Backend)
poetry run ruff check .               # Lint
poetry run ruff format .              # Format
poetry run mypy src                   # Type check
poetry run pytest                     # Run tests

# Frontend
cd frontend
npm install                           # Install dependencies
npm start                             # Dev server (http://localhost:4200)
npm run build                         # Production build
```

## CLI Commands

### Main Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize database and configuration (`--data-dir` supported) |
| `memo TEXT` | Ingest a note into the knowledge graph with retrieval-augmented extraction (`--data-dir` supported) |
| `ask QUERY` | Query with RAG-enhanced response (`--data-dir` supported) |
| `stats` | Display knowledge graph statistics (`--data-dir` supported) |
| `serve` | Start FastAPI server (`--data-dir` supported) |
| `ollama-init` | Initialize Ollama processing model |
| `help` | Show comprehensive help with all options |

### Model Management (`model` subcommand)

| Command | Description |
|---------|-------------|
| `model list` | List available models with recommendations |
| `model get` | Show currently selected model |
| `model set <name>` | Set processing model (add `--persist` to save) |
| `model pull <name>` | Download an Ollama model |
| `model select` | Interactive model selection |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| GET | `/graph` | Full graph (nodes + edges) |
| GET | `/node/{id}` | Node details with edges |
| GET | `/stats` | Graph statistics |
| POST | `/ask` | Query with RAG response |
| POST | `/memo` | Ingest memo via LangGraph retrieval-augmented pipeline |

## MCP Integration

The MCP server (`src/mind_map/mcp/server.py`) exposes the following tools via FastMCP. All tools accept `data_dir` for CLI parity and optional `workspace_id` for multi-workspace isolation. Business logic lives in `src/mind_map/app/services.py` — both CLI and MCP are thin wrappers around it.

| Tool | Description |
|------|-------------|
| `mind_map_retrieve` | Similarity search with relation-factor enrichment and context expansion |
| `mind_map_memo` | Ingest text through the resolved memo target (CommandCode gateway by default) |
| `mind_map_ask` | RAG-enhanced LLM query (read-only by default; `back_feed=True` for CLI parity) |
| `mind_map_stats` | Knowledge graph statistics |
| `mind_map_report` | JSON report with summary stats and top-5 nodes |
| `mind_map_prune` | Prune the least important nodes (configurable `percent`, default 10%) |
| `mind_map_health` | System health check (Ollama, databases, ProcessingLLM, integration tests) |

### Prune Algorithm (`mind_map_prune`)

1. Calculate importance for all concept/entity nodes (tags excluded as direct candidates)
2. Sort ascending, take bottom `max(1, floor(total * 0.1))`
3. Tags removed only if **all** their edges connect to nodes in the prune set (shared tags preserved)
4. Delete edges, single-connected tags, then prune target nodes
5. Returns JSON: `{ deleted_nodes, deleted_tags, deleted_edges_count, summary }`

## Configuration

### config.yaml

```yaml
processing_llm:
  provider: auto
  model: phi3.5
  temperature: 0.1
  auto_pull: false

reasoning_llm:
  provider: deepseek            # DeepSeek model family, served via the CommandCode gateway
  model: deepseek/deepseek-v4.1-flash
  temperature: 0.7
  timeout: 120
```

`config.yaml` and `.env` are resolved from the **package location**, not the process working directory, so behaviour does not depend on where the process was started. Copy `.env.example` to `.env` to get started.

### .env

```bash
# Default cloud target — both LLM paths use this route
COMMANDCODE_API_KEY=your-commandcode-api-key

# Optional overrides
#MIND_MAP_LLM_MODEL=deepseek/deepseek-v4.1-flash
#MIND_MAP_LLM_BASE_URL=https://api.commandcode.ai/provider/v1

# Legacy fallback
#MINIMAX_API_KEY=

# Processing LLM providers
GOOGLE_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
```

## Storage

- **ChromaDB** (`data/chroma/` by default): vector embeddings + node metadata
- **SQLite** (`data/edges.db` by default): edge registry with source, target, weight, relation_type
- Override the default storage root with `--data-dir /path/to/db` on supported CLI commands

## Data Flow

### Ingestion (`memo` command)

```text
Text Input
  → FilterAgent
  → Similarity Retrieval (top relevant existing nodes)
  → KnowledgeProcessor
      - resolved memo target (CommandCode gateway by default)
      - processing LLM fallback
      - heuristic fallback
  → GraphStore
      - concept node
      - tag edges
      - entity mentions edges
      - relationship edges
      - validated existing_links edges
```

### Query (`ask` command)

```text
Query → ChromaDB Search (max_distance=0.5) → Enrich (relation factor) → ResponseGenerator → Response
```

- the reasoning LLM always generates a response, even for new topics with no context
- the processing LLM pipeline has a heuristic fallback if LLM calls fail at runtime
- `ask` uses its own linking flow, separate from the memo-ingestion `existing_links` behaviour

## Key Files

### Core (shared types & config)

- `src/mind_map/core/schemas.py` — Pydantic models including `ExistingLink` and the extraction schema
- `src/mind_map/core/config.py` — configuration loader (`config.yaml`, `.env`); resolves paths from the package root

### Processor (LLM-B)

- `src/mind_map/processor/processing_llm.py` — multi-provider processing LLM: cloud APIs + Ollama
- `src/mind_map/processor/filter_agent.py` — `FilterAgent` for keep/discard decisions
- `src/mind_map/processor/knowledge_processor.py` — `KnowledgeProcessor` for retrieval-aware extraction
- `src/mind_map/processor/cli_executor.py` — builds and runs the memo target CLI command

### RAG (storage & reasoning)

- `src/mind_map/rag/graph_store.py` — hybrid ChromaDB + SQLite storage
- `src/mind_map/rag/reasoning_llm.py` — multi-provider reasoning LLM routing
- `src/mind_map/rag/response_generator.py` — `ResponseGenerator` (LLM-A) for RAG synthesis
- `src/mind_map/rag/llm_status.py` — health/status checks for both LLM providers

### App (orchestration, CLI, API)

- `src/mind_map/app/services.py` — shared business logic; single source of truth for CLI and MCP
- `src/mind_map/app/pipeline.py` — LangGraph ingestion pipeline with retrieval before extraction
- `src/mind_map/app/cli/main.py` — Typer CLI entry point (thin wrappers around `services.py`)
- `src/mind_map/app/api/routes.py` — FastAPI endpoints for the frontend

### MCP (Model Context Protocol server)

- `src/mind_map/mcp/server.py` — FastMCP server (thin wrappers around `services.py`) with 7 tools

### Frontend (Angular 18+)

- `frontend/src/app/app.component.ts` — main layout with three-panel design
- `frontend/src/app/core/api.service.ts` — HTTP client with caching
- `frontend/src/app/features/graph/` — D3.js graph visualization
- `frontend/src/app/features/chat/` — chat interface with markdown support
- `frontend/src/app/features/inspector/` — node detail panel

## Frontend Architecture

**Framework**: Angular 18+ with standalone components and Signals for reactivity.

**Project structure**:

```text
frontend/
├── src/app/
│   ├── core/           # ApiService, ErrorInterceptor
│   ├── shared/         # StatusIndicator, LoadingSpinner, Toast, EmptyState
│   ├── features/
│   │   ├── graph/      # GraphContainer, GraphCanvas (D3.js), NodeSearch, Controls
│   │   ├── chat/       # ChatContainer, MessageList, ChatInput (markdown)
│   │   └── inspector/  # InspectorPanel (node details, edges)
│   └── models/         # TypeScript interfaces matching backend schemas
```

**Key features**:

- D3.js force-directed graph with zoom/pan/drag
- node types visually distinguished: concept (large/purple), entity (medium/green), tag (small/yellow)
- real-time graph refresh after mutations
- markdown rendering in chat responses
- scrollable chat and inspector panels
- HTTP caching with configurable TTLs
- error handling with toast notifications
- responsive design (desktop/tablet/mobile)
