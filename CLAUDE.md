# Mind Map

Knowledge Graph-based Mind Map with LangGraph orchestration for intelligent context management.

## Build & Run

```bash
# Install dependencies
poetry install

# Run CLI
poetry run mind-map help              # Show comprehensive help
poetry run mind-map init              # Initialize database
poetry run mind-map init --with-ollama  # Initialize with Ollama model
poetry run mind-map init --data-dir /path/to/db  # Initialize a custom database path
poetry run mind-map memo "text"       # Ingest a note (uses retrieval-augmented extraction if LLM available)
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
npm start                             # Start dev server (http://localhost:4200)
npm run build                         # Production build
```

## Architecture

**Knowledge Graph approach**: Data stored as nodes (concepts/tags/entities) with edges (relationships). Retrieval traverses graph clusters rather than simple keyword matching.

**LangGraph Ingestion Pipeline**:
1. `FilterAgent` (LLM-B) → Keep/discard decision based on information gain
2. `Similarity Retrieval` → Query similar existing nodes before extraction
3. `KnowledgeProcessor` (LLM-B) → Extract summary, tags, entities, relationships, and links to existing retrieved nodes
4. `GraphStore` → Dual-write to ChromaDB (vectors) + SQLite (edges)

**Memo ingestion is now retrieval-augmented**:
- `mind-map memo` no longer extracts from only the new memo text
- it first retrieves top similar existing nodes from the graph
- those retrieved nodes are passed to the extractor with their IDs and documents
- extractor output can include `existing_links`, which are validated against retrieved IDs before storage
- standalone extracted entities are persisted even if they do not appear in relationship tuples

### Memo Extraction Chain

Processing for memo extraction follows this order:

1. **OpenClaw MiniMax primary path**
   - uses local `openclaw agent --agents minimax --message "..."`
   - prompt includes the new memo and retrieved existing node records
2. **Configured processing LLM fallback**
   - commonly Ollama `phi3.5`
3. **Heuristic fallback**
   - used when both model-backed paths fail

This structure improves grounded linking while keeping ingestion resilient.

**LLM Configuration**:

| Role | Provider | Default Model | Purpose |
|------|----------|---------------|---------|
| Processing (general LLM-B) | Cloud APIs (auto) / Ollama fallback | gemini-2.0-flash | Filtering, extraction, summarization |
| Memo extraction primary | OpenClaw agent | minimax | Retrieval-grounded memo ingestion |
| Memo extraction fallback | Ollama / configured processing model | phi3.5 | Structured extraction fallback |
| Reasoning (LLM-A) | OpenClaw Agent / Claude CLI / Cloud APIs | main | Response generation |

- **Processing (general LLM-B)**: Cloud-first with validated fallback to Ollama
  - Provider priority (`auto`): Gemini → Anthropic → OpenAI → Ollama
  - Each cloud provider is validated with a test API call before use; if validation fails, the next provider is tried
  - Cloud models: `gemini-2.0-flash`, `claude-haiku-4-5-20250901`, `gpt-4o-mini`
  - Ollama recommended: `phi3.5`, `phi3`, `llama3.2`, `mistral`, `gemma2:2b`, `qwen2.5:3b`
  - Config: `processing_llm.provider` in `config.yaml` (`auto`|`gemini`|`anthropic`|`openai`|`ollama`)
  - Auto-pull (Ollama): disabled by default

- **Reasoning (LLM-A)**: OpenClaw Agent (default) with Claude CLI and cloud fallbacks
  - Priority: OpenClaw Agent → Claude CLI → Gemini → Anthropic Claude → OpenAI GPT
  - Default: `openclaw-agent`

**Importance Score**: `S = (C_node / C_max) * e^(-λ * Δt)`
- `C_node` and `C_max` are both counted bidirectionally (source OR target)
- Score is clamped to `[0.0, 1.0]`

**Similarity Threshold**:
- `query_similar()` uses `max_distance=0.5` (cosine) to filter irrelevant results
- memo ingestion retrieval also uses a conservative threshold so only reasonably relevant existing nodes are passed into extraction

**Relation Factor**: Context nodes are weighted by edge density to the most query-relevant node.
Combined score: `importance * (1 + relation_factor)` where `relation_factor = edges_between(anchor, node) / total_edges(anchor)`.

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

## MCP Tools

The MCP server (`src/mind_map/mcp/server.py`) exposes the following tools via FastMCP. All tools accept an optional `workspace_id` for multi-workspace isolation.

| Tool | Description |
|------|-------------|
| `mind_map_retrieve` | Similarity search with relation-factor enrichment |
| `mind_map_memo` | Ingest text through the retrieval-augmented LangGraph pipeline |
| `mind_map_stats` | Knowledge graph statistics |
| `mind_map_report` | JSON report with summary stats and top-5 nodes |
| `mind_map_prune` | Prune the least important 10% of nodes |
| `mind_map_health` | System health check (Ollama, databases, ProcessingLLM, integration tests) |

### Prune Algorithm (`mind_map_prune`)
1. Calculate importance for all concept/entity nodes (tags excluded as direct candidates)
2. Sort ascending, take bottom `max(1, floor(total * 0.1))`
3. Tags removed only if **all** their edges connect to nodes in the prune set (shared tags preserved)
4. Delete edges, single-connected tags, then prune target nodes
5. Returns JSON: `{ deleted_nodes, deleted_tags, deleted_edges_count, summary }`

## Key Files

### Core (shared types & config)
- `src/mind_map/core/schemas.py` - Pydantic models including `ExistingLink` and extraction schema
- `src/mind_map/core/config.py` - Configuration loader (`config.yaml`, `.env`)

### Processor (LLM-B)
- `src/mind_map/processor/processing_llm.py` - Multi-provider processing LLM: Cloud APIs + Ollama
- `src/mind_map/processor/filter_agent.py` - FilterAgent for keep/discard decisions
- `src/mind_map/processor/knowledge_processor.py` - KnowledgeProcessor for retrieval-aware extraction with OpenClaw MiniMax primary path

### RAG (storage & reasoning)
- `src/mind_map/rag/graph_store.py` - Hybrid ChromaDB + SQLite storage
- `src/mind_map/rag/reasoning_llm.py` - Multi-provider reasoning LLM: Claude CLI, Gemini, Anthropic, OpenAI
- `src/mind_map/rag/response_generator.py` - ResponseGenerator (LLM-A) for RAG synthesis
- `src/mind_map/rag/llm_status.py` - Health/status checks for both LLM providers

### MCP (Model Context Protocol server)
- `src/mind_map/mcp/server.py` - FastMCP server with retrieve, memo, stats, report, prune tools

### App (orchestration, CLI, API)
- `src/mind_map/app/pipeline.py` - LangGraph ingestion pipeline with retrieval before extraction
- `src/mind_map/app/cli/main.py` - Typer CLI entry point with all commands
- `src/mind_map/app/api/routes.py` - FastAPI endpoints for frontend

### Frontend (Angular 18+)
- `frontend/src/app/app.component.ts` - Main layout with three-panel design
- `frontend/src/app/core/api.service.ts` - HTTP client with caching
- `frontend/src/app/features/graph/` - D3.js graph visualization
- `frontend/src/app/features/chat/` - Chat interface with markdown support
- `frontend/src/app/features/inspector/` - Node detail panel

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

### .env
```bash
GOOGLE_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
```

## Storage

- **ChromaDB** (`data/chroma/` by default): Vector embeddings + node metadata
- **SQLite** (`data/edges.db` by default): Edge registry with source, target, weight, relation_type
- Override the default storage root with `--data-dir /path/to/db` on supported CLI commands

## Data Flow

### Ingestion (`memo` command)
```text
Text Input
  → FilterAgent
  → Similarity Retrieval (top relevant existing nodes)
  → KnowledgeProcessor
      - OpenClaw MiniMax primary
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
- Reasoning LLM always generates a response, even for new topics with no context
- Processing LLM pipeline has heuristic fallback if LLM calls fail at runtime
- `ask` still uses its current linking flow and is separate from the new memo-ingestion `existing_links` behavior

## Frontend Architecture

**Framework**: Angular 18+ with standalone components and Signals for reactivity.

**Project Structure**:
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

**Key Features**:
- D3.js force-directed graph with zoom/pan/drag
- Node types visually distinguished: Concept (large/purple), Entity (medium/green), Tag (small/yellow)
- Real-time graph refresh after mutations
- Markdown rendering in chat responses
- Scrollable chat and inspector panels
- HTTP caching with configurable TTLs
- Error handling with toast notifications
- Responsive design (desktop/tablet/mobile)
