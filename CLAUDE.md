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
poetry run mind-map memo "text"       # Ingest a note (uses LLM if available)
poetry run mind-map memo "text" --no-llm  # Ingest with heuristic only
poetry run mind-map ask "query"       # Query the knowledge graph
poetry run mind-map stats             # View graph statistics
poetry run mind-map serve             # Start FastAPI server

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

**LangGraph Pipeline**:
1. `FilterAgent` (LLM-B) → Keep/discard decision based on information gain
2. `KnowledgeProcessor` (LLM-B) → Extract entities, tags, relationships, summarize
3. `GraphStore` → Dual-write to ChromaDB (vectors) + SQLite (edges)
4. `ResponseGenerator` (LLM-A/Claude CLI) → RAG-enhanced response synthesis

**LLM Configuration**:

| Role | Provider | Default Model | Purpose |
|------|----------|---------------|---------|
| Processing (LLM-B) | Cloud APIs (auto) / Ollama fallback | gemini-2.0-flash | Filtering, extraction, summarization |
| Reasoning (LLM-A) | Claude CLI / Antigravity-OAuth / Cloud APIs | sonnet | Response generation |

- **Processing (LLM-B)**: Cloud-first with validated fallback to Ollama
  - Provider priority (`auto`): Gemini → Anthropic → OpenAI → Ollama
  - Each cloud provider is validated with a test API call before use; if validation fails (e.g., depleted credits), the next provider is tried
  - Cloud models: `gemini-2.0-flash`, `claude-haiku-4-5-20251001`, `gpt-4o-mini`
  - Ollama recommended: `phi3.5`, `phi3`, `llama3.2`, `mistral`, `gemma2:2b`, `qwen2.5:3b`
  - Config: `processing_llm.provider` in `config.yaml` (`auto`|`gemini`|`anthropic`|`openai`|`ollama`)
  - Auto-pull (Ollama): disabled by default (enable in `config.yaml`)

- **Reasoning (LLM-A)**: Claude CLI (default) with Antigravity-OAuth and cloud fallbacks
  - Priority: Claude CLI → Antigravity-OAuth → Gemini → Anthropic Claude → OpenAI GPT
  - Default: `claude-cli` (uses Claude Pro subscription, no API costs)
  - Antigravity-OAuth: Uses Google Antigravity IDE OAuth credentials to access `gemini-3-flash`
  - Requires: `claude` CLI installed and authenticated, OR Antigravity credentials, OR API key in `.env`

**Importance Score**: `S = (C_node / C_max) * e^(-λ * Δt)` - balances connectivity with time decay.
- `C_node` and `C_max` are both counted bidirectionally (source OR target) for consistency
- Score is clamped to `[0.0, 1.0]`

**Similarity Threshold**: `query_similar()` uses `max_distance=0.5` (cosine) to filter irrelevant results. Only nodes with >= 50% similarity are returned as context, preventing cross-topic edge pollution.

**Relation Factor**: Context nodes are weighted by edge density to the most query-relevant node.
Combined score: `importance * (1 + relation_factor)` where `relation_factor = edges_between(anchor, node) / total_edges(anchor)`.

## CLI Commands

### Main Commands
| Command | Description |
|---------|-------------|
| `init` | Initialize database and configuration |
| `memo TEXT` | Ingest a note into the knowledge graph |
| `ask QUERY` | Query with RAG-enhanced response |
| `stats` | Display knowledge graph statistics |
| `serve` | Start FastAPI server |
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
| POST | `/memo` | Ingest memo via LangGraph pipeline |
| GET | `/openclaw/manifest` | OpenClaw tool manifest |
| POST | `/openclaw/invoke` | OpenClaw unified tool invocation |

## Key Files

### Core (shared types & config)
- `src/mind_map/core/schemas.py` - Pydantic models (GraphNode, Edge, FilterDecision, ExtractionResult)
- `src/mind_map/core/config.py` - Configuration loader (config.yaml, .env)

### Processor (LLM-B)
- `src/mind_map/processor/processing_llm.py` - Multi-provider processing LLM: Cloud APIs + Ollama
- `src/mind_map/processor/filter_agent.py` - FilterAgent for keep/discard decisions
- `src/mind_map/processor/knowledge_processor.py` - KnowledgeProcessor for entity extraction

### RAG (storage & reasoning)
- `src/mind_map/rag/graph_store.py` - Hybrid ChromaDB + SQLite storage
- `src/mind_map/rag/reasoning_llm.py` - Multi-provider reasoning LLM: Claude CLI, Antigravity-OAuth, Gemini, Anthropic, OpenAI
- `src/mind_map/rag/response_generator.py` - ResponseGenerator (LLM-A) for RAG synthesis
- `src/mind_map/rag/llm_status.py` - Health/status checks for both LLM providers

### App (orchestration, CLI, API)
- `src/mind_map/app/pipeline.py` - LangGraph ingestion pipeline
- `src/mind_map/app/cli/main.py` - Typer CLI entry point with all commands
- `src/mind_map/app/api/routes.py` - FastAPI endpoints for frontend
- `src/mind_map/app/openclaw/tool_manifest.py` - OpenClaw tool manifest definition
- `src/mind_map/app/openclaw/plugin.py` - OpenClaw plugin registration logic

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
  provider: auto          # auto | gemini | anthropic | openai | ollama
  model: phi3.5           # Ollama model (used when provider is ollama or auto-fallback)
  temperature: 0.1
  auto_pull: false        # Auto-download Ollama models if not available

reasoning_llm:
  provider: claude-cli    # Options: claude-cli, antigravity, gemini, anthropic, openai
  model: sonnet           # For claude-cli: sonnet, opus, haiku
  temperature: 0.7
  timeout: 120            # CLI timeout in seconds

openclaw:
  enabled: false
  base_url: http://localhost:3000   # OpenClaw server URL
  api_key: ""                       # OpenClaw API key for registration
```

### .env (for cloud providers)
```
GOOGLE_API_KEY=your-key              # Gemini API (processing + reasoning fallback)
ANTHROPIC_API_KEY=your-key           # Claude API (processing + reasoning fallback)
OPENAI_API_KEY=your-key              # GPT API (processing + reasoning fallback)
ANTIGRAVITY_CLIENT_ID=your-id        # Antigravity-OAuth (gemini-3-flash via Antigravity IDE)
ANTIGRAVITY_CLIENT_SECRET=your-secret  # Antigravity-OAuth client secret
# Note: Claude CLI uses your Claude Pro subscription - no API key needed
```

### Claude CLI Setup
```bash
# Install Claude CLI
npm install -g @anthropic-ai/claude-code

# Authenticate
claude login
```

## Storage

- **ChromaDB** (`data/chroma/`): Vector embeddings + node metadata
- **SQLite** (`data/edges.db`): Edge registry with source, target, weight, relation_type

## Data Flow

### Ingestion (memo command)
```
Text Input → FilterAgent → KnowledgeProcessor → GraphStore
                ↓                  ↓                ↓
           keep/discard    tags, entities,    ChromaDB + SQLite
                           relationships
```

### Query (ask command)
```
Query → ChromaDB Search (max_distance=0.5) → Enrich (relation factor) → ResponseGenerator → Response
              ↓                                       ↓                         ↓                ↓
        Similar nodes                          Re-sort by combined        RAG synthesis    Q&A pair → FilterAgent → KnowledgeProcessor → GraphStore
        (filtered by                           importance + edge          (always runs,          ↓                  ↓                ↓
         similarity threshold)                 density to anchor           even if no        keep/discard    tags, entities,    ChromaDB + SQLite
                                                                           context found)                   relationships      (enriches future queries)
```
- Reasoning LLM always generates a response, even for new topics with no context
- Processing LLM pipeline has heuristic fallback if LLM calls fail at runtime
- Q&A nodes are only linked to context nodes if context was found (no cross-topic edges)

## Frontend Architecture

**Framework**: Angular 18+ with standalone components and Signals for reactivity.

**Project Structure**:
```
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
- Scrollable chat and inspector panels (flex chain with `min-height: 0` at each level)
- HTTP caching with configurable TTLs
- Error handling with toast notifications
- Responsive design (desktop/tablet/mobile)
