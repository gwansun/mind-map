# Mind Map - Implementation Progress

## ✅ Overall Status

**Phase 1: Backend & CLI** - ✅ **COMPLETED**
**Phase 2: Frontend Visualization** - ⏸️ **NOT STARTED**
**Phase 3: Deployment & Integration** - 🔄 **PARTIAL**

---

## 📊 Detailed Progress

### Phase 1: Backend & CLI - ✅ **COMPLETED**

#### Project Setup
- ✅ Poetry environment configured
- ✅ Core dependencies installed:
  - `langgraph`, `langchain`, `chromadb`
  - `typer`, `ollama`, `rich`, `pydantic`
  - `fastapi`, `uvicorn`

#### Data Layer (Hybrid: ChromaDB + SQLite)
- ✅ **Node Storage**: ChromaDB collection `mind_map_nodes` for vector embeddings
- ✅ **Edge Storage**: SQLite database (`data/edges.db`) for graph relationships
- ✅ **GraphStore Wrapper**:
  - Dual-write operations (Chroma + SQLite)
  - `get_subgraph()` for graph traversal
  - Connection count tracking and updates
  - Importance score calculation: $S = (C_{node} / C_{max}) \cdot e^{-\lambda \Delta t}$

#### Intelligence Layer (LangGraph Pipeline)
- ✅ **FilterAgent**: LLM(B)-powered keep/discard decisions
- ✅ **KnowledgeProcessor**: Entity extraction, tag generation, summarization
- ✅ **GraphUpdater**: Dual-write to ChromaDB + SQLite with edge management
- ✅ **ResponseGenerator**: RAG-enhanced response synthesis with LLM(A)

#### CLI Commands (Typer)
- ✅ `init`: Database initialization
- ✅ `init --with-ollama`: Initialize with Ollama configuration
- ✅ `ollama-init`: Pull phi3.5 model
- ✅ `memo "text"`: Ingest notes with LLM processing
- ✅ `memo "text" --no-llm`: Heuristic-only ingestion
- ✅ `ask "query"`: RAG-enhanced queries
- ✅ `stats`: Graph statistics display
- ✅ `serve`: Launch FastAPI server

#### FastAPI Backend
- ✅ `GET /`: API root endpoint
- ✅ `GET /health`: Health check
- ✅ `GET /graph`: Full graph data for visualization
- ✅ `GET /node/{id}`: Detailed node view with edges and importance
- ✅ `POST /ask`: RAG-enhanced query with automatic Q&A ingestion
- ✅ `POST /memo`: Memo ingestion endpoint
- ✅ `GET /stats`: Graph statistics

---

### Phase 2: Frontend Visualization - ⏸️ **NOT STARTED**

#### Pending Items
- ⏸️ Angular 17+ workspace initialization
- ⏸️ D3.js integration for graph visualization
- ⏸️ GraphVisualizer component (force-directed layout)
- ⏸️ InspectorPanel component (node details sidebar)
- ⏸️ ChatOverlay component (LLM interaction interface)
- ⏸️ Angular Signals for reactive state management

#### Completed Items
- ✅ FastAPI backend API (ready for frontend consumption)

---

### Phase 3: Deployment & Integration - 🔄 **PARTIAL**

#### Completed
- ✅ `.env` management for API keys
- ✅ `config.yaml` for model selection (Ollama/OpenAI/Anthropic)

#### Pending
- ⏸️ pip-installable package distribution
- ⏸️ Bundle Angular build as static assets
- ⏸️ Single binary deployment

---

## 🎯 Recent Updates (2026-02-03)

### Fix Claude CLI as Reasoning LLM (LLM-A)
**File**: `src/mind_map/core/reasoning_llm.py`

Three bugs prevented `claude-cli` (the default reasoning provider) from working end-to-end via `mind-map ask`. All three were discovered and fixed in sequence during a single debugging session.

#### Bug 1 — Pipe-mode hang in auth check
**What**: `check_claude_cli_available()` passed the test prompt as a positional CLI argument (`["claude", "-p", "--model", "haiku", "respond with only: OK"]`), but `-p` (pipe mode) reads from stdin. The subprocess hung indefinitely waiting for input, hitting the 30 s timeout every time.
**Fix**: Moved the prompt to the `input=` parameter, matching how `ClaudeCLILLM._generate` already calls the CLI.

#### Bug 2 — `ANTHROPIC_API_KEY` leaking into the claude subprocess
**What**: `llm.py` calls `load_dotenv()` at module import time, which loads `ANTHROPIC_API_KEY` into `os.environ` for the Anthropic SDK fallback. The `claude` CLI subprocess inherits this env and authenticates via the API key (which had zero credits) instead of the local Pro subscription, returning `rc=1 "Credit balance is too low"`.
**Fix**: Added `_claude_cli_env()` helper that copies `os.environ` and strips `ANTHROPIC_API_KEY`. Both `check_claude_cli_available` and `ClaudeCLILLM._generate` now pass `env=_claude_cli_env()` to `subprocess.run`.

#### Bug 3 — Placeholder Gemini key blocked the fallback chain
**What**: `.env` had `GOOGLE_API_KEY=tbd`. `check_gemini_available()` only checks truthiness of the env var, so it passed. `get_gemini_llm()` initialised the LangChain client successfully (key is not validated at init). The crash happened later at `chain.invoke()`, which was uncaught — so the fallback chain never reached Anthropic or OpenAI.
**Fix**: Cleared the placeholder (`GOOGLE_API_KEY=`) in `.env` so Gemini is correctly skipped when no real key is configured.

**Result**: `mind-map ask` now uses Claude CLI via Pro subscription as intended. All 15 unit tests in `tests/test_claude_cli_llm.py` continue to pass.

---

### Full-stack design audit and implementation (2026-02-03, session 2)

A system-wide audit compared every documented design point in CLAUDE.md against the actual code. Four concrete issues were found and fixed. One outdated doc reference was corrected.

#### Fix 1 — Entity nodes and relationship edges were never stored
**Files**: `src/mind_map/models/schemas.py`, `src/mind_map/agents/pipeline.py`

**What**: `KnowledgeProcessor` (LLM-B) correctly extracted entities and relationships from text, but the storage step in the LangGraph pipeline (`create_storage_node`) had a `pass` stub where entity nodes and relationship edges should have been written. Entities were silently discarded after extraction.

**Fix**:
- Added `NodeType.ENTITY = "entity"` to the schema enum.
- Replaced the stub with full storage logic: entity nodes are created with deterministic IDs (`entity_{name}`) for deduplication (same pattern as tags), linked to their parent concept via `mentions` edges, and relationship edges (e.g. `include`, `consist of`) are written between entity pairs.

#### Fix 2 — `POST /memo` API endpoint was a stub
**File**: `src/mind_map/api/routes.py`

**What**: The endpoint returned a hardcoded `"Memo ingestion not yet implemented"` response. The CLI `memo` command already ran the full FilterAgent → KnowledgeProcessor → GraphStore pipeline; the API endpoint did not.

**Fix**: Wired `POST /memo` to call `ingest_memo()` with `get_processing_llm()`, identical to the CLI path. Returns `status`, `message`, and created `node_ids`.

#### Fix 3 — `get_stats` lumped entity nodes into concept count
**Files**: `src/mind_map/core/graph_store.py`, `src/mind_map/cli/main.py`

**What**: `get_stats()` computed `concept_count = total - tag_count`. With entity nodes now in the graph, entities were silently counted as concepts. The `stats` CLI display had no entity row.

**Fix**: `get_stats()` now counts `entity_nodes` explicitly and subtracts both tags and entities from the total for `concept_count`. The `stats` table displays Concept / Entity / Tag as three distinct rows.

#### Fix 4 — LLM(B) Q&A ingestion added to CLI `ask` command
**File**: `src/mind_map/cli/main.py`

**What**: The API `POST /ask` already ran LLM(B) on the Q&A pair after LLM(A) responded (extract tags, summarise, store, create `derived_from` edges). The CLI `ask` command did not — it printed the response and stopped.

**Fix**: Added the same post-response block to the CLI `ask` command: get LLM(B) via `get_ollama_llm`, run `ingest_memo` on the formatted Q&A, create `derived_from` edges from the new concept to every context node that was used. Falls back to heuristic extraction if Ollama is unavailable.

#### Fix 5 — `ExtractionResult.relationships` validator
**File**: `src/mind_map/models/schemas.py`

**What**: LLM(B) (Ollama phi3.5) occasionally returned relationships with 2 elements instead of the required `[source, relation, target]` 3-tuple, crashing Pydantic validation and aborting the entire ingestion.

**Fix**: Added a `field_validator` on `relationships` (mode `before`) that silently drops any entry that is not exactly a 3-element list. Valid tuples pass through unchanged.

#### Doc corrections — CLAUDE.md
- `ResponseGenerator (LLM-A/Gemini)` → `ResponseGenerator (LLM-A/Claude CLI)` — Gemini was never the default.
- `POST /memo` endpoint entry changed from `(TODO)` to `Ingest memo via LangGraph pipeline`.
- `ask` data flow diagram updated to show the Q&A feedback loop: after LLM(A) responds, the Q&A pair is pushed through the full ingestion pipeline to enrich the graph for future queries.

**Result after all fixes — graph state: 29 nodes / 41 edges**
- 8 concept nodes (LLM(A) Q&A summaries + original memos)
- 6 entity nodes (extracted by LLM(B): "dietary changes", "wet food consumption", "hydration methods", etc.)
- 15 tag nodes (#CatNutrition, #CatHydration, #CatHealth, #HydrationTips, …)
- 5 edge types active: `tagged_as`, `derived_from`, `mentions`, plus LLM(B)-extracted semantic relations (`include`, `consist of`, `aim to improve`)

---

## 🎯 Recent Updates (2026-02-01)

### Enhanced `/ask` Endpoint
**File**: `src/mind_map/api/routes.py:153-188`

Implemented automatic knowledge graph enrichment from user queries:

1. **Importance Score Updates**: Updates `last_interaction` timestamp for all context nodes used in responses
2. **Q&A Ingestion**: Processes query-response pairs through LLM(B) pipeline
   - Extracts relevant tags
   - Generates summaries
   - Creates new concept nodes
3. **Provenance Tracking**: Creates "derived_from" edges linking Q&A nodes to source context
4. **Self-Expansion**: Knowledge graph grows organically through user interactions

**Impact**: Every query now enriches the knowledge base, making future queries more accurate.

---

## 🔧 Known Issues

### 1. ~~CLI LLM Configuration Hardcoding~~ — ✅ RESOLVED (2026-02-03)
Replaced by `get_reasoning_llm()` multi-provider fallback chain.  Three additional bugs in the Claude CLI path (pipe-mode hang, `ANTHROPIC_API_KEY` env leakage, placeholder Gemini key blocking fallbacks) were also fixed.  See "Recent Updates (2026-02-03)" above.

### 2. ChromaDB Telemetry Warnings
**Issue**: Harmless telemetry errors appear in logs.
**Impact**: None (functionality not affected).
**Priority**: Low

---

## 📋 Next Steps

### Immediate Priorities
1. **Add Integration Tests**
   - Test `/ask` endpoint with Q&A ingestion
   - Verify "derived_from" edge creation
   - Test importance score updates

2. **Documentation**
   - API documentation (OpenAPI/Swagger)
   - Usage examples and tutorials
   - Architecture diagram

### Medium-Term Goals
3. **Frontend Development**
   - Initialize Angular workspace
   - Implement D3.js graph visualization
   - Create interactive node inspector

4. **Enhanced Features**
   - Graph pruning (remove low-importance nodes)
   - Export/import functionality
   - Multi-user support

### Long-Term Vision
5. **Deployment**
   - Package as pip-installable tool
   - Docker containerization
   - Cloud deployment options

---

## 📈 Metrics

**As of 2026-02-03:**
- Total Nodes: 29
- Total Edges: 41
- Concept Nodes: 8
- Entity Nodes: 6
- Tag Nodes: 15
- Average Connections: 2.83

**Edge types in graph:**
- `tagged_as` (17) — concept → tag
- `derived_from` (15) — Q&A concept → context nodes used by LLM(A)
- `mentions` (6) — concept → entity nodes it references
- Semantic relations (3) — entity → entity (`include`, `consist of`, `aim to improve`)

**Tags in Graph:**
- `tag_catnutrition` (4 connections)
- `tag_cathydration`, `tag_dietarychangesforcats` (3 connections each)
- `tag_api`, `tag_python`, `tag_preferenceassessment`, `tag_test` (2 connections each)
- `tag_auth`, `tag_database`, `tag_environmentalfactors`, `tag_waterfountainforcats`, `tag_multiplebowls`, `tag_cathealth`, `tag_nutritionforcats`, `tag_hydrationtips` (1 connection each)
