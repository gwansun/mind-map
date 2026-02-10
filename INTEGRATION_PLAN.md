# Mind Map → OpenClaw Integration Plan (Plan B: Component Embedding)

## 1. Executive Summary

**Goal**: Embed mind-map's knowledge graph capabilities directly into OpenClaw as a Python library dependency — no separate server, no frontend, no additional reasoning LLM.

**Approach**: Component-based integration. Cherry-pick the processing pipeline and RAG retrieval modules from mind-map and wire them into OpenClaw's existing architecture.

| Provided by mind-map | Provided by OpenClaw |
|---|---|
| Processing LLM pipeline (LLM-B) — filtering, extraction, summarization | Reasoning LLM (LLM-A) — response generation |
| Structured knowledge extraction (tags, entities, relationships) | User-facing API and frontend |
| ChromaDB vector storage + SQLite edge registry | Orchestration and routing |
| RAG retrieval with importance scoring and relation factors | Authentication, session management |
| Heuristic fallback when LLM is unavailable | Configuration and deployment |

---

## 2. Component Map

```
mind-map package structure
==========================

┌─────────────────────────────────────────────────────────────────────┐
│  INCLUDED in integration                                            │
│                                                                     │
│  core/                                                              │
│  ├── schemas.py          Pydantic models: GraphNode, Edge,          │
│  │                       FilterDecision, ExtractionResult,          │
│  │                       QueryResult, NodeType, NodeMetadata        │
│  └── config.py           load_config() → dict[str, Any]            │
│                                                                     │
│  processor/                                                         │
│  ├── processing_llm.py   get_processing_llm() → LangChain LLM      │
│  │                       Cloud-first: Gemini→Anthropic→OpenAI→Ollama│
│  ├── filter_agent.py     FilterAgent.evaluate(text) → FilterDecision│
│  └── knowledge_processor.py  KnowledgeProcessor.extract(text)       │
│                              → ExtractionResult                     │
│                                                                     │
│  rag/                                                               │
│  └── graph_store.py      GraphStore: ChromaDB + SQLite hybrid       │
│                           query_similar(), enrich_context_nodes(),   │
│                           calculate_importance(), add_node/edge      │
│                                                                     │
│  app/                                                               │
│  └── pipeline.py         ingest_memo(), build_ingestion_pipeline()  │
│                          PipelineState, LangGraph orchestration      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  EXCLUDED from integration                                          │
│                                                                     │
│  rag/reasoning_llm.py        OpenClaw has its own reasoning LLM     │
│  rag/response_generator.py   OpenClaw handles response generation   │
│  rag/llm_status.py           Health checks for excluded LLMs        │
│  app/cli/                    CLI interface not needed                │
│  app/api/                    FastAPI server not needed               │
│  app/openclaw/               OpenClaw plugin (circular dependency)   │
│  frontend/                   Angular UI not needed                   │
└─────────────────────────────────────────────────────────────────────┘
```

**External dependencies (required)**:
- `chromadb` — vector storage
- `langchain-core` — prompt templates, output parsers
- `langgraph` — pipeline orchestration
- `pydantic` — data models
- `pyyaml` — config loading
- `python-dotenv` — env var loading
- `ollama` — Ollama client (only if using Ollama provider)

**External dependencies (conditional, for cloud providers)**:
- `langchain-google-genai` — Gemini provider
- `langchain-anthropic` — Anthropic provider
- `langchain-openai` — OpenAI provider

---

## 3. Integration Interface: Ingestion Pipeline

### Entry Point

```python
from mind_map.app.pipeline import ingest_memo
from mind_map.rag.graph_store import GraphStore
from mind_map.processor.processing_llm import get_processing_llm
from pathlib import Path

# Initialize once
store = GraphStore(data_dir=Path("data"))
store.initialize()

llm = get_processing_llm()  # Returns LangChain LLM or None

# Ingest a memo
success, message, node_ids = ingest_memo(
    text="Python's GIL prevents true parallelism in CPU-bound threads",
    store=store,
    llm=llm,            # Pass None for heuristic-only mode
    source_id=None,      # Optional: link back to source document
)
# success=True, message="Created 5 nodes", node_ids=["uuid-1", "tag_python", ...]
```

### Pipeline Stages

```
Text Input
    │
    ▼
┌──────────┐   action="discard"
│  Filter   │──────────────────────► END (success=False, "Discarded: {reason}")
│  Agent    │
└──────────┘
    │ action="keep"
    ▼
┌──────────┐
│ Knowledge │
│ Processor │
└──────────┘
    │
    ▼
┌──────────┐
│  Graph    │
│  Store    │
└──────────┘
    │
    ▼
END (success=True, "Created N nodes", [node_ids])
```

**Stage 1 — FilterAgent** (`filter_agent.py`):
- Input: raw text
- Output: `FilterDecision(action="keep"|"discard", reason="...", summary="...")`
- Evaluates information gain, relevance, and clarity
- Heuristic fallback (no LLM): discards text < 10 chars or trivial patterns (`"hello"`, `"thanks"`, `"ok"`, etc.)

**Stage 2 — KnowledgeProcessor** (`knowledge_processor.py`):
- Input: filtered text (or summary from FilterDecision)
- Output: `ExtractionResult(summary, tags, entities, relationships)`
- `relationships` are `(source, relation, target)` tuples
- Heuristic fallback (no LLM): extracts hashtags, capitalized words, keyword-matched tags

**Stage 3 — GraphStore** (`graph_store.py`):
- Creates a concept node (UUID) with the summary
- Creates tag nodes (`tag_{name}`) and entity nodes (`entity_{name}`)
- Creates edges: concept→tag (`tagged_as`), concept→entity (`mentions`), entity→entity (extracted relation)
- Deduplicates tags/entities by ID convention

### Data Structures

```python
# FilterDecision — output of Stage 1
class FilterDecision(BaseModel):
    action: Literal["keep", "discard"]
    reason: str
    summary: str | None = None

# ExtractionResult — output of Stage 2
class ExtractionResult(BaseModel):
    summary: str
    tags: list[str] = []                          # e.g., ["#Python", "#Concurrency"]
    entities: list[str] = []                       # e.g., ["GIL", "CPU-bound threads"]
    relationships: list[tuple[str, str, str]] = [] # e.g., [("GIL", "prevents", "parallelism")]

# GraphNode — stored in ChromaDB
class GraphNode(BaseModel):
    id: str
    document: str
    metadata: NodeMetadata
    embedding: list[float] | None = None
    relation_factor: float | None = None  # Set at query time

# Edge — stored in SQLite
class Edge(BaseModel):
    source: str
    target: str
    weight: float = 1.0
    relation_type: str = "related_context"  # tagged_as, mentions, or custom
```

### Return Values

| Scenario | Return |
|---|---|
| Success | `(True, "Created {n} nodes", [node_id1, node_id2, ...])` |
| Discarded by filter | `(False, "Discarded: {reason}", [])` |
| Pipeline error | `(False, "{error_message}", [])` |

---

## 4. Integration Interface: RAG Retrieval

### Similarity Search

```python
from mind_map.rag.graph_store import GraphStore
from pathlib import Path

store = GraphStore(data_dir=Path("data"))

# Query by text — ChromaDB handles embedding internally
nodes = store.query_similar(
    query="How does Python handle concurrency?",
    n_results=10,
    max_distance=0.5,  # Cosine distance threshold (0=identical, 2=opposite)
)
# Returns: list[GraphNode] sorted by relevance
# Only nodes with cosine distance <= 0.5 are returned (>= 50% similarity)
```

Each returned `GraphNode` has `metadata.importance_score` set to `max(0.0, 1 - distance)`, clamped to `[0.0, 1.0]`.

### Context Enrichment with Relation Factor

```python
# Enrich with edge-density scoring
enriched_nodes = store.enrich_context_nodes(nodes)

# Each node now has relation_factor set
# Combined score: importance_score * (1 + relation_factor)
# Nodes are re-sorted by combined score
for node in enriched_nodes:
    combined = node.metadata.importance_score * (1 + (node.relation_factor or 0))
    print(f"{node.id}: importance={node.metadata.importance_score:.2f}, "
          f"relation_factor={node.relation_factor:.2f}, combined={combined:.2f}")
```

**How `enrich_context_nodes()` works**:
1. Takes the first node (highest similarity) as the **anchor**
2. For each other node, calculates: `relation_factor = edges_between(anchor, node) / total_edges(anchor)`
3. Re-sorts all nodes by `importance_score * (1 + relation_factor)`
4. Nodes with more edges to the anchor get boosted (up to 2x)

### Importance Score Formula

```
S = (C_node / C_max) * e^(-λ * Δt)
```

| Symbol | Meaning | Default |
|---|---|---|
| `C_node` | Bidirectional edge count for this node | — |
| `C_max` | Maximum bidirectional edge count across all nodes | — |
| `λ` | Time decay factor | `0.05` |
| `Δt` | Days since last interaction | — |
| `S` | Final score, clamped to `[0.0, 1.0]` | — |

```python
# Recalculate importance for a specific node
score = store.calculate_importance(
    node_id="entity_python",
    lambda_decay=0.05,
    time_unit_days=1.0,
)
```

### Minimal Retrieval for OpenClaw

```python
from mind_map.rag.graph_store import GraphStore
from pathlib import Path

def retrieve_context(query: str, data_dir: Path = Path("data")) -> str:
    """Retrieve relevant context from the knowledge graph for OpenClaw's reasoning LLM.

    Returns a formatted context string, or empty string if no relevant nodes found.
    """
    store = GraphStore(data_dir=data_dir)

    # 1. Similarity search (filtered by max_distance=0.5)
    nodes = store.query_similar(query, n_results=10, max_distance=0.5)
    if not nodes:
        return ""

    # 2. Enrich with relation factors and re-sort
    nodes = store.enrich_context_nodes(nodes)

    # 3. Format context for the reasoning LLM
    context_parts = []
    for node in nodes:
        combined = node.metadata.importance_score * (1 + (node.relation_factor or 0))
        context_parts.append(
            f"[{node.metadata.type.value}] (score={combined:.2f}) {node.document}"
        )

    return "\n".join(context_parts)
```

---

## 5. Processing LLM Provider Details

### Auto-Detection Chain

`get_processing_llm()` resolves providers in this order (when `provider: auto`):

| Priority | Provider | Model | Env Var Required | Validation |
|---|---|---|---|---|
| 1 | Gemini | `gemini-2.0-flash` | `GOOGLE_API_KEY` | Test API call |
| 2 | Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | Test API call |
| 3 | OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` | Test API call |
| 4 | Ollama | `phi3.5` (configurable) | None (local) | Server check |

Each cloud provider is **validated with a real test API call** before use. If validation fails (invalid key, depleted credits, network error), the next provider is tried.

### Configuration

In `config.yaml`:
```yaml
processing_llm:
  provider: auto    # auto | gemini | anthropic | openai | ollama
  model: phi3.5     # Ollama model name (used for ollama provider or auto-fallback)
  temperature: 0.1
  auto_pull: false  # Auto-download Ollama models if missing
```

### Three Options for OpenClaw

**Option A — Use the auto chain as-is** (recommended):
```python
llm = get_processing_llm()  # Resolves automatically based on config + env vars
```

**Option B — Provide OpenClaw's own LLM**:
```python
from mind_map.app.pipeline import ingest_memo

# Use any LangChain-compatible LLM
from langchain_openai import ChatOpenAI
custom_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

success, msg, ids = ingest_memo(text="...", store=store, llm=custom_llm)
```

**Option C — Heuristic-only (no LLM)**:
```python
success, msg, ids = ingest_memo(text="...", store=store, llm=None)
# Uses regex-based extraction: hashtags, capitalized words, keyword matching
```

---

## 6. Storage Architecture

### ChromaDB (Vector Storage)

- **Path**: `{data_dir}/chroma/`
- **Collection**: `mind_map_nodes`
- **Distance metric**: Cosine (`{"hnsw:space": "cosine"}`)
- **Stored per node**: `id`, `document` (text), `embedding` (auto-generated by ChromaDB), `metadata` (type, timestamps, scores)

### SQLite (Edge Registry)

- **Path**: `{data_dir}/edges.db`

```sql
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    relation_type TEXT DEFAULT 'related_context',
    created_at REAL NOT NULL,
    UNIQUE(source, target, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
```

### Node ID Conventions

| Node Type | ID Pattern | Example |
|---|---|---|
| Concept | UUID v4 | `a1b2c3d4-e5f6-...` |
| Tag | `tag_{lowercase}` | `tag_python`, `tag_concurrency` |
| Entity | `entity_{lowercase}` | `entity_gil`, `entity_cpu_bound_threads` |

### Edge Types

| Relation | Source → Target | Created By |
|---|---|---|
| `tagged_as` | Concept → Tag | Storage node (always) |
| `mentions` | Concept → Entity | Storage node (always) |
| `derived_from` | Q&A Concept → Context Node | Q&A feedback loop |
| Custom (e.g., `"prevents"`) | Entity → Entity | LLM extraction |

---

## 7. Q&A Feedback Loop (Optional Pattern)

In mind-map's ask workflow, every question-answer pair is fed back through the ingestion pipeline. This enriches the knowledge graph over time — answers become retrievable context for future queries.

### How It Works

1. User asks a question → reasoning LLM generates a response
2. The Q&A pair is formatted as: `"Q: {question}\nA: {answer}"`
3. This text is ingested through `ingest_memo()` (filter → extract → store)
4. If the original query found context nodes, `derived_from` edges link the Q&A concept to those context nodes
5. Future queries can now retrieve both the original knowledge and the synthesized Q&A

### OpenClaw Implementation

```python
from mind_map.app.pipeline import ingest_memo
from mind_map.rag.graph_store import GraphStore
from mind_map.core.schemas import Edge

def store_qa_pair(
    question: str,
    answer: str,
    store: GraphStore,
    llm=None,
    context_node_ids: list[str] | None = None,
) -> list[str]:
    """Store a Q&A pair in the knowledge graph and link to context.

    Args:
        question: The user's question
        answer: The generated answer
        store: GraphStore instance
        llm: Optional processing LLM (None for heuristic)
        context_node_ids: Node IDs that were used as context for the answer

    Returns:
        List of created node IDs
    """
    qa_text = f"Q: {question}\nA: {answer}"
    success, message, node_ids = ingest_memo(text=qa_text, store=store, llm=llm)

    if not success or not node_ids or not context_node_ids:
        return node_ids

    # Link Q&A concept node to original context nodes
    qa_concept_id = node_ids[0]  # First node is always the concept
    for ctx_id in context_node_ids:
        store.add_edge(Edge(
            source=qa_concept_id,
            target=ctx_id,
            relation_type="derived_from",
            weight=0.8,
        ))

    return node_ids
```

---

## 8. Excluded Components & Rationale

| File | Reason for Exclusion |
|---|---|
| `rag/reasoning_llm.py` | OpenClaw has its own reasoning LLM; including this would create a redundant provider chain |
| `rag/response_generator.py` | Response synthesis is OpenClaw's responsibility; this module depends on reasoning_llm.py |
| `rag/llm_status.py` | Health checks for reasoning LLM and combined status — not relevant without reasoning_llm.py |
| `app/cli/main.py` | Typer CLI interface; OpenClaw has its own user interface |
| `app/api/routes.py` | FastAPI server; OpenClaw has its own API layer |
| `app/openclaw/` | OpenClaw plugin adapter — would create a circular dependency |
| `frontend/` | Angular UI; OpenClaw has its own frontend |

---

## 9. Contingency Plan

### 9.1 ChromaDB Initialization Failure

| | |
|---|---|
| **Symptoms** | `ValueError` or `RuntimeError` on `store.initialize()`, missing `data/chroma/` directory |
| **Causes** | Filesystem permissions, disk space, incompatible chromadb version |
| **Mitigation** | Ensure `data_dir` is writable before calling `initialize()`. Pin `chromadb` version to match mind-map's `pyproject.toml`. |
| **Fallback** | Wrap `GraphStore.initialize()` in try/except. If ChromaDB fails, log the error and disable knowledge graph features. Ingestion and retrieval return empty results gracefully. |

### 9.2 Embedding Model Unavailable

| | |
|---|---|
| **Symptoms** | `query_similar()` returns empty results or raises on first call |
| **Causes** | ChromaDB's default embedding function requires `onnxruntime` or a network-accessible model |
| **Mitigation** | Ensure `onnxruntime` is in OpenClaw's dependencies. ChromaDB uses `all-MiniLM-L6-v2` by default — verify it downloads on first use. |
| **Fallback** | Pre-warm embeddings at startup: call `store.query_similar("test", n_results=1)` during initialization. If it fails, disable RAG retrieval and fall back to OpenClaw's native context. |

### 9.3 SQLite Lock / Corruption

| | |
|---|---|
| **Symptoms** | `sqlite3.OperationalError: database is locked` or `database disk image is malformed` |
| **Causes** | Concurrent writes from multiple processes, ungraceful shutdown |
| **Mitigation** | Use `WAL` journal mode (`PRAGMA journal_mode=WAL`). Ensure only one writer process accesses `edges.db`. |
| **Fallback** | If locked: retry with exponential backoff (3 attempts, 100ms/500ms/2s). If corrupted: delete `edges.db` and reinitialize — edge data is supplementary and will rebuild over time from new ingestions. ChromaDB nodes remain intact. |

### 9.4 Processing LLM Unavailable

| | |
|---|---|
| **Symptoms** | `get_processing_llm()` returns `None` |
| **Causes** | No API keys configured, no Ollama server running, all cloud providers failed validation |
| **Mitigation** | The pipeline is designed to handle `llm=None`. Pass `None` to `ingest_memo()`. |
| **Fallback** | Heuristic mode activates automatically. FilterAgent keeps all non-trivial text (> 10 chars, not a greeting). KnowledgeProcessor extracts hashtags, capitalized words, and keyword-matched tags. Quality is lower but the system remains functional. |

### 9.5 LLM Call Failure at Runtime

| | |
|---|---|
| **Symptoms** | `FilterAgent.evaluate()` or `KnowledgeProcessor.extract()` raises an exception mid-pipeline |
| **Causes** | Network timeout, rate limit, API error after initial validation passed |
| **Mitigation** | Both `FilterAgent` and `KnowledgeProcessor` catch exceptions internally and fall through to heuristic logic within the same pipeline run. |
| **Fallback** | The pipeline nodes (`create_filter_node`, `create_extraction_node`) wrap LLM calls in try/except blocks. On failure, they produce heuristic results and continue — the pipeline does not abort. |

### 9.6 Import / Version Conflicts in OpenClaw

| | |
|---|---|
| **Symptoms** | `ImportError`, `ModuleNotFoundError`, or version mismatch errors on import |
| **Causes** | Conflicting versions of shared dependencies (`pydantic`, `langchain-core`, `chromadb`) |
| **Mitigation** | Pin compatible version ranges. Key constraints: `pydantic>=2.0`, `langchain-core>=0.3`, `chromadb>=0.5`. Test imports in a fresh virtualenv before deploying. |
| **Fallback** | If version conflicts are unresolvable, isolate mind-map in a subprocess: spawn a lightweight worker that communicates via stdin/stdout JSON. This eliminates all dependency conflicts at the cost of IPC overhead. |

### 9.7 Rollback Strategy

If integration proves unworkable:

1. **Feature flag**: Wrap all mind-map calls behind a config flag (`knowledge_graph.enabled: false`). Disabling the flag returns OpenClaw to its pre-integration behavior with zero code changes.

2. **Facade pattern**: All mind-map interactions should go through a single facade class in OpenClaw's codebase (e.g., `KnowledgeGraphFacade`). To roll back, replace the facade implementation with a no-op stub.

3. **Clean removal**: `pip uninstall mind-map` removes the package. Delete the `data/` directory to remove stored data. No database migrations or schema changes to revert.

---

## 10. Integration Checklist

- [ ] Add `mind-map` as a dependency in OpenClaw's `pyproject.toml` (or `requirements.txt`)
- [ ] Install required extras: `pip install mind-map[chromadb,langgraph]`
- [ ] Create `data/` directory with write permissions for ChromaDB + SQLite
- [ ] Copy/adapt `config.yaml` with processing_llm settings (see Section 11)
- [ ] Set environment variables for desired cloud provider (see Section 11)
- [ ] Initialize storage: `store = GraphStore(data_dir); store.initialize()`
- [ ] Verify embedding model downloads: `store.query_similar("test", n_results=1)`
- [ ] Initialize processing LLM: `llm = get_processing_llm()`
- [ ] Test ingestion: `ingest_memo("test knowledge", store, llm)`
- [ ] Test retrieval: `store.query_similar("test", max_distance=0.5)`
- [ ] Implement facade class wrapping `ingest_memo()` and `retrieve_context()`
- [ ] Add feature flag for knowledge graph (`knowledge_graph.enabled`)
- [ ] Wire facade into OpenClaw's ingestion path (when user sends data)
- [ ] Wire facade into OpenClaw's retrieval path (before reasoning LLM)
- [ ] (Optional) Implement Q&A feedback loop (Section 7)
- [ ] Run integration tests in staging environment
- [ ] Monitor ChromaDB disk usage and SQLite WAL size in production

---

## 11. Minimal Configuration

### config.yaml (trimmed for OpenClaw)

```yaml
processing_llm:
  provider: auto          # auto | gemini | anthropic | openai | ollama
  model: phi3.5           # Ollama model (used when provider is ollama or auto-fallback)
  temperature: 0.1
  auto_pull: false
```

### .env (only the providers you need)

```bash
# Set ONE OR MORE of these for cloud processing (checked in order):
GOOGLE_API_KEY=your-gemini-key         # Priority 1: Gemini
ANTHROPIC_API_KEY=your-anthropic-key   # Priority 2: Anthropic
OPENAI_API_KEY=your-openai-key         # Priority 3: OpenAI
# If none are set, falls back to Ollama (requires local Ollama server)
# If Ollama is also unavailable, falls back to heuristic mode (no LLM)
```

**Not needed** (excluded components):
- `reasoning_llm` section in config.yaml
- `openclaw` section in config.yaml
- `ANTIGRAVITY_CLIENT_ID` / `ANTIGRAVITY_CLIENT_SECRET` env vars
- Claude CLI installation
