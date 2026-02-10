# Mind Map Application - Version 2 (Evolution)

> **Note**: This is the implementation plan. For current progress and status, see [PROGRESS.md](PROGRESS.md).

## 📋 Description
An intelligent Mind Map system designed to capture, synthesize, and persist project contexts, conversations, and decisions. It utilizes a Knowledge Graph-inspired RAG (Retrieval Augmented Generation) approach to ensure that LLM interactions are always contextually aware and high-value data is prioritized.

## 🎯 Project Goals

### Primary Objectives
1. **Intelligent Context Management**: Automatically capture and organize project knowledge using LLM-powered processing
2. **Knowledge Graph Architecture**: Build a network of interconnected concepts rather than isolated notes
3. **Relevance-Based Retrieval**: Use importance scoring with time-decay to prioritize current and foundational information
4. **Quality-Focused Ingestion**: Filter out low-value data before it pollutes the knowledge base
5. **Interactive Visualization**: Provide an intuitive graph-based interface for exploring knowledge

### Success Criteria
- System can successfully filter trivial vs. valuable information
- Query responses leverage graph structure for better context
- Importance scores accurately reflect node relevance over time
- Frontend provides clear visualization of knowledge relationships
- CLI and API provide seamless user experience

## 🧠 Advanced Methodology

### 1. Knowledge Graph (KG) Approach
Unlike simple keyword matching, the KG approach treats data as a network of entities and relationships.
- **Node-Link Structure**: Every piece of summarized information is a **Node**. 
- **Tags as Nodes**: Tags are not metadata on a file, but first-class nodes in the graph.
- **Relationships**: When LLM(B) processes data, it identifies multiple related entities. A conversation about "Implementing OAuth in Python" will create links between `#OAuth`, `#Python`, and `#Security`.
- **Retrieval**: When a query is made, we don't just search for a single tag; we traverse the graph to find clusters of related context.

### 2. The Summarization Filter: Keep/Discard Decision
To prevent "database pollution," LLM(B) performs a quality gate check before any ingestion.
- **Filtering Logic**: LLM(B) evaluates data based on:
    - **Entropy/Information Gain**: Does this provide new context or repeat existing knowledge?
    - **Relevance**: Is this trivial (e.g., "Hello," "Thanks") or structural (logic, decisions, facts)?
    - **Clarity**: Is the content interpretable?
- **Workflow**: 
    - LLM(B) receives raw data.
    - LLM(B) outputs a JSON object: `{"action": "keep" | "discard", "reason": "...", "summary": "..."}`.
    - If "discard," the data is ignored to keep the Vector DB lean and relevant.

### 3. Importance with Time-Decay Factor
To ensure the system focuses on currently relevant information while retaining "pivotal" historic facts, an importance score $S$ is calculated.

#### Calculation Formula:
$$S = \left( \frac{C_{node}}{C_{max}} \right) \cdot e^{-\lambda \Delta t}$$

- **$C_{node}$**: Number of connections this specific node/tag has (counted bidirectionally: source OR target).
- **$C_{max}$**: The maximum number of connections any node has in the current DB, counted the same way as $C_{node}$ (bidirectional) to ensure $C_{node}/C_{max} \leq 1$.
- **$e$**: Euler's number (~2.718).
- **$\lambda$ (Lambda)**: Decay constant (e.g., 0.05). A higher value makes the importance drop faster over time.
- **$\Delta t$**: Time elapsed since the last **Interaction** or **Activation** of this node (days/weeks).
- **Effect**: A node remains important if it is *highly connected* (foundational) or *recently used* (active context).

---

## 🛠️ Tech Stack

| Layer | Technology | Details |
| :--- | :--- | :--- |
| **Logic & Orchestration** | **Python** / **LangGraph** | LangGraph handles the cyclic flow (Filter -> KG Update -> RAG -> Respond). |
| **Processing LLM (B)** | **Cloud APIs (auto) / Ollama (Phi-3.5) fallback** | Cloud-first with validated fallback; each provider tested before use. |
| **Reasoning LLM (A)** | **Claude Code / GPT-4o / Gemini** | High-level reasoning for final user output. |
| **Vector Database** | **ChromaDB** | Stores embeddings and graph metadata (node IDs, connection counts). |
| **CLI Deployment** | **Typer** | Provides a modern, fast CLI experience with command autocompletion. |
| **Frontend Deployment** | **Angular** | A robust framework for building a dynamic "Graph Explorer" visualization. |

---

## � Database Schema Design (ChromaDB + Graph Metadata)

To support the **Knowledge Graph** and **Importance Score ($S$)** calculations, we will use two primary collections in ChromaDB or a hybrid structure where "Edges" are managed via metadata or a separate mapping. Below is the proposed schema structure.

### 1. Nodes Collection (`mind_map_nodes`)
Stores the actual content chunks or tags as individual nodes.

| Field | Type | Description |
| :--- | :--- | :--- |
| **`id`** | String (UUID) | Unique identifier for the node. |
| **`embedding`** | Vector[Float] | The vector representation of `content`. |
| **`document`** | String | The actual summarized text or the Tag name (e.g., "#Deployment"). |
| **`metadata`** | JSON Object | Stores the graph attributes and importance metrics. |

**Metadata Structure (`metadata` field):**
```json
{
  "type": "concept" | "tag",        // Distinguishes regular content from structural tags
  "created_at": 1712000000,         // Timestamp of creation
  "last_interaction": 1712500000,   // "Delta t" reference: Updated on query/retrieval
  "connection_count": 5,            // "C_node": Number of edges connected to this node
  "importance_score": 0.85,         // "S": Cached score, re-calc on updates
  "original_source_id": "msg_123"   // Reference to the raw interaction (optional trace)
}
```

### 2. Edges / Relationships
Since ChromaDB is vector-first, explicit graph edges can be stored either as an adjacency list in the Node's metadata or in a separate lightweight collection/table (e.g., SQLite or a separate Chroma collection for "Links").

**Separate "Edges" Registry (Scalable)**
A simple JSON or SQLite lookup to manage pure graph topology without polluting vector metadata.
```json
[
  {
    "source": "node_uuid_1",
    "target": "tag_uuid_#Python",
    "weight": 1.0,
    "relation_type": "tagged_as"
  },
  {
    "source": "node_uuid_1",
    "target": "node_uuid_2",
    "weight": 0.5,
    "relation_type": "related_context"
  }
]
```
---

## �🚀 Implementation Roadmap

### 1. Backend & CLI (Phase 1)
- **Project Initialization**:
    - Set up Python environment (Poetry/Pipenv).
    - Install core dependencies: `langgraph`, `langchain`, `chromadb`, `typer`, `ollama`, `rich`, `pydantic`.
- **Data Layer (Hybrid: ChromaDB + SQLite)**:
    - **Node Storage (ChromaDB)**: Implement `SchemaManager` for content nodes and tags.
    - **Edge Storage (SQLite)**:
        - Initialize a lightweight SQLite DB for the "Edge Registry".
        - Define schema: `source_id`, `target_id`, `weight`, `relation_type`.
    - **GraphStore Wrapper**:
        - Orchestrate dual-writes: `add_node` (Chroma) + `add_edge` (SQLite).
        - Implement `get_subgraph(node_id, depth)` by querying SQLite for connections and then fetching content from Chroma.
        - Handle metadata updates (connection counts) efficiently across both stores.
- **Intelligence Layer (LangGraph)**:
    - **Node 1: FilterAgent (LLM B)**:
        - Implement "Keep/Discard" logic using Ollama (Phi-3.5) with JSON output.
    - **Node 2: KnowledgeProcessor (LLM B)**:
        - Extract entities, generate tags, and summarize content.
    - **Node 3: GraphUpdater**:
        - Logic to update ChromaDB and increment connection counts.
        - **Importance Scorer**: Implement formula $S = (C_{node} / C_{max}) \cdot e^{-\lambda \Delta t}$.
    - **Node 4: ResponseGenerator (LLM A)**:
        - Retrieval of context clusters -> Synthesis of answer.
- **CLI Layer (Typer)**:
    - `init`: Setup local DB and configuration.
    - `memo [text]`: Ingest raw notes/thoughts.
    - `ask [query]`: Query the KG-RAG.
    - `stats`: View graph metrics (top connected nodes) using `rich` tables.

### 2. Frontend Visualization (Phase 2)
- **Environment**:
    - Initialize Angular 17+ Workspace.
    - Install `d3.js` and `@types/d3` for visualization.
- **Backend API Bridge**:
    - Create a lightweight **FastAPI** server (integrated with the CLI tool) to expose:
        - `GET /graph`: Fetch full/partial graph for rendering.
        - `GET /node/{id}`: Detailed view of a node.
        - `POST /ask`: Chat endpoint.
- **Angular Components**:
    - **GraphVisualizer**: D3 Force Directed Graph with zoom/pan and node click handling.
    - **InspectorPanel**: Sidebar to show Node summary, importance score, and tags.
    - **ChatOverlay**: Floating interface to interactions with LLM(A).
- **Integration**:
    - Use Angular Signals for reactive state management of the graph data.

### 3. Deployment & Integration
- **Configuration**:
    - `.env` management for API keys.
    - `config.yaml` for model selection (e.g., swapping Ollama models).
- **Packaging**:
    - Package the Python backend as a pip-installable tool.
    - Bundle the Angular build into the Python package to serve it as a static asset (Single Binary/Command experience).

---

## 🏗️ Architecture Decisions

### Why Hybrid Storage (ChromaDB + SQLite)?
**Decision**: Use ChromaDB for vector embeddings and SQLite for graph edges.

**Rationale**:
- ChromaDB excels at similarity search but lacks native graph traversal
- SQLite provides efficient graph edge queries with minimal overhead
- Hybrid approach leverages strengths of both systems
- Allows independent scaling of vector and graph operations

**Trade-offs**:
- Requires dual-write operations (potential consistency issues)
- Additional complexity in GraphStore wrapper
- Benefit: Significantly faster graph traversal vs. pure vector approach

### Why Two LLMs (A and B)?
**Decision**: Use lightweight LLM(B) for processing, powerful LLM(A) for reasoning.

**Rationale**:
- **LLM(B)** (Phi-3.5): Fast, efficient for repetitive tasks (filtering, tagging)
- **LLM(A)** (Claude/GPT-4): High-quality reasoning for user-facing responses
- Cost optimization: Bulk processing with cheap model, final synthesis with premium model
- Ollama enables local LLM(B) with no API costs

**Trade-offs**:
- More complex configuration and model management
- Benefit: 10-100x cost reduction vs. using premium model for all operations

### Why LangGraph for Orchestration?
**Decision**: Use LangGraph for the ingestion pipeline.

**Rationale**:
- Declarative graph-based workflows match our conceptual model
- Built-in state management and conditional routing
- Easy to visualize and debug pipeline flow
- Native LangChain integration for LLM operations

**Alternative Considered**: Custom pipeline logic
- Rejected: More maintenance, reinventing workflow patterns

### Frontend Technology: Angular + D3.js
**Decision**: Use Angular 17+ with D3.js for graph visualization.

**Rationale**:
- Angular Signals provide reactive state management for dynamic graphs
- D3.js is industry standard for force-directed graph layouts
- TypeScript ensures type safety for complex graph operations
- Mature ecosystem with extensive documentation

**Alternative Considered**: React + Cytoscape.js
- Both viable; Angular chosen for Signals and built-in dependency injection

---

## 🔍 Key Implementation Challenges

### Challenge 1: Maintaining Graph Consistency
**Problem**: Dual-write to ChromaDB and SQLite can lead to inconsistency.

**Proposed Solution**:
1. Wrap operations in transactions where possible
2. Implement reconciliation job to detect/fix inconsistencies
3. Add retry logic with exponential backoff
4. Log all write operations for audit trail

### Challenge 2: Importance Score Calculation Performance
**Problem**: Calculating importance for large graphs can be expensive.

**Proposed Solution**:
1. Cache importance scores in node metadata
2. Recalculate only on updates, not every query
3. Use background job for batch importance updates
4. Index SQLite for fast connection count queries

### Challenge 3: Graph Traversal Depth
**Problem**: Unbounded traversal can retrieve too much irrelevant context.

**Proposed Solution**:
1. Default depth of 2 (configurable)
2. Limit max results per query (e.g., top 10 by importance)
3. Implement "smart pruning" based on edge weights
4. Add breadth-first traversal with early stopping

### Challenge 4: Tag Normalization
**Problem**: "Python", "python", "#Python" should be the same tag.

**Proposed Solution**:
1. Normalize tags to lowercase, remove special chars
2. Use consistent ID format: `tag_{normalized_name}`
3. Store original display format in document field
4. Implement tag merging for duplicates

---

## 📝 Testing Strategy

### Unit Tests
- GraphStore operations (add_node, add_edge, query_similar)
- Importance score calculation
- Tag normalization logic
- FilterAgent decision logic

### Integration Tests
- Full ingestion pipeline (memo → nodes + edges)
- Query flow (ask → retrieval → response)
- API endpoints with mock LLMs
- CLI commands end-to-end

### Performance Tests
- Large graph queries (1000+ nodes)
- Concurrent writes to GraphStore
- Importance recalculation on big graphs
- Frontend rendering with 500+ nodes

---

## Implemented Improvements (2026-02-09)

### 1. Modularized Package Structure
Codebase split into 4 cohesive packages under `src/mind_map/`:
- **`core/`**: `schemas.py` (Pydantic models), `config.py` (load_config) — dependency root
- **`processor/`**: `processing_llm.py`, `filter_agent.py`, `knowledge_processor.py` — LLM-B processing
- **`rag/`**: `graph_store.py`, `reasoning_llm.py`, `response_generator.py`, `llm_status.py` — storage + LLM-A
- **`app/`**: `pipeline.py`, `cli/main.py`, `api/routes.py` — orchestration layer

Removed: `llm.py` facade (consumers import directly), `importance.py` (unused duplicate).

### 2. Processing LLM Validation & Ollama Fallback
Cloud providers (Gemini, Anthropic, OpenAI) are now validated with a test API call during `get_processing_llm()`. If validation fails (e.g., depleted credits, invalid key), the next provider is tried, eventually falling through to Ollama. This ensures `get_processing_llm()` returns a working LLM or None.

### 3. Ask Workflow Fix
Both CLI and API `ask` commands now always call the Reasoning LLM regardless of whether context nodes were found. The flow:
1. Search DB for context (may return empty)
2. Reasoning LLM always generates response (with or without context)
3. Processing LLM extracts Q&A knowledge and stores to graph
4. Q&A nodes linked to context nodes only if context existed (no cross-topic edges)

### 4. Pipeline Resilience
Filter and extraction pipeline nodes fall back to heuristic processing when LLM calls fail at runtime (transient errors). This ensures Q&A data is always stored even if the processing LLM has intermittent failures.

### 5. Importance Score Fix
Fixed `C_max` calculation to count edges bidirectionally (matching `C_node`), preventing importance > 100%. Added `min(score, 1.0)` safety clamp.

### 6. Similarity Threshold
Added `max_distance=0.5` to `query_similar()`. Only nodes with >= 50% cosine similarity are returned. Prevents unrelated topics (e.g., Python and Cat) from being linked via `derived_from` edges.

---

## 🔮 Future Enhancements

### Phase 4: Advanced Features
1. **Graph Pruning**: Automatically remove low-importance nodes
2. **Multi-User Support**: Separate graphs per user with sharing
3. **Export/Import**: JSON/GraphML export for backup/migration
4. **Plugin System**: Custom processors and filters
5. **Semantic Search**: Hybrid keyword + vector search
6. **Time-Travel**: View graph state at any point in history

### Phase 5: Enterprise Features
1. **Authentication & Authorization**: User management, RBAC
2. **Collaboration**: Real-time multi-user editing
3. **Analytics Dashboard**: Usage metrics, popular nodes
4. **API Rate Limiting**: Prevent abuse
5. **Cloud Deployment**: AWS/GCP with autoscaling

---

## 📚 References

### Technical Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Model Library](https://ollama.ai/library)
- [D3.js Force Layout](https://d3js.org/d3-force)

### Related Work
- Personal Knowledge Management (PKM) systems
- Graph-based note-taking (Obsidian, Roam Research)
- RAG architectures and best practices
- Time-decay algorithms in recommendation systems
