# Implementation Subparts - Retrieve, Memo Pipeline, and Packaging Findings

## Scope
This document now reflects three concrete implementation threads that were completed in the repo:

1. retrieve-context enrichment
2. memo pipeline duplicate-handling refactor
3. deployment/runtime packaging findings for the OpenClaw-used `mind-map` CLI

It is not a generic roadmap. It is a practical record of what actually changed and what was learned.

---

## Part 1. Retrieve Context Enhancement

### 1. CLI Surface Update
**File:** `src/mind_map/app/cli/main.py`

Added options:
- `--show-context/--no-context`
- `--max-context-per-node`

Purpose:
- allow callers to include or suppress connected graph context in retrieve output
- cap context verbosity per matched result
- default to prompt-friendly output

Status:
- implemented

### 2. Connected Context Lookup in GraphStore
**File:** `src/mind_map/rag/graph_store.py`

Added method:
- `get_connected_context(node_ids)`

Purpose:
- for each matched node, return direct neighbors and the connecting edge

Behavior:
- returns `node_id -> [(neighbor_node, edge), ...]`
- excludes neighbors that are also in the matched node set
- supports bidirectional edge discovery
- returns empty lists for isolated/nonexistent nodes
- sorts results deterministically by weight/importance/relation/document/id

Status:
- implemented

### 3. Retrieve Output Enrichment
**File:** `src/mind_map/app/cli/main.py`

Purpose:
- print connected nodes beneath each matched retrieve result
- reduce ambiguity that indented nodes are strict child nodes

Status:
- implemented

### 4. Retrieve Test Coverage
**Files:**
- `tests/test_graph_store.py`
- `tests/test_cli_retrieve.py`

Covered:
- empty input
- no connections
- single neighbor
- shared neighbors
- filtering internal matched-node links
- bidirectional edges
- nonexistent node handling
- multiple edges to same neighbor
- deterministic ordering
- retrieve output control flags

Status:
- implemented

---

## Part 2. Memo Pipeline Refactor

### 1. Pipeline Reordering
**File:** `src/mind_map/app/pipeline.py`

Implemented flow:
- `retrieve -> filter -> extract -> store`

Previous flow:
- `filter -> retrieve -> extract -> store`

Status:
- implemented

### 2. Retrieval Expansion for Memo Flow
**Files:**
- `src/mind_map/app/pipeline.py`
- `src/mind_map/rag/graph_store.py`
- `src/mind_map/core/schemas.py`

Implemented:
- retrieve top similar concept candidates
- expand first-hop neighbors
- include only:
  - `entity`
  - `tag`
- exclude neighboring concepts
- store retrieval context separately by role

Status:
- implemented

### 3. Filter Decision Simplification
**Files:**
- `src/mind_map/core/schemas.py`
- `src/mind_map/app/pipeline.py`

Implemented decision set:
- `discard`
- `duplicate`
- `new`

Behavior:
- `duplicate` skips extraction and storage
- `discard` ends early
- `new` proceeds

Status:
- implemented

### 4. Extraction Contract Cleanup
**Files:**
- `src/mind_map/processor/knowledge_processor.py`
- `src/mind_map/core/schemas.py`

Implemented:
- extraction uses new text only
- retrieved concept content removed from extraction input
- retrieved entity/tag references allowed as lightweight grounding hints
- `existing_links` removed from extraction contract

Status:
- implemented

### 5. Storage Cleanup
**File:** `src/mind_map/app/pipeline.py`

Implemented:
- create concept node only for `new`
- reuse tag/entity nodes by deterministic ID normalization
- create concept -> tag, concept -> entity, entity -> entity edges
- link new concepts to retrieved concepts with deterministic `related_context`
- no merge/update-existing path

Status:
- implemented

### 6. Memo Pipeline Test Coverage
**Files:**
- `tests/test_pipeline.py`
- `tests/test_graph_store.py`
- `tests/test_api_routes.py`

Covered:
- retrieval context presence
- first-hop neighbor logic
- duplicate short-circuiting
- standalone entity persistence
- deterministic relationship creation
- API-adjacent duplicate behavior

Status:
- implemented

---

## Part 3. Filter Fallback Chain

### 1. Filter-Specific LLM Chain
**Files:**
- `src/mind_map/processor/filter_agent.py`
- `src/mind_map/processor/processing_llm.py`
- `src/mind_map/app/pipeline.py`
- `src/mind_map/app/api/routes.py`
- `src/mind_map/app/cli/main.py`

Implemented filter chain:
1. Ollama `phi3.5`
2. OpenClaw MiniMax CLI
3. heuristic fallback in pipeline

Important scope choice:
- cloud-provider routing removed from **filter path only**
- broader processing/extraction provider routing left intact

Status:
- implemented

### 2. FilterAgent Backend Changes
**File:** `src/mind_map/processor/filter_agent.py`

Implemented:
- `FilterAgentWithFallback`
- MiniMax CLI helper
- prompt contract shared across phi3.5 and MiniMax

Status:
- implemented

### 3. Filter LLM Selection
**File:** `src/mind_map/processor/processing_llm.py`

Implemented:
- `get_filter_llm()`
- phi3.5-only filter LLM path
- no cloud-auto routing for filter stage

Status:
- implemented

### 4. Filter Fallback Tests
**File:** `tests/test_filter_fallback.py`

Covered:
- phi3.5 primary success
- MiniMax fallback on failure
- safe fallback when both backends fail
- integration with pipeline `filter_llm`
- no cloud-auto filter routing

Status:
- implemented

---

## Part 4. Packaging and Runtime Findings

### 1. OpenClaw Execution Path
OpenClaw resolves and executes:
- `which mind-map`
- current path on this machine:
  - `~/.local/bin/mind-map`

That local executable launches the uv-managed tool runtime.

### 2. Critical Packaging Mismatch Discovered
A major issue was found during live duplicate testing.

Observed behavior:
- repo source contained the new filter fallback-chain code
- built wheel and sdist also contained the new code
- but the installed uv-tool runtime still loaded older `site-packages` content

This was proven by comparing:
- repo `src/mind_map/...`
- installed `~/.local/share/uv/tools/mind-map/lib/python3.11/site-packages/mind_map/...`

The stale installed runtime lacked:
- `get_filter_llm`
- `FilterAgentWithFallback`
- `filter_llm` integration in pipeline/CLI/API

### 3. Reliable Installation Procedure
The reliable fix was:

```bash
./.venv/bin/python -m build
uv tool uninstall mind-map
uv tool install dist/mind_map-0.1.0-py3-none-any.whl
```

This succeeded where direct repo-path installation was unreliable.

After wheel-based install:
- installed runtime symbols matched repo source
- live duplicate memo test correctly returned `Skipped duplicate`

### 4. Operational Lesson
For this project on this machine:
- prefer **wheel-based install** over direct repo-path `uv tool install`
- verify installed `site-packages` when debugging runtime mismatches
- do not assume a successful install means the live runtime matches repo source

---

## Part 5. Backend Runtime Operations

### Backend rebuild and restart
The backend was rebuilt and restarted successfully.

Runtime:
- uvicorn serving `mind_map.app.api.routes:app`
- host: `127.0.0.1`
- port: `8000`

Health verification:
- `/stats` responded successfully after restart

---

## Final Summary

The work is no longer just “retrieve enhancement”.

The repo now contains:
- retrieve-time graph context enrichment
- retrieve-first memo pipeline with duplicate short-circuiting
- filter-specific fallback chain: phi3.5 -> MiniMax -> heuristic
- validated packaging lesson: install from built wheel for reliable OpenClaw-used runtime updates

The most important practical lesson from this cycle:
- **live CLI validation matters**
- the logic can be correct in repo and tests, but still fail in the installed runtime if packaging/install behavior is stale
