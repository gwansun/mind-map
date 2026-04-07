# Retrieve Context Enhancement Plan

## Overview
This branch is focused on improving `mind-map retrieve` so retrieval results include immediate graph context, not just top matched nodes.

The current implementation on `enhance-retrieve-context` adds:
- a `--show-context/--no-context` CLI flag
- connected-neighbor expansion for matched nodes
- plain-text output showing related nodes under each result
- tests for the graph context lookup behavior

This file replaces an older stale ingestion plan that did not match the branch.

---

## Goal

When a user runs:

```bash
mind-map retrieve "query"
```

show:
1. the top matched nodes from vector retrieval
2. the directly connected external neighbors for each matched node

This gives richer prompt context for downstream consumers such as OpenClaw prompt injection.

---

## Current Implemented Behavior

### CLI
The retrieve command now supports:

```bash
mind-map retrieve "query" --show-context
mind-map retrieve "query" --no-context
```

Default:
- `--show-context = true`

### Output shape
The output is plain text:

```text
### Relevant Context from Mind Map:
- [concept] (Relevance: 1.37): Some matched node
  └─ [tag] (via tagged_as): #SomeTag
  └─ [entity] (via mentions): Some Entity
```

### Graph expansion rule
For each matched node:
- fetch direct neighbors via graph edges
- include only neighbors **not already in the matched result set**
- show the neighbor type and the edge relation type

This avoids echoing internal matched-node links while still surfacing useful surrounding context.

---

## Implementation Details

### 1. CLI flag
File:
- `src/mind_map/app/cli/main.py`

Add:
- `--show-context/--no-context`

Responsibility:
- controls whether connected neighbors are displayed
- defaults to enabled for richer retrieval output

### 2. GraphStore neighbor lookup
File:
- `src/mind_map/rag/graph_store.py`

Add method:
- `get_connected_context(node_ids: list[str]) -> dict[str, list[tuple[GraphNode, Edge]]]`

Behavior:
- returns direct neighbors for each source node
- treats edges bidirectionally
- excludes neighbors that are already in `node_ids`
- returns empty lists for isolated or nonexistent nodes

### 3. Retrieve output enrichment
Flow:
1. run vector retrieval
2. enrich matched nodes with relation factor
3. optionally fetch connected context for matched node IDs
4. print top-level matched nodes
5. print connected neighbors as indented lines

---

## Why this branch matters

Plain top-k retrieval often loses the graph structure.
This change makes retrieve output much more useful for:
- prompt injection
- user-facing explanations
- inspecting why a result matters
- surfacing tags/entities/concepts linked to a matched node

In short:
- retrieval remains vector-first
- output becomes graph-aware

---

## Included Tests

File:
- `tests/test_graph_store.py`

Covered cases:
- empty node list
- isolated node
- single connection
- shared neighbor across multiple source nodes
- filtering internal connections between matched nodes
- bidirectional edge handling
- nonexistent node IDs
- multiple edges to the same neighbor
- baseline `get_edges()` behavior

---

## Known Limits

1. Only **direct neighbors** are shown
   - no recursive traversal
   - no depth > 1 expansion

2. No ranking of neighbor nodes yet
   - neighbors are included if directly connected
   - there is no secondary scoring or cap per matched node

3. Potential verbosity
   - nodes with many edges may print a lot of context
   - future work may want per-node limits or relation filtering

---

## Recommended Next Steps

### Option A: Keep this branch tight
If the goal is a clean retrieval enhancement only:
1. keep current behavior
2. maybe add a max-neighbor limit
3. maybe add relation-type filtering
4. ship it

### Option B: Improve retrieval quality further
Possible follow-ups:
1. rank connected neighbors by edge weight / type
2. cap neighbor count per node
3. allow `--context-depth` for limited traversal
4. allow `--relation-type` filters
5. add machine-readable JSON output mode

---

## Non-Goals for this branch

The following are **not** part of the actual branch work:
- file ingestion pipeline
- LLM-based document classification
- LLM-based chunking
- KG write-back from external files
- new `mind-map ingest` command

Those belonged to older planning notes and should not be treated as the current branch scope.

---

## Summary

This branch is a focused retrieval UX / prompt-context improvement.

Actual scope:
- vector retrieval results
- direct graph-neighbor expansion
- optional CLI toggle
- tests

Actual intent:
- make `mind-map retrieve` more informative and more useful for prompt injection without changing the underlying retrieval model.
