# Retrieve Context Enhancement - Implementation Subparts

## Scope
This document reflects the actual work on branch `enhance-retrieve-context`.

It is **not** a file-ingestion implementation plan.
The real branch scope is to enhance `mind-map retrieve` so matched nodes can show directly connected context.

---

## Implemented Subparts

### 1. CLI Surface Update
**File:** `src/mind_map/app/cli/main.py`

Added option:
- `--show-context/--no-context`

Purpose:
- allow callers to include or suppress connected graph context in retrieve output
- default is enabled

Status:
- implemented

---

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

Status:
- implemented

---

### 3. Retrieve Output Enrichment
**File:** `src/mind_map/app/cli/main.py`

Purpose:
- print connected nodes beneath each matched retrieve result

Output example:

```text
### Relevant Context from Mind Map:
- [concept] (Relevance: 1.37): Some matched node
  └─ [tag] (via tagged_as): #SomeTag
  └─ [entity] (via mentions): Some Entity
```

Status:
- implemented

---

### 4. Automated Test Coverage
**File:** `tests/test_graph_store.py`

Purpose:
- validate the new graph context lookup behavior

Covered:
- empty input
- no connections
- single neighbor
- shared neighbors
- filtering internal matched-node links
- bidirectional edges
- nonexistent node handling
- multiple edges to same neighbor
- existing `get_edges` expectations

Status:
- implemented

---

## Real Architecture of This Change

### Retrieval flow
1. query vector store for top matching nodes
2. enrich matched nodes with relation factor
3. optionally fetch connected external neighbors
4. print matched nodes
5. print neighbor context underneath each result

### Important semantic choice
Connected context is:
- **display-time enrichment**, not new retrieval ranking logic
- **one-hop graph expansion**, not multi-hop reasoning
- **external-neighbor only**, not links between already matched nodes

---

## Why this is useful

This improves retrieval in two practical ways:

1. **Better human readability**
   - users can see why a node matters in graph context

2. **Better prompt injection context**
   - downstream systems get a richer text block from one retrieve call

---

## Constraints / Current Limits

1. No neighbor ranking beyond existence of a direct edge
2. No max-neighbor cap
3. No relation filtering
4. No recursive traversal
5. Plain-text output only

---

## Suggested Follow-up Subparts

### A. Neighbor limiting
- add `--max-context-per-node`
- prevent noisy output on highly connected nodes

### B. Relation filtering
- add `--relation-type` include/exclude support
- useful when callers only want tags, entities, or mention links

### C. Structured output mode
- add JSON output for easier downstream machine parsing

### D. Context ranking
- sort connected neighbors by edge weight, node importance, or relation priority

### E. Depth control
- optionally allow depth-2 traversal later
- keep default at depth-1

---

## What should be considered stale

The old ideas below are not part of the current branch implementation:
- document classifier
- LLM chunking engine
- file readers
- ingest command
- relation discovery prompt for file chunks
- KG writer for external file ingestion

If those ideas are revisited later, they should live in a separate plan document with separate scope.

---

## Final Summary

This branch is a focused retrieval enhancement.

Implemented:
- CLI toggle for context display
- graph neighbor lookup helper
- enriched retrieve output
- solid unit tests

Not implemented:
- ingestion pipeline
- LLM chunking/classification
- write-side graph expansion from files
