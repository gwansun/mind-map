# Retrieve Context Enhancement - Implementation Subparts

## Scope
This document reflects the actual work on branch `enhance-retrieve-context` and the follow-up packaging/debugging findings after merge.

It is **not** a file-ingestion implementation plan.
The real branch scope is to enhance `mind-map retrieve` so matched nodes can show directly connected context in a way that is suitable for OpenClaw prompt injection.

---

## Implemented Subparts

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
- sorts results deterministically by weight/importance/relation/document/id

Status:
- implemented

---

### 3. Retrieve Output Enrichment
**File:** `src/mind_map/app/cli/main.py`

Purpose:
- print connected nodes beneath each matched retrieve result
- reduce ambiguity that indented nodes are strict child nodes

Output example:

```text
### Relevant Context from Mind Map:
- [concept] (Relevance: 1.37): Some matched node
  └─ related [tag] via tagged_as: #SomeTag
  └─ related [entity] via mentions: Some Entity
```

Status:
- implemented

---

### 4. Automated GraphStore Test Coverage
**File:** `tests/test_graph_store.py`

Purpose:
- validate the graph context lookup behavior and ordering

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
- existing `get_edges` expectations

Status:
- implemented

---

### 5. CLI Retrieve Test Coverage
**File:** `tests/test_cli_retrieve.py`

Purpose:
- verify actual end-user retrieve behavior

Covered:
- default retrieve shows context
- `--no-context` suppresses related lines
- `--max-context-per-node` caps visible neighbors

Status:
- implemented

Verified:

```bash
./.venv/bin/pytest -q tests/test_graph_store.py tests/test_cli_retrieve.py
# 14 passed
```

---

## Real Architecture of This Change

### Retrieval flow
1. query vector store for top matching nodes
2. enrich matched nodes with relation factor
3. optionally fetch connected external neighbors
4. sort connected neighbors deterministically
5. print matched nodes
6. print bounded neighbor context underneath each result

### Important semantic choice
Connected context is:
- **display-time enrichment**, not new retrieval ranking logic
- **one-hop graph expansion**, not multi-hop reasoning
- **external-neighbor only**, not links between already matched nodes
- **bounded for prompt usability**, not an unbounded graph dump

---

## OpenClaw Integration Findings

### What OpenClaw actually executes
OpenClaw code uses PATH resolution:
- `which mind-map`
- `execFileSync("mind-map", ...)`

So the actual binary in use is whatever `mind-map` resolves to in the running environment.

### Actual resolved path on this machine
- `~/.local/bin/mind-map`
- symlink → `~/.local/share/uv/tools/mind-map/bin/mind-map`

### Launcher linkage
The uv-managed launcher imports:
- `mind_map.app.cli.main:app`

So the real execution chain is:
1. OpenClaw → `mind-map`
2. PATH symlink → uv tool launcher
3. uv launcher → installed `site-packages/mind_map/app/cli/main.py`

### Packaging issue discovered
Installing via direct repo path:

```bash
uv tool install --force /Users/gwansun/Desktop/projects/mind-map
```

resulted in a uv-managed install whose `site-packages` contents were older than the repo source.

This was proven by diffing:
- repo `src/mind_map/app/cli/main.py`
- installed uv tool `site-packages/mind_map/app/cli/main.py`

Observed mismatch:
- repo had `--max-context-per-node` and updated wording
- installed tool still lacked the new option and used older output formatting

### Reliable installation fix
The successful procedure was:

```bash
./.venv/bin/python -m build
uv tool uninstall mind-map
uv tool install ./dist/mind_map-0.1.0-py3-none-any.whl
```

Result:
- installed `site-packages` now matched repo source
- installed CLI accepted `--max-context-per-node`
- OpenClaw-used binary became current

### Operational lesson
For this project, when updating the OpenClaw-used mind-map CLI:
- prefer **fresh wheel install** over direct repo-path `uv tool install`
- verify both:
  - PATH resolution
  - installed uv tool `site-packages` contents

---

## Why this is useful

This improves retrieval in three practical ways:

1. **Better human readability**
   - users can see why a node matters in graph context

2. **Better prompt injection context**
   - downstream systems get a richer and more bounded text block from one retrieve call

3. **More reliable deployment behavior**
   - the actual OpenClaw-used binary path and installation method are now understood and documented

---

## Constraints / Current Limits

1. No relation filtering yet
2. No recursive traversal
3. No JSON output mode yet
4. Multiple edges to the same neighbor are still displayed separately

---

## Suggested Follow-up Subparts

### A. Relation filtering
- add `--relation-type` include/exclude support

### B. Structured output mode
- add JSON output for easier downstream machine parsing

### C. Duplicate-neighbor compaction
- optionally collapse multiple relations for the same neighbor into one summarized line

### D. Depth control
- optionally allow depth-2 traversal later
- keep default at depth-1

### E. Release/install note
- document wheel-based install as the preferred OpenClaw deployment path for this tool

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

This branch is a focused retrieval enhancement plus a packaging/deployment investigation.

Implemented:
- CLI toggle for context display
- context cap per matched node
- graph neighbor lookup helper
- deterministic ordering
- enriched retrieve output
- GraphStore tests
- retrieve CLI tests
- verified OpenClaw binary path and wheel-based installation fix

Not implemented:
- ingestion pipeline
- LLM chunking/classification
- write-side graph expansion from files
