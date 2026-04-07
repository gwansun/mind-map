# Retrieve Context Enhancement Plan

## Overview
This branch is focused on improving `mind-map retrieve` so retrieval results include immediate graph context, not just top matched nodes.

The implemented and verified scope now includes:
- a `--show-context/--no-context` CLI flag
- a `--max-context-per-node` CLI flag
- connected-neighbor expansion for matched nodes
- deterministic connected-neighbor ordering
- plain-text output showing related nodes under each result
- unit tests for graph context lookup
- CLI tests for retrieve behavior
- packaging/install findings for the OpenClaw-used binary

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
3. a bounded, deterministic amount of neighbor context suitable for OpenClaw prompt injection

This gives richer prompt context for downstream consumers such as OpenClaw prompt injection while keeping output readable.

---

## Current Implemented Behavior

### CLI
The retrieve command now supports:

```bash
mind-map retrieve "query" --show-context
mind-map retrieve "query" --no-context
mind-map retrieve "query" --max-context-per-node 3
```

Defaults:
- `--show-context = true`
- `--max-context-per-node = 3`

### Output shape
The output is plain text:

```text
### Relevant Context from Mind Map:
- [concept] (Relevance: 1.37): Some matched node
  └─ related [tag] via tagged_as: #SomeTag
  └─ related [entity] via mentions: Some Entity
```

### Graph expansion rule
For each matched node:
- fetch direct neighbors via graph edges
- include only neighbors **not already in the matched result set**
- sort connected neighbors deterministically
- cap displayed neighbors per matched node unless unlimited mode is requested
- show the neighbor type and the edge relation type

This avoids echoing internal matched-node links while still surfacing useful surrounding context.

---

## Implementation Details

### 1. CLI flags
File:
- `src/mind_map/app/cli/main.py`

Adds:
- `--show-context/--no-context`
- `--max-context-per-node`

Responsibilities:
- control whether connected neighbors are displayed
- limit verbosity of connected-context output
- default to prompt-friendly behavior

### 2. GraphStore neighbor lookup
File:
- `src/mind_map/rag/graph_store.py`

Method:
- `get_connected_context(node_ids: list[str]) -> dict[str, list[tuple[GraphNode, Edge]]]`

Behavior:
- returns direct neighbors for each source node
- treats edges bidirectionally
- excludes neighbors already present in `node_ids`
- returns empty lists for isolated or nonexistent nodes
- sorts results deterministically by:
  1. edge weight descending
  2. neighbor importance score descending
  3. relation type ascending
  4. neighbor document ascending
  5. neighbor id ascending

### 3. Retrieve output enrichment
Flow:
1. run vector retrieval
2. enrich matched nodes with relation factor
3. optionally fetch connected context for matched node IDs
4. print top-level matched nodes
5. print connected neighbors as indented related-node lines
6. cap neighbor lines per node using `--max-context-per-node`

---

## Why this branch matters

Plain top-k retrieval often loses graph structure.
This change makes retrieve output much more useful for:
- prompt injection
- user-facing explanations
- inspecting why a result matters
- surfacing tags/entities/concepts linked to a matched node
- reducing ambiguity between “related node” and “child/subnode” interpretation

In short:
- retrieval remains vector-first
- output becomes graph-aware
- output is now more bounded and deterministic

---

## Tests Included

Files:
- `tests/test_graph_store.py`
- `tests/test_cli_retrieve.py`

### GraphStore coverage
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
- deterministic connected-context ordering

### CLI coverage
Covered cases:
- default retrieve shows context
- `--no-context` suppresses related lines
- `--max-context-per-node` limits displayed neighbors correctly

Verified run in repo venv:

```bash
./.venv/bin/pytest -q tests/test_graph_store.py tests/test_cli_retrieve.py
# 14 passed
```

---

## Packaging / OpenClaw Integration Findings

### What OpenClaw uses
OpenClaw does **not** hardcode a special mind-map binary path.
It resolves `mind-map` from PATH via:
- `which mind-map`
- `execFileSync("mind-map", ...)`

### Active OpenClaw-used path
Resolved path:
- `~/.local/bin/mind-map`

Symlink target:
- `~/.local/share/uv/tools/mind-map/bin/mind-map`

Launcher behavior:
- imports `mind_map.app.cli.main:app`

### Important packaging issue found
Installing with:

```bash
uv tool install --force /path/to/repo
```

left the uv-managed installed package **stale** relative to repo source, even though the repo `main` contained newer CLI code.

Observed symptom:
- repo source had `--max-context-per-node`
- installed uv tool environment still rejected that option

Direct inspection showed the installed file under:
- `~/.local/share/uv/tools/mind-map/lib/python3.11/site-packages/mind_map/app/cli/main.py`

was older than the repo source file.

### Successful fix
The working fix was:
1. build a fresh wheel from the repo
2. uninstall the uv tool
3. install from the built wheel artifact

Example:

```bash
./.venv/bin/python -m build
uv tool uninstall mind-map
uv tool install ./dist/mind_map-0.1.0-py3-none-any.whl
```

After that, the installed OpenClaw-used binary correctly exposed:
- `--show-context`
- `--no-context`
- `--max-context-per-node`

### Operational takeaway
For this repo/tooling setup:
- **fresh wheel install is more reliable than direct repo-path tool install**
- when verifying OpenClaw behavior, inspect both:
  - PATH resolution
  - installed uv tool site-packages contents

---

## Known Limits

1. Only **direct neighbors** are shown
   - no recursive traversal
   - no depth > 1 expansion

2. No relation filtering yet
   - all direct neighbor relation types may appear

3. Multiple edges to the same neighbor are still shown separately
   - acceptable for now
   - could be compacted later if prompt size becomes a concern

4. Plain-text output only
   - no JSON mode yet

---

## Recommended Next Steps

### Option A: Keep this branch tight
If the goal is a clean retrieval enhancement only:
1. keep current behavior
2. maybe add relation-type filtering
3. maybe add structured JSON output
4. ship and observe prompt quality

### Option B: Improve retrieval quality further
Possible follow-ups:
1. relation-type include/exclude filters
2. machine-readable JSON output mode
3. limited `--context-depth` support
4. compact duplicate-neighbor display across multiple relation types

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
- bounded connected-context output
- deterministic ordering
- GraphStore tests
- retrieve CLI tests
- verified OpenClaw binary path + install behavior

Actual intent:
- make `mind-map retrieve` more informative and more useful for prompt injection without changing the underlying retrieval model.
