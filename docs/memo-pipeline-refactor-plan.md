# Historical Plan: Memo Pipeline Refactor

## Status

**Archived design record**

This file captures the refactor plan that led to the current retrieve-first memo pipeline. It is retained for history, but it is **not** the primary source of truth.

## Current reality

The current implementation has already applied the core pipeline refactor:

```text
retrieve -> filter -> extract -> store
```

Current behavior includes:
- retrieval runs before filtering
- duplicate memos skip extraction and storage
- extraction is focused on new memo text plus limited grounded references
- storage handles deterministic graph linking
- strict memo CLI ingestion is separated from legacy/internal ingestion
- strict memo CLI ingestion requires an explicit target and rejects on explicit-target failure

Current source-of-truth files:
- `src/mind_map/app/pipeline.py`
- `src/mind_map/app/cli/main.py`
- `src/mind_map/processor/filter_agent.py`
- `src/mind_map/processor/knowledge_processor.py`
- `README.md`
- `PROGRESS.md`

## Why this file still exists

This document is preserved as a historical planning artifact showing the intended memo-pipeline restructuring that was later implemented.

## Historical summary

The original refactor plan aimed to:
- move duplicate detection ahead of extraction
- keep extraction focused on new memo text
- remove model-driven `existing_links` behavior from extraction
- use retrieval context in a narrower, more grounded way
- make storage responsible for deterministic linking behavior

## Important clarification

If you are trying to understand or modify the current system, do **not** rely on this document alone.
Read the source files and active top-level documentation instead.
