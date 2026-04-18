# Historical Plan: Filter Agent Ollama -> OpenClaw MiniMax Fallback

## Status

**Archived / outdated design note**

This document describes an intermediate fallback-chain design that is **not the current source of truth** for memo CLI behavior.

## Current reality

As of the current implementation:
- strict memo CLI ingestion uses an **explicit resolved target**
- memo CLI filtering/extraction do **not** use an implicit fallback chain when an explicit target is provided
- if the explicit target fails, the memo is rejected
- legacy/internal ingestion paths may still use separate internal helpers, but that is distinct from the strict memo CLI path

Current source-of-truth files:
- `src/mind_map/app/pipeline.py`
- `src/mind_map/app/cli/main.py`
- `src/mind_map/processor/filter_agent.py`
- `src/mind_map/processor/knowledge_processor.py`
- `src/mind_map/processor/cli_executor.py`
- `README.md`
- `PROGRESS.md`

## Why this file still exists

This file is preserved as historical implementation context for a prior approach where filter behavior was designed around a fallback sequence such as:
1. local Ollama `phi3.5`
2. OpenClaw MiniMax CLI fallback
3. heuristic fallback

That design was useful during refactoring, but it should not be read as the current contract for strict memo CLI ingestion.

## Historical summary

The older plan aimed to:
- remove generic cloud-auto routing from the filter path
- prefer local filter execution first
- fall back to OpenClaw MiniMax for filtering
- fall back again to heuristic logic when model paths failed

## Important clarification

If you are trying to understand the current system, do **not** implement from this document.
Use the source files and top-level docs listed above instead.
