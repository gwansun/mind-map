# Memo Pipeline Refactor Plan

## Goal

Refactor the memo ingestion flow to prevent duplicate concepts by moving duplicate detection ahead of extraction, keeping extraction focused on new text, and using retrieved first-hop entity/tag context only as lightweight reference material.

## Target Flow

1. `retrieve`
2. `filter`
3. `extract`
4. `store`

Branching behavior:
- `discard` -> end
- `duplicate` -> end
- `new` -> `extract -> store`

If a memo is classified as `duplicate`, extraction is skipped.

---

## 1. Refactor the memo pipeline flow

- Change the ingestion order to:
  - `retrieve`
  - `filter`
  - `extract`
  - `store`
- Remove the current `filter -> retrieve -> extract -> store` ordering
- Update pipeline branching so only `new` proceeds to extraction
- If filter returns `discard` or `duplicate`, end immediately

## 2. Refactor retrieval stage

- Retrieve top-k similar existing nodes from the graph
- Treat retrieved **concept** nodes as duplicate-check candidates
- Expand each retrieved concept by one hop
- Collect only first-hop neighbors of type:
  - `entity`
  - `tag`
- Exclude neighboring concepts from expansion
- Deduplicate expanded entity/tag reference nodes by ID
- Keep retrieval outputs separated by role:
  - retrieved concept candidates
  - retrieved entity references
  - retrieved tag references

## 3. Redesign pipeline state

Add or refactor state fields so they separately track:
- raw input text
- retrieved concept candidates
- retrieved entity references
- retrieved tag references
- filter decision
- extraction result
- created node IDs
- error state

Remove the old assumption that a single retrieved node list is used for both duplicate detection and extraction grounding.

## 4. Simplify filter decisions

Replace current keep/discard behavior with:
- `discard`
- `duplicate`
- `new`

Filter should use:
- the new memo text
- retrieved concept candidates

Filter should decide whether the new memo is:
- trivial/noisy and should be discarded
- already represented and should be skipped as duplicate
- genuinely new and should proceed

## 5. Skip extraction for duplicates

If filter returns `duplicate`:
- do not run extraction
- do not create any nodes or edges
- end the pipeline immediately
- return a clear duplicate result message from `ingest_memo()`

## 6. Separate extraction from concept retrieval context

Extraction should not receive retrieved concept content.

Extraction should use:
- the new memo text
- optional retrieved entity/tag references only

Purpose of entity/tag references:
- help identify canonical entities/tags
- help infer cleaner relationships

The extraction step must not:
- copy facts from reference context unless supported by the new memo text
- treat reference items as newly introduced facts automatically

## 7. Update extraction schema

Keep extraction focused on:
- `summary`
- `tags`
- `entities`
- `relationships`

Remove `existing_links` from extraction output.

Extraction is no longer responsible for deciding how the new concept links to retrieved graph nodes.

## 8. Refactor prompts

### Filter prompt
- compare the new memo against retrieved concept candidates
- return only one of:
  - `discard`
  - `duplicate`
  - `new`
- optionally return the matched duplicate concept IDs from retrieval candidates only

### Extraction prompt
- extract only from the new memo text
- include retrieved entity/tag references as optional grounding aids
- explicitly forbid copying unsupported facts from references
- explicitly forbid using retrieved concepts as extraction input

## 9. Refactor storage stage

Storage runs only for `new` memos.

For `new` memos:
- create a new concept node
- reuse existing tag nodes when normalized IDs match
- reuse existing entity nodes when normalized IDs match
- create:
  - concept -> tag edges
  - concept -> entity edges
  - entity -> entity relationship edges

Do not use model-generated `existing_links`.

## 10. Define deterministic linking rules

Linking to retrieved nodes should be handled by storage rules, not extraction output.

Recommended rules:
- retrieved **concepts**:
  - link from new concept using `related_context`
- retrieved **entities**:
  - only link/reuse when they align with extracted entities
- retrieved **tags**:
  - only link/reuse when they align with extracted tags

Avoid broad automatic linking that could overconnect the graph.

## 11. Update schemas and code paths

- Replace or extend `FilterDecision` to support `discard`, `duplicate`, and `new`
- remove `ExistingLink` usage from the memo ingestion flow
- update pipeline branching logic
- update `ingest_memo()` result messages
- update CLI/API code paths if they assume the old extraction contract

## 12. Rewrite and expand tests

Add or update tests for:
- retrieval runs before filter
- first-hop entity/tag expansion works correctly
- neighboring concepts are excluded from expansion
- duplicate memos skip extraction
- duplicate memos skip storage
- new memos proceed to extraction and storage
- extraction does not receive retrieved concept text
- entity/tag references are available to extraction
- existing entity/tag nodes are reused without duplication
- new concept nodes connect to retrieved concepts/entities/tags only through deterministic rules

## 13. Validate with seeded examples

Manual or automated validation scenarios:
- same concept phrased twice -> second memo classified as duplicate
- new memo related to an existing concept -> stored as new and linked appropriately
- new memo mentioning known entities/tags -> existing nodes reused cleanly
- no duplicate concept growth caused by retrieval-grounded ingestion

## Notes

Key design principle:
- **duplicate detection belongs before extraction**
- **extraction should operate on new text, with only limited entity/tag reference help**
- **duplicate memos must not reach extraction or storage**
