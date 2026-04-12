# Filter Agent Ollama -> OpenClaw MiniMax Fallback Plan

## Goal

Refactor the FilterAgent execution path so filtering uses:
1. local Ollama `phi3.5` as the primary model
2. `openclaw agent --agent minimax --message ...` as the last model fallback
3. heuristic rules as the final non-LLM fallback

Cloud-provider routing should be removed from the **FilterAgent path**.

## Status

**Implemented** on 2026-04-12.

This document remains as the design and implementation record.

Implemented outcomes:
- filter path now uses a dedicated backend chain:
  1. Ollama `phi3.5`
  2. OpenClaw MiniMax CLI
  3. heuristic pipeline fallback
- cloud-auto provider routing removed from the **filter path only**
- separate `filter_llm` wiring added to pipeline, CLI, and API
- dedicated filter fallback tests added

Important operational finding:
- a direct repo-path `uv tool install` did not reliably refresh the installed runtime on this machine
- the fallback-chain behavior only worked live after reinstalling from the built wheel

Recommended deployment install path:
```bash
./.venv/bin/python -m build
uv tool uninstall mind-map
uv tool install dist/mind_map-0.1.0-py3-none-any.whl
```

---

## Target Filter Execution Order

1. heuristic trivial-discard fast path
2. Ollama `phi3.5` via LangChain
3. OpenClaw MiniMax CLI fallback
4. heuristic duplicate/new fallback

This gives:
- cheap/local filtering first
- stronger backup when the local model fails
- deterministic safety fallback when all model paths fail

---

## 1. Separate filter model policy from general processing model policy

### Objective
Prevent FilterAgent from using the generic multi-provider processing LLM routing.

### Files
- `src/mind_map/processor/filter_agent.py`
- optionally `src/mind_map/processor/processing_llm.py`

### Changes
- Do **not** use cloud-auto provider selection for filter classification
- Introduce a filter-specific model selection rule:
  - primary model: Ollama `phi3.5`
  - fallback model: OpenClaw MiniMax CLI
- Keep this scoped to FilterAgent only

### Notes
- Do not remove cloud support repo-wide unless explicitly requested
- Keep extraction and other processing flows unchanged unless separately refactored

---

## 2. Add direct OpenClaw MiniMax filter fallback

### Objective
Make FilterAgent capable of falling back to the same OpenClaw MiniMax CLI pattern already used by extraction.

### Files
- `src/mind_map/processor/filter_agent.py`

### Changes
Add helper functions similar to extraction:
- JSON extraction/parsing helper for filter output
- OpenClaw CLI filter call helper, for example:
  - `_call_openclaw_minimax_filter(prompt, timeout=...)`

### CLI contract
Use:
```bash
openclaw agent --agent minimax --message "<filter prompt>"
```

### Expected JSON output
```json
{
  "action": "new" | "duplicate" | "discard",
  "reason": "...",
  "summary": "cleaned text or null"
}
```

### Error handling
- if CLI missing -> return `None`
- if timeout -> return `None`
- if invalid JSON -> return `None`
- if non-zero exit -> return `None`

---

## 3. Lock primary filter model to Ollama phi3.5

### Objective
Ensure the first LLM attempt is always local `phi3.5`, not cloud routing.

### Files
- `src/mind_map/processor/filter_agent.py`
- optionally `src/mind_map/processor/processing_llm.py`

### Changes
Choose one of these two approaches:

#### Option A: Keep logic inside FilterAgent
- instantiate a dedicated Ollama LangChain model directly for filtering
- explicitly request `phi3.5`
- avoid generic `get_processing_llm()`

#### Option B: Add a narrow helper in processing_llm.py
- create something like `get_filter_ollama_llm()`
- always returns Ollama `phi3.5`
- no cloud provider resolution

### Recommendation
- prefer **Option B** if you want cleaner reuse and easier testing
- prefer **Option A** if you want minimum file churn

---

## 4. Refactor FilterAgent execution order

### Objective
Make FilterAgent own the fallback chain explicitly.

### Files
- `src/mind_map/processor/filter_agent.py`

### Changes
Add a method like:
- `evaluate_with_fallbacks(text, retrieved_concepts)`

Execution order:
1. try LangChain/Ollama `phi3.5`
2. if that fails, try OpenClaw MiniMax CLI
3. if that fails, return `None` so pipeline uses heuristic fallback

### Suggested internal split
- `_evaluate_with_ollama(...)`
- `_evaluate_with_openclaw_minimax(...)`
- `_parse_filter_decision(...)`

---

## 5. Keep prompt contract identical across both model backends

### Objective
Prevent behavior drift between phi3.5 and MiniMax.

### Files
- `src/mind_map/processor/filter_agent.py`

### Changes
Use one shared prompt structure for both paths:
- same system instructions
- same retrieved concept formatting
- same JSON contract

### Prompt requirements
- classify only as `discard`, `duplicate`, or `new`
- retrieved concepts are only for duplicate checking
- do not perform extraction
- summary should be populated only when useful for `new`

---

## 6. Update pipeline integration

### Objective
Ensure `pipeline.py` uses the new filter backend chain correctly.

### Files
- `src/mind_map/app/pipeline.py`

### Changes
Filter stage should do:
1. heuristic discard fast path
2. call FilterAgent with phi3.5 primary and MiniMax fallback
3. if FilterAgent returns `None` or raises, use heuristic duplicate/new fallback

### Important behavior
- `duplicate` still skips extraction and storage
- `discard` still ends the pipeline immediately
- `new` continues to extraction

---

## 7. Remove cloud-provider setup from filter path

### Objective
Ensure filter flow no longer depends on cloud provider routing.

### Files
- `src/mind_map/app/pipeline.py`
- `src/mind_map/processor/filter_agent.py`
- optionally `src/mind_map/processor/processing_llm.py`

### Changes
- stop routing FilterAgent through generic cloud-first model selection
- if current filter flow indirectly gets cloud LLM from `get_processing_llm()`, replace that with filter-specific setup
- leave non-filter paths unchanged unless requested

### Scope clarification
This plan removes cloud providers from the **filter path only**, not necessarily from the whole repository.

---

## 8. Testing plan

### Files
- `tests/test_pipeline.py`
- `tests/test_filter_fallback.py`
- optionally extend integration tests

### Add tests for
1. phi3.5 filter path success
2. phi3.5 failure -> MiniMax fallback used
3. MiniMax failure -> heuristic fallback used
4. duplicate decision from MiniMax correctly skips extraction/store
5. cloud provider routing is not used by FilterAgent path
6. invalid JSON from either backend is handled safely
7. missing `openclaw` binary handled safely
8. timeout in MiniMax CLI handled safely

---

## 9. Risks and edge cases

### Risks
- phi3.5 may still be weak on nuanced semantic duplicates
- MiniMax CLI fallback adds latency
- two-model behavior can drift if prompts diverge

### Edge cases
- Ollama unavailable locally
- `phi3.5` not installed
- OpenClaw CLI missing
- OpenClaw returns wrapper prose instead of raw JSON
- retrieved concept list empty

### Mitigation
- keep a robust JSON parser
- preserve heuristic fallback
- keep prompt identical across both model backends
- validate live behavior through the installed binary, not repo-only tests

---

## 10. Recommended implementation order

1. Add OpenClaw MiniMax filter helper
2. Refactor FilterAgent into explicit fallback chain
3. Lock primary filter model to Ollama `phi3.5`
4. Update pipeline integration
5. Add filter-specific tests
6. Run targeted pipeline + filter tests
7. Run broader regression tests
8. Validate through installed wheel-based CLI runtime

---

## Recommended scope decision

Implement this as a **filter-specific backend chain**.

Do **not** rip out cloud-provider support from the entire repo unless separately requested.
That keeps the change safer and focused on the part that now matters most: duplicate-vs-new filtering.
