# OpenClaw Agent Implementation Plan

## Goal
Implement the migration from **Claude CLI** to **OpenClaw Agent runtime** for `mind-map ask`, while keeping changes minimal and localized.

---

## Scope

This change is a **transport/backend migration**, not a reasoning architecture redesign.

### In scope
- Replace default reasoning backend transport
- Add `openclaw-agent` provider support
- Keep `ResponseGenerator` flow intact
- Keep changes mostly inside `src/mind_map/rag/reasoning_llm.py`
- Update config defaults and docs if needed

### Out of scope
- Large refactor of RAG flow
- Changes to retrieval logic
- Changes to ask pipeline structure
- Broader redesign of config system

---

## Target Runtime Call

```bash
openclaw agent --agent main --message "<composed prompt>"
```

### Locked runtime behavior
- target agent: `main`
- use normal gateway/runtime path
- do not use `--local`
- send one composed prompt string

---

## Files to Change

### Primary implementation target
1. `src/mind_map/rag/reasoning_llm.py`

### Secondary only if required
2. `src/mind_map/core/config.py` *(only if provider enum/default needs update)*
3. `CLAUDE.md` *(update reasoning backend docs)*
4. `README.md` or related docs *(if backend behavior is documented there)*

---

## Implementation Steps

### Step 1 — Add helper functions
In `reasoning_llm.py`, add:
- `_find_openclaw_cli()`
- `check_openclaw_agent_available()`
- `get_openclaw_agent_llm()`

Behavior:
- verify `openclaw` exists in PATH
- optionally run a tiny test prompt to confirm command works
- return wrapper instance if available

### Step 2 — Add OpenClaw Agent backend wrapper
File:
- `src/mind_map/rag/reasoning_llm.py`

Add a new LangChain-compatible wrapper class, e.g.:
- `OpenClawAgentLLM(BaseChatModel)`

Responsibilities:
- convert LangChain messages into one composed prompt string
- call `openclaw agent --agent main --message <prompt>`
- capture stdout
- return `ChatResult`

### Step 3 — Add provider selection
Update `get_reasoning_llm()` to support:
- `provider == "openclaw-agent"`

Suggested fallback order:
1. `openclaw-agent`
2. Gemini
3. Anthropic
4. OpenAI
5. Claude CLI (optional legacy fallback only if kept)

### Step 4 — Change default provider
Update default config behavior so reasoning defaults to:
- `provider: openclaw-agent`
- `model: main`

If config model already supports free-form provider strings, keep change minimal.
If not, update config schema just enough to allow `openclaw-agent`.

### Step 5 — Run one real integration test
Run a real `mind-map ask` query and verify:
- valid response
- clean output
- no recursion
- no architecture breakage

### Step 6 — Keep legacy path optional
Do not fully delete Claude CLI support in first pass.
Keep it available as fallback / legacy path unless removal is trivial and safe.

### Step 7 — Update docs
Update:
- `CLAUDE.md`
- any README/docs mentioning Claude CLI as the default reasoning backend

---

## Prompt Composition Rules

The new wrapper should compose prompt text from LangChain messages like this:
- system messages
- human/user messages
- assistant context messages if present

Output target:
- plain text final answer only

Do not add extra wrappers unless necessary.
Do **not** inject extra agent-behavior instructions beyond what the existing prompt path already provides.
The runtime should behave like a controlled reasoning backend, not a free-form chat assistant.

---

## Example Internal Flow

```text
mind-map ask
  -> get_reasoning_llm()
  -> OpenClawAgentLLM
  -> compose prompt
  -> subprocess: openclaw agent --agent main --message <prompt>
  -> capture stdout
  -> return ChatResult
  -> ResponseGenerator continues normally
```

---

## Validation Plan

### Functional tests
1. `mind-map ask "What is mind-map project?"`
2. `mind-map ask` with no relevant context
3. `mind-map ask` with relevant retrieved context
4. verify output returns as plain text
5. verify no accidental recursion loop
6. verify output does **not** contain OpenClaw-style meta text (no reply tags, no session chatter, no extra assistant framing)

### Command-level checks
- `openclaw agent --agent main --message "hello"`
- verify response comes back cleanly
- verify wrapper captures stdout correctly

### Regression checks
- no breakage in `ResponseGenerator`
- no breakage in fallback providers
- no breakage when OpenClaw CLI is unavailable

---

## Minimal Code Change Strategy

To keep risk low:
- concentrate implementation in `reasoning_llm.py`
- avoid changing `main.py` ask flow unless absolutely needed
- avoid changing retrieval/store logic
- only touch config schema if current config rejects `openclaw-agent`

---

## Risks to Test

1. Output cleanliness
2. Prompt size handling
3. Runtime latency
4. Recursion risk
5. CLI availability/path issues

These are test targets, not redesign triggers.

---

## Deliverable Definition

Implementation is complete when:
- `mind-map ask` works using `openclaw agent`
- `openclaw-agent` is the default reasoning provider
- output remains plain text and usable by current flow
- docs are updated
- legacy fallback path remains safe

---

## Recommendation

Implement in this order:
1. wrapper class
2. provider selection
3. default provider switch
4. validation tests
5. docs update

---

*Implementation plan v1.0*
