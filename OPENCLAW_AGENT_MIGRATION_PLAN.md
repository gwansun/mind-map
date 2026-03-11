# OpenClaw Agent Migration Plan

## Goal
Replace the current **Claude CLI-based reasoning path** in the `mind-map` project with **OpenClaw Agent runtime** using:

```bash
openclaw agent --agent main --message "..."
```

OpenClaw Agent should be treated as a **reasoning runtime backend**, not as a raw LLM equivalent. Prompt construction and output behavior must therefore be controlled explicitly.

This change is needed because **Claude CLI is unavailable** in the current environment.

---

## Current State

### Current reasoning path
`mind-map ask` currently uses:
- `get_reasoning_llm()`
- default provider: `claude-cli`
- `ResponseGenerator`
- subprocess call to:

```bash
claude -p --model sonnet
```

### Problem
- Claude CLI is not reliably available/authenticated
- The current default path depends on local Claude CLI installation
- This makes `mind-map ask` fragile in this environment

---

## Target State

`mind-map ask` should use **OpenClaw agent runtime** as the default reasoning layer.

Instead of:
```bash
claude -p --model sonnet
```

Use:
```bash
openclaw agent --agent main --message "..."
```

This would let the project use the configured OpenClaw agent stack rather than depending on Claude CLI.
For MVP, use the **normal gateway/runtime path** (do not use `--local`).

---

## Proposed Change

### 1. Replace Claude CLI wrapper with OpenClaw agent wrapper
Current file:
- `src/mind_map/rag/reasoning_llm.py`

Plan:
- add a new wrapper that shells out to:
  ```bash
  openclaw agent --agent main --message <prompt>
  ```
- parse stdout as the final response
- use that wrapper as the default reasoning backend

---

### 2. Update default reasoning provider
Current config behavior:
- default reasoning provider = `claude-cli`

Proposed behavior:
- default reasoning provider = `openclaw-agent`

Optional fallback order:
1. `openclaw-agent`
2. Gemini
3. Anthropic API
4. OpenAI API

---

### 3. Keep configuration explicit
Update config model to allow something like:

```yaml
reasoning_llm:
  provider: openclaw-agent
  model: main
  timeout: 120
```

Possible meaning:
- `provider: openclaw-agent` → use OpenClaw runtime
- `model: main` → target the main configured agent

---

### 4. Prompt construction compatibility
Current `ResponseGenerator` sends a prompt to the reasoning backend.

Locked design:
- use **one composed prompt string**
- include system instruction + user question + retrieved context
- `openclaw agent --message` receives that composed prompt string
- output should remain plain text for `ResponseGenerator`

This keeps the migration minimal and avoids redesigning the retrieval layer.

---

## Implementation Approach

### Option A — Minimal Migration (recommended, locked)
Keep current architecture and only replace the backend transport.

- keep `ResponseGenerator`
- keep `get_reasoning_llm()` abstraction
- keep overall code change minimal and mainly inside `get_reasoning_llm()` / `reasoning_llm.py`
- replace Claude CLI subprocess wrapper with OpenClaw agent subprocess wrapper

### Option B — Larger Refactor
Refactor the reasoning layer around OpenClaw-native messaging abstractions.

Not recommended for first pass because:
- more moving parts
- more risk
- unnecessary if simple subprocess call works

**Recommended:** Option A

---

## New Wrapper Design

### Proposed class
Example concept:
- `OpenClawAgentLLM(BaseChatModel)`

Responsibilities:
- convert LangChain messages into one composed prompt string
- call:
  ```bash
  openclaw agent --agent main --message <prompt>
  ```
- capture stdout
- return LangChain-compatible response object

This mirrors the current Claude CLI wrapper design.

---

## Files Likely to Change

### Primary
- `src/mind_map/rag/reasoning_llm.py`

### Possibly
- `src/mind_map/core/config.py` (if provider enum/config needs update)
- `CLAUDE.md` (update docs)
- `README.md` or project docs (update reasoning backend docs)

---

## Benefits

### Reliability
- no dependence on Claude CLI auth/session
- uses existing OpenClaw runtime configuration

### Flexibility
- OpenClaw can route to whichever model/provider is configured
- easier to change model behavior centrally

### Consistency
- same agent stack as the rest of the environment
- avoids split reasoning behavior

---

## Risks / Noted Concerns

These concerns are acknowledged in the plan, but they are not blockers at this review stage.

### 1. Prompt formatting
Need to verify that `openclaw agent --message` behaves well with a large composed RAG prompt.

### 2. Output cleanliness
Need to verify that `openclaw agent --message` returns plain text suitable for direct capture.

### 3. Recursion risk
If OpenClaw itself uses mind-map context in certain flows, need to avoid accidental loops.

### 4. Performance / latency
This may be slower than direct Claude CLI if OpenClaw adds extra runtime layers.

### 5. Session/runtime behavior
For MVP, this is locked to:
- target agent = `main`
- normal gateway/runtime path
- no `--local`

So session behavior is mostly resolved, but should still be verified during testing.

---

## Recommended First Test Plan

1. Build a minimal `openclaw agent` wrapper in `reasoning_llm.py`
2. Run `mind-map ask "test query"`
3. Verify:
   - response returns successfully
   - output formatting is clean
   - no recursion/loop behavior
4. Update config defaults
5. Update docs

---

## Locked Decisions

1. `openclaw-agent` becomes the new default reasoning provider
2. target agent is `main` for MVP
3. use normal gateway/runtime path (no `--local`)
4. use one composed prompt string
5. use Option A minimal migration
6. overall code change should stay minimal and mostly within `get_reasoning_llm()` / `reasoning_llm.py`
7. keep noted concerns in the plan, but do not expand scope to solve all of them up front

## Remaining Review Questions

1. Do we want fallback to cloud APIs if `openclaw agent` fails?
2. Should we preserve the Claude CLI code path for optional future use?

---

## Recommendation

For the first migration:
- make `openclaw-agent` the **new default reasoning provider**
- keep Claude CLI code in place as optional legacy fallback
- implement the smallest possible wrapper change
- update docs after successful test

---

*Draft plan for review*
