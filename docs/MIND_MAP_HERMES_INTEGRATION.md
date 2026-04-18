# Mind Map ↔ Hermes Integration Guide

This document explains how the `mind-map` knowledge graph is integrated into Hermes Agent, why it was implemented this way, and how to reimplement it quickly later.

It is written as an implementation instruction file, not just a summary.

---

## 1. Goal

Make Hermes automatically enrich non-trivial user messages with relevant context from the local `mind-map` CLI before the LLM answers.

Target behavior:
- Hermes should **not** wait for the model to decide whether to call a retrieval tool.
- Relevant mind-map context should be **forcefully injected** into the current turn.
- The injection should be **ephemeral** and **must not mutate the cached system prompt**.
- The injection should **not pollute persisted conversation history**.
- Failures must be **fail-open**: if `mind-map` is unavailable or returns nothing, Hermes should continue normally.

---

## 2. Core Design Decision

### 2.1 Do **not** put live mind-map retrieval into the system prompt

Hermes deliberately separates:
- stable cached system prompt state
- ephemeral per-call context injected at API-call time

This is documented in:
- `website/docs/developer-guide/prompt-assembly.md`

Why this matters:
- the Hermes system prompt is cached per session
- changing it every turn hurts prompt caching
- live retrieval content is query-specific and should not be frozen into session-wide prompt state
- query-specific context belongs in the **current user turn**, not in the stable system prompt

### 2.2 Follow the OpenClaw pattern conceptually, but adapt it to Hermes architecture

OpenClaw’s behavior:
- prefetch mind-map context for non-trivial user messages
- wrap it in `<mind_map_context> ... </mind_map_context>`
- inject it before the model runs
- skip trivial messages and duplicate injections
- fail open on errors

Hermes equivalent should do the same **behaviorally**, but using Hermes’s own ephemeral user-message injection path.

---

## 3. Where Hermes Actually Supports This

The critical entry point is:
- `run_agent.py`

### 3.1 Conversation entry point
- `run_agent.py: run_conversation(...)`

Relevant behavior:
- user message is added to `messages`
- Hermes builds or reuses a cached system prompt
- then it prepares API-call messages later

### 3.2 Plugin hook for pre-turn user context
Hermes already has a plugin hook:
- `pre_llm_call`

Relevant code area:
- around `run_agent.py:8448`

Hermes calls plugin hooks before the tool-calling loop and collects extra context from them.

### 3.3 Actual ephemeral injection point
Relevant code area:
- around `run_agent.py:8603`

Hermes iterates through API messages and, for the **current turn’s user message only**, appends:
- external memory prefetch context
- plugin `pre_llm_call` context

Important detail:
- Hermes modifies the **API payload copy**, not the persisted `messages` list
- this means the injected context is **ephemeral**

This is the key architectural reason the integration should be a plugin first.

---

## 4. Recommended Integration Strategy

## Preferred approach: project plugin using `pre_llm_call`

Why this is preferred:
- minimal invasive patching
- aligns with Hermes architecture
- keeps system prompt stable
- keeps transcript clean
- easy to disable, replace, or port later
- avoids merge pain when updating Hermes upstream

---

## 5. Implemented Plugin Location

Current implementation was added as a **project plugin** inside the Hermes repo:

- `./.hermes/plugins/mind-map-context/plugin.yaml`
- `./.hermes/plugins/mind-map-context/__init__.py`
- `./.hermes/plugins/mind-map-context/README.md`

Absolute path in this repo clone:
- `/Users/gwansun/Desktop/projects/custom-hermes/hermes-agent/.hermes/plugins/mind-map-context/`

Project plugins load only when this environment variable is enabled:

```bash
export HERMES_ENABLE_PROJECT_PLUGINS=1
```

---

## 6. Plugin Behavior Specification

The plugin should:

1. Register a `pre_llm_call` hook
2. Read the current `user_message`
3. Skip retrieval for trivial queries
4. Skip retrieval if `<mind_map_context>` is already present
5. Check whether `mind-map` exists on PATH
6. Run `mind-map retrieve`
7. If output is non-empty, wrap it in:

```text
<mind_map_context>
...
</mind_map_context>
```

8. Return that wrapped block as plugin-provided context
9. Let Hermes append it to the current turn’s user message at API-call time
10. Fail open on timeout, missing CLI, non-zero return code, or empty output

---

## 7. Current Implementation Details

File:
- `./.hermes/plugins/mind-map-context/__init__.py`

### 7.1 Constants used

```python
MIND_MAP_DATA_DIR = "/Users/gwansun/.openclaw/workspace/projects/mind-map/data"
MIND_MAP_N_RESULTS = 3
MIND_MAP_TIMEOUT_SEC = 10
```

### 7.2 Skip logic

Current skip conditions:
- message length less than 3
- starts with `/`
- already contains `<mind_map_context>`
- greeting / acknowledgement regex match

Current regex:

```python
r"^(hi|ok|hello|thanks|thank you|hey|yo|sup|bye|goodbye|yes|no|sure|k|okay|안녕|ㅎㅇ|하이)\\b"
```

### 7.3 Retrieval command used

```bash
mind-map retrieve --data-dir /Users/gwansun/.openclaw/workspace/projects/mind-map/data --n-results 3 <query>
```

Important note:
- use `--data-dir`, not `-d`

### 7.4 Return shape from the hook

Current hook returns:

```python
{"context": "<mind_map_context>...</mind_map_context>"}
```

Hermes accepts either a dict with `context` or a raw string, but the dict form is cleaner and more explicit.

---

## 8. Current Plugin Source

At the time this guide was written, the plugin logic is:

```python
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

MIND_MAP_DATA_DIR = "/Users/gwansun/.openclaw/workspace/projects/mind-map/data"
MIND_MAP_N_RESULTS = 3
MIND_MAP_TIMEOUT_SEC = 10
_ACK_RE = re.compile(
    r"^(hi|ok|hello|thanks|thank you|hey|yo|sup|bye|goodbye|yes|no|sure|k|okay|안녕|ㅎㅇ|하이)\\b",
    re.IGNORECASE,
)


def _is_skippable_query(text: str) -> bool:
    body = (text or "").strip()
    if len(body) < 3:
        return True
    if body.startswith("/"):
        return True
    if "<mind_map_context>" in body:
        return True
    if _ACK_RE.match(body):
        return True
    return False


def _is_mind_map_installed() -> bool:
    return shutil.which("mind-map") is not None


def _retrieve_mind_map_context(query: str) -> str | None:
    if not _is_mind_map_installed():
        return None
    try:
        proc = subprocess.run(
            [
                "mind-map",
                "retrieve",
                "--data-dir",
                MIND_MAP_DATA_DIR,
                "--n-results",
                str(MIND_MAP_N_RESULTS),
                query,
            ],
            capture_output=True,
            text=True,
            timeout=MIND_MAP_TIMEOUT_SEC,
            check=False,
        )
        if proc.returncode != 0:
            logger.debug("mind-map retrieve failed: rc=%s stderr=%r", proc.returncode, (proc.stderr or "")[:300])
            return None
        result = (proc.stdout or "").strip()
        return result or None
    except Exception as exc:
        logger.debug("mind-map retrieve exception: %s", exc)
        return None


def _format_context_block(context: str) -> str:
    return f"<mind_map_context>\n{context.strip()}\n</mind_map_context>"


def pre_llm_call(*, user_message: str = "", **kwargs: Any) -> dict[str, str] | None:
    if _is_skippable_query(user_message):
        return None
    context = _retrieve_mind_map_context(user_message)
    if not context:
        return None
    return {"context": _format_context_block(context)}


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", pre_llm_call)
```

---

## 9. Why This Works Well in Hermes

This approach preserves Hermes’s core invariants:

### 9.1 System prompt remains stable
The system prompt is cached per session.
The plugin does **not** modify it.

### 9.2 Injection is current-turn only
The context is added only to the API payload for the current user message.

### 9.3 Transcript remains clean
The persisted `messages` list is not mutated.
This is cleaner than permanently rewriting the message body.

### 9.4 Retrieval is forced, not model-chosen
The model does not need to decide whether to call a retrieval tool first.
The retrieval happens before the model turn.

### 9.5 Fail-open behavior
If `mind-map` is missing, slow, broken, or empty, Hermes continues normally.

---

## 10. How to Reimplement Quickly Later

If the integration disappears or must be recreated from scratch, do this:

### Step 1. Confirm Hermes still supports `pre_llm_call`
Check these areas:
- `hermes_cli/plugins.py`
- `run_agent.py`

Specifically verify:
- plugin hooks still include `pre_llm_call`
- hook results are still added to current-turn user message content at API-call time

### Step 2. Recreate the plugin directory
From repo root:

```bash
mkdir -p .hermes/plugins/mind-map-context
```

### Step 3. Create `plugin.yaml`
Minimal version:

```yaml
name: mind-map-context
version: 0.1.0
description: Prefetch mind-map context into the current user turn via pre_llm_call.
author: Gwansun
provides_hooks:
  - pre_llm_call
```

### Step 4. Create `__init__.py`
Use the source in section 8, or update constants as needed.

### Step 5. Enable project plugins

```bash
export HERMES_ENABLE_PROJECT_PLUGINS=1
```

### Step 6. Ensure Hermes runtime env exists
Recommended local setup:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Step 7. Load-test the plugin

```bash
source .venv/bin/activate
HERMES_ENABLE_PROJECT_PLUGINS=1 python - <<'PY'
from hermes_cli.plugins import PluginManager
pm = PluginManager()
pm.discover_and_load()
print('plugins=', sorted(pm._plugins.keys()))
print('mind-map-context-enabled=', pm._plugins.get('mind-map-context').enabled if pm._plugins.get('mind-map-context') else None)
print('pre_llm_call_hooks=', len(pm._hooks.get('pre_llm_call', [])))
PY
```

Expected output should show:
- plugin discovered
- plugin enabled
- one `pre_llm_call` hook registered

---

## 11. Build and Launch Workflow

From repo root:

```bash
cd ~/Desktop/projects/custom-hermes/hermes-agent
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
export HERMES_ENABLE_PROJECT_PLUGINS=1
hermes
```

If you want package artifacts too:

```bash
source .venv/bin/activate
uv pip install build
python -m build
ls -la dist
```

---

## 12. Functional Test Checklist

After the plugin loads, verify behavior with a real run.

### 12.1 Positive case
Input:
- non-trivial user message

Expected:
- `mind-map retrieve` runs
- plugin returns context
- current turn user payload contains:

```text
<mind_map_context>
...
</mind_map_context>
```

### 12.2 Trivial case
Input examples:
- `hi`
- `ok`
- `/model`

Expected:
- plugin skips retrieval
- no injected context block

### 12.3 Duplicate prevention
Input already contains `<mind_map_context>`

Expected:
- plugin skips
- no duplicate block

### 12.4 Fail-open
Scenarios:
- `mind-map` missing from PATH
- retrieval command returns non-zero
- timeout
- empty stdout

Expected:
- no crash
- no context block
- Hermes continues normally

---

## 13. If You Want This As a Core Feature Instead

Current recommendation is plugin-first.

If you decide to move the feature into Hermes core later:
- add retrieval before or alongside `_plugin_user_context`
- keep it in the **ephemeral user-message injection path**
- do **not** move it into `_build_system_prompt()`

Best core-level landing area conceptually:
- near `run_agent.py` where `_ext_prefetch_cache` and `_plugin_user_context` are merged into the current-turn user API message

Core migration rule:
- preserve all current semantics:
  - non-trivial only
  - duplicate-safe
  - fail-open
  - API-call-time only
  - no transcript mutation

---

## 14. Things That Must Not Change Accidentally

When reworking this integration, protect these invariants:

1. **Do not inject into cached system prompt**
2. **Do not persist `<mind_map_context>` into conversation history by default**
3. **Do not require the model to explicitly choose a retrieval tool first**
4. **Do not hard-fail the turn if `mind-map` retrieval fails**
5. **Do not use shell interpolation when building the retrieval command**
6. **Do not replace `--data-dir` with `-d` unless the CLI actually supports it**

---

## 15. Future Improvements

Possible improvements later:

### 15.1 Configurable settings
Move these into config/env:
- data directory
- result count
- timeout
- skip regex

### 15.2 Better logging / diagnostics
Add structured logs for:
- skipped due to trivial query
- skipped due to duplicate tag
- retrieval success
- retrieval timeout/failure

### 15.3 Better query shaping
Instead of sending the raw `user_message`, you could later:
- trim quoted history
- strip synthetic prefixes
- cap query length
- optionally enrich with recent local context

Do this only if retrieval quality becomes a problem.

### 15.4 Tests
Add automated tests for:
- plugin loading
- skip logic
- duplicate prevention
- retrieval success formatting
- fail-open behavior

---

## 16. Quick Reimplementation Summary

If you only have one minute later, remember this:

- Hermes already supports ephemeral current-turn user-message injection through `pre_llm_call`
- use a project plugin
- run `mind-map retrieve --data-dir ... --n-results 3 <query>`
- wrap output in `<mind_map_context>` tags
- return it as plugin context
- enable project plugins with:

```bash
export HERMES_ENABLE_PROJECT_PLUGINS=1
```

That is the correct Hermes-native way to reproduce OpenClaw-style forced mind-map injection.
