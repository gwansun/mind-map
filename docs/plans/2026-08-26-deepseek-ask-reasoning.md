# DeepSeek Reasoning Provider for `mind-map ask` — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
> **Status:** PLAN ONLY — no code changed yet. Owner gate: approve before execution.
> **APPLIED 2026-08-26** — all tasks shipped; the companion checklist carries the per-task
> commit hashes (`8f8cef3` … `4790610`, all confirmed present in history). The line above is
> superseded: verified 2026-10-03 while preparing this commit that `config.yaml` has
> `reasoning_llm.provider: deepseek` and `reasoning_llm.py` defines `DeepSeekChatLLM`, which
> the "PLAN ONLY" status contradicts. Paths in this document describe the pre-relocation
> layout (`~/Desktop/projects/mind-map`); the repo now lives at `~/Projects/mind-map`.

**Goal:** Make the DeepSeek API the configured reasoning LLM (LLM-A) for the mind-map `ask` workflow (CLI `mind-map ask`, API `POST /ask`, MCP `mind_map_ask`), replacing MiniMax-M2.5 as the default.

**Architecture:** Add a new `deepseek` reasoning provider to `src/mind_map/rag/reasoning_llm.py`, implemented as a `DeepSeekChatLLM` LangChain `BaseChatModel` that POSTs directly to `https://api.deepseek.com/v1/chat/completions` via `requests` — mirroring the existing, proven `MiniMaxChatLLM` class. Reuse `DEEPSEEK_API_KEY` and `MIND_MAP_DEEPSEEK_MODEL` (both already used by the memo path). No new dependencies, no changes to `ResponseGenerator`, `services.py`, pipeline, or the memo path.

**Tech Stack:** Python 3.11, LangChain `BaseChatModel`, `requests` 2.32.5 (already importable transitively), DeepSeek chat completions API.

---

## Verified Code Reality (2026-08-26)

| Item | Finding |
|---|---|
| Reasoning entry point | `src/mind_map/rag/reasoning_llm.py:443` `get_reasoning_llm()` — reads `reasoning_llm` from `config.yaml`, returns LangChain LLM or `None` |
| Callers | `services.ask_question()` (`services.py:302`), `app/api/routes.py` `POST /ask`, CLI `ask` (`cli/main.py:522`), MCP `mind_map_ask` (same services path) — all consume the `BaseChatModel` interface, so **no caller changes needed** |
| Current default provider | `config.yaml:16` → `provider: minimax-direct`, `model: MiniMax-M2.5` |
| MiniMax pattern to mirror | `MiniMaxChatLLM` (`reasoning_llm.py:223-314`): `requests.post(base_url + /v1/chat/completions, headers=Bearer, json={model, messages, max_tokens})`, `_llm_type` property, factory `get_minimax_llm()` returning `None` when key missing |
| Fallback chain | `reasoning_llm.py:495-518`: Claude CLI → Gemini → Anthropic → OpenAI (each skipped if it was the configured provider) |
| `langchain-openai` limitation | Installed 0.3.35 (+ openai 2.16.0) — `ChatOpenAI.__init__` exposes **no** `base_url`/`openai_api_base` param (verified live via inspect). Reusing the `openai` provider for DeepSeek is NOT possible without upgrading langchain-openai → rejected (see Decisions) |
| Env availability | `DEEPSEEK_API_KEY` already set in project `.env` (gitignored, loaded by `load_dotenv()` in `core/config.py`) and in Hermes `~/.hermes/config.yaml` `mcp_servers.mind-map.env`. `MIND_MAP_DEEPSEEK_MODEL` unset. Live-verified 2026-08-26: API accepts `deepseek-chat` (alias → `deepseek-v4-flash`), `deepseek-v4-flash`, `deepseek-v4-pro`; `/v1/models` lists `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`, `deepseek-v4-pro` |
| `llm_status.py` drift | `src/mind_map/rag/llm_status.py:38` defaults reasoning provider to `"claude-cli"` when key absent and has no `minimax-direct`/`deepseek` branch — health endpoint currently misreports the configured provider as offline |
| `requests` availability | Not declared in `pyproject.toml` main deps, but importable (2.32.5) and already used by `MiniMaxChatLLM`/`cli_executor` — same import pattern is safe |

**Good parts to preserve:** `MiniMaxChatLLM`'s exact class shape (field defaults, OpenAI message conversion, error handling), `get_reasoning_llm()`'s config-then-fallback structure, `ResponseGenerator` untouched.

**Primary mismatch:** no `deepseek` provider exists in the reasoning seam; `config.yaml` points at MiniMax; health status has no DeepSeek awareness.

**Scope boundary:** memo ingestion (CLI/MCP `memo`, `services.memo_ingest`, `resolve_default_memo_target`) already uses DeepSeek and must NOT change. Q&A back-feed extraction uses the *processing* LLM (`ingest_memo_internal`) — out of scope.

---

## Decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | Transport | New `DeepSeekChatLLM(BaseChatModel)` with direct `requests` POST — mirrors `MiniMaxChatLLM`. **Rejected:** reusing `ChatOpenAI(base_url=...)` because installed langchain-openai 0.3.35 has no base_url param; upgrading it (openai 2.x incompatibility surface) violates minimal-change |
| 2 | Model | Default **`deepseek-v4-flash`** (explicit id — `deepseek-chat` is a server-side alias for the same model; explicit id avoids alias ambiguity). Overridable via existing `MIND_MAP_DEEPSEEK_MODEL` env var. Memo path keeps its own default untouched |
| 3 | `max_tokens` for reasoning | **8192** (quality-safe ceiling per DeepSeek ecosystem guidance; MiniMax uses 124000 but that is MiniMax-specific). Not configurable this pass (YAGNI) |
| 4 | Config default | `reasoning_llm.provider: deepseek`, `model: deepseek-v4-flash` in tracked `config.yaml` — MiniMax stays in fallback chain (Decision 5) |
| 5 | Fallback chain | Insert `deepseek` as the first entry of the existing chain when it is not the configured provider; rest unchanged (Claude CLI → Gemini → Anthropic → OpenAI) |
| 6 | `llm_status.py` | Add `deepseek` branch + fix default to `deepseek` (aligns health endpoint with config reality) |
| 7 | Memo path | Untouched |
| 8 | MCP | No server.py changes — MCP process already receives `DEEPSEEK_API_KEY` via Hermes config env block |

## Non-goals

- No changes to memo ingestion / `LocalTarget` / `resolve_default_memo_target`
- No changes to `ResponseGenerator`, `services.ask_question`, pipeline, CLI argument surfaces
- No removal of `MiniMaxChatLLM` or any existing provider
- No new dependencies; no langchain-openai upgrade
- No `max_tokens`/timeout config schema additions

---

## Implementation Tasks (TDD)

### Task 1: RED — DeepSeekChatLLM contract tests

**Files:**
- Create: `tests/test_deepseek_chat_llm.py`

**Step 1: Write failing tests** (mirror `tests/test_minimax_chat_llm.py`):

```python
"""Tests for DeepSeekChatLLM (reasoning provider)."""

import os
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from mind_map.rag.reasoning_llm import DeepSeekChatLLM


class TestDeepSeekChatLLM:
    def test_llm_type_is_deepseek(self):
        llm = DeepSeekChatLLM()
        assert llm._llm_type == "deepseek"

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-123"}):
            llm = DeepSeekChatLLM()
            assert llm.api_key == "sk-test-123"

    def test_defaults(self):
        llm = DeepSeekChatLLM(api_key="sk-test")
        assert llm.model == "deepseek-v4-flash"
        assert llm.base_url == "https://api.deepseek.com/v1"
        assert llm.max_tokens == 8192

    def test_model_override_env(self):
        with patch.dict(os.environ, {"MIND_MAP_DEEPSEEK_MODEL": "deepseek-v4-pro"}):
            llm = DeepSeekChatLLM(api_key="sk-test")
            assert llm.model == "deepseek-v4-pro"

    def test_messages_converted_to_openai_format(self):
        llm = DeepSeekChatLLM(api_key="sk-test")
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
        ]
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hi there!"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            llm._generate(messages)

            kwargs = mock_post.call_args.kwargs
            assert kwargs["json"]["messages"] == [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            assert kwargs["json"]["model"] == "deepseek-v4-flash"

    def test_missing_key_raises(self):
        llm = DeepSeekChatLLM(api_key="")
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY not set"):
            llm._generate([HumanMessage(content="hi")])


class TestDeepSeekFactory:
    def test_returns_none_when_key_missing(self):
        from mind_map.rag.reasoning_llm import get_deepseek_llm

        with patch.dict(os.environ, {}, clear=True):
            assert get_deepseek_llm() is None

    def test_check_available(self):
        from mind_map.rag.reasoning_llm import check_deepseek_available

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            assert check_deepseek_available() is True
```

(Add `import pytest` at top.)

**Step 2: Run to verify failure**

Run: `cd /Users/gwansun/Desktop/projects/mind-map && poetry run pytest tests/test_deepseek_chat_llm.py -v`
Expected: FAIL — `ImportError: cannot import name 'DeepSeekChatLLM'`.

**Step 3: Commit (tests only)**

```bash
git add tests/test_deepseek_chat_llm.py
git commit -m "test: add RED tests for DeepSeekChatLLM reasoning provider"
```

---

### Task 2: GREEN — Add `DeepSeekChatLLM` + factory to `reasoning_llm.py`

**Files:**
- Modify: `src/mind_map/rag/reasoning_llm.py` (insert after the MiniMax Direct section, i.e. after `get_minimax_llm` around line 341, before `# ============== Cloud Provider LLMs ==============`)

**Step 1: Implement**

```python
# ============== DeepSeek Direct LLM ==============


class DeepSeekChatLLM(BaseChatModel):
    """LangChain-compatible wrapper for direct DeepSeek API calls.

    Uses POST to https://api.deepseek.com/v1/chat/completions with the
    deepseek-v4-flash model (OpenAI-compatible transport). Mirrors MiniMaxChatLLM.
    """

    model: str = Field(
        default_factory=lambda: os.getenv("MIND_MAP_DEEPSEEK_MODEL", "deepseek-v4-flash"),
        description="DeepSeek model name (override via MIND_MAP_DEEPSEEK_MODEL)",
    )
    timeout: int = Field(default=120, description="Timeout in seconds for API calls")
    api_key: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
        description="DeepSeek API key (from DEEPSEEK_API_KEY env var)",
    )
    base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API base URL",
    )
    max_tokens: int = Field(default=8192, description="Maximum completion tokens")

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response via direct DeepSeek API call."""
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. Set it in your environment or .env file."
            )

        try:
            import requests
        except ImportError:
            raise RuntimeError("requests is required for DeepSeek API calls")

        openai_messages = []
        for msg in messages:
            if msg.type == "system":
                openai_messages.append({"role": "system", "content": str(msg.content)})
            elif msg.type == "human":
                openai_messages.append({"role": "user", "content": str(msg.content)})
            elif msg.type == "ai":
                openai_messages.append({"role": "assistant", "content": str(msg.content)})
            else:
                openai_messages.append({"role": "user", "content": str(msg.content)})

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.Timeout:
            raise RuntimeError(f"DeepSeek API timed out after {self.timeout}s")
        except requests.RequestException as e:
            raise RuntimeError(f"DeepSeek API request failed: {e}")

        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"DeepSeek API returned no choices: {body}")

        response_text = choices[0].get("message", {}).get("content", "")
        if not response_text:
            raise RuntimeError("DeepSeek API returned empty content")

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=response_text))
            ]
        )


def check_deepseek_available() -> bool:
    """Check if DeepSeek API key is configured."""
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def get_deepseek_llm(
    model: str | None = None, timeout: int = 120
) -> Any:
    """Get DeepSeek Direct LLM for response generation.

    Returns DeepSeekChatLLM instance or None if API key not configured.
    """
    if not check_deepseek_available():
        console.print("[yellow]DEEPSEEK_API_KEY not set. Cannot use DeepSeek API.[/yellow]")
        return None

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    llm = DeepSeekChatLLM(
        timeout=timeout,
        api_key=api_key,
    )
    if model is not None:
        llm.model = model
    return llm
```

Note: all names used (`BaseChatModel`, `BaseMessage`, `AIMessage`, `ChatGeneration`, `ChatResult`, `CallbackManagerForLLMRun`, `Field`, `console`) are already imported at module top.

**Step 2: Run tests**

Run: `poetry run pytest tests/test_deepseek_chat_llm.py -v`
Expected: PASS (all 8).

**Step 3: Commit**

```bash
git add src/mind_map/rag/reasoning_llm.py
git commit -m "feat: add DeepSeekChatLLM reasoning provider (direct API transport)"
```

---

### Task 3: Wire `deepseek` into `get_reasoning_llm()` configured-provider dispatch

**Files:**
- Modify: `src/mind_map/rag/reasoning_llm.py` — `get_reasoning_llm()` (lines 443-521)

**Step 1: Add configured-provider branch** — after the `elif provider == "minimax-direct":` block (ends line 470), insert:

```python
    elif provider == "deepseek":
        llm = get_deepseek_llm(model, timeout)
        if llm:
            console.print(f"[dim]Using DeepSeek API ({model})[/dim]")
            return llm
        console.print("[yellow]DeepSeek API not available, trying fallback...[/yellow]")
```

**Step 2: Add fallback-chain entry** — in the fallback section (before the existing `if provider != "claude-cli" and check_claude_cli_installed():` at line 496), insert as the FIRST entry:

```python
    if provider != "deepseek" and check_deepseek_available():
        llm = get_deepseek_llm("deepseek-v4-flash", timeout)
        if llm:
            console.print("[dim]Using DeepSeek API as fallback[/dim]")
            return llm
```

**Step 3: Add RED test first** — extend `tests/test_deepseek_chat_llm.py`:

```python
class TestDeepSeekProviderDispatch:
    def test_configured_provider_deepseek_returns_deepseek_llm(self):
        from mind_map.rag.reasoning_llm import get_reasoning_llm

        config = {
            "reasoning_llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "temperature": 0.7,
                "timeout": 120,
            }
        }
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("mind_map.core.config.load_config", return_value=config):
                llm = get_reasoning_llm()
        assert llm is not None
        assert llm._llm_type == "deepseek"

    def test_deepseek_unavailable_falls_through(self):
        from mind_map.rag.reasoning_llm import get_reasoning_llm

        config = {
            "reasoning_llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "temperature": 0.7,
                "timeout": 120,
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            with patch("mind_map.core.config.load_config", return_value=config):
                with patch(
                    "mind_map.rag.reasoning_llm.check_claude_cli_installed",
                    return_value=False,
                ):
                    llm = get_reasoning_llm()
        assert llm is None
```

Run → first test FAILS (KeyError/None before dispatch added), second passes trivially; then implement Steps 1-2 and re-run.

Expected after implementation: PASS (both).

**Step 4: Regression run**

Run: `poetry run pytest tests/test_minimax_chat_llm.py tests/test_claude_cli_llm.py tests/test_processing_llm_providers.py -q`
Expected: PASS — existing provider tests unaffected.

**Step 5: Commit**

```bash
git add src/mind_map/rag/reasoning_llm.py tests/test_deepseek_chat_llm.py
git commit -m "feat: wire deepseek provider into get_reasoning_llm dispatch and fallback chain"
```

---

### Task 4: `llm_status.py` — deepseek branch + default alignment

**Files:**
- Modify: `src/mind_map/rag/llm_status.py` (lines 36-52)

**Step 1: Import**

```python
from mind_map.rag.reasoning_llm import (
    check_anthropic_available,
    check_claude_cli_available,
    check_deepseek_available,
    check_gemini_available,
    check_openai_available,
)
```

**Step 2: Default + branch**

```python
    reasoning_provider = reasoning_config.get("provider", "deepseek")

    reasoning_status = "offline"
    if reasoning_provider == "deepseek":
        if check_deepseek_available():
            reasoning_status = "online"
    elif reasoning_provider == "claude-cli":
        if check_claude_cli_available():
            reasoning_status = "online"
```

(keep the remaining branches as-is)

**Step 3: Add RED test** — extend `tests/test_processing_llm_providers.py` `TestGetLLMStatus`:

```python
    def test_reports_deepseek_online_when_key_present(self):
        from mind_map.rag.llm_status import get_llm_status

        config = {
            "processing_llm": {"provider": "ollama", "model": "phi3.5"},
            "reasoning_llm": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        }
        # NOTE: do NOT use the shared _clean_env fixture here — it does not
        # clear DEEPSEEK_API_KEY. Use patch.dict to control it explicitly.
        # NOTE: llm_status imports load_config at module level
        # (`from mind_map.core.config import load_config`), so patch the
        # llm_status-local binding — LOAD_CONFIG_PATCH (the config-module
        # attribute) does not rebind it.
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("mind_map.rag.llm_status.load_config", return_value=config):
                with patch(
                    "mind_map.rag.llm_status.check_ollama_available",
                    return_value=False,
                ):
                    status = get_llm_status()
        assert status["reasoning_llm"]["provider"] == "deepseek"
        assert status["reasoning_llm"]["status"] == "online"
```

Run → FAIL before implementation; PASS after.

**Step 4: Regression run**

Run: `poetry run pytest tests/test_processing_llm_providers.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/mind_map/rag/llm_status.py tests/test_processing_llm_providers.py
git commit -m "feat: report deepseek reasoning provider status in llm_status"
```

---

### Task 5: ⛔ OWNER GATE — flip `config.yaml` default provider

**Files:**
- Modify: `config.yaml` (tracked in git)

Change:

```yaml
reasoning_llm:
  provider: minimax-direct
  model: MiniMax-M2.5
```

to:

```yaml
reasoning_llm:
  provider: deepseek            # Options: deepseek, minimax-direct, claude-cli, gemini, anthropic, openai
  model: deepseek-v4-flash      # For deepseek: model id (default deepseek-v4-flash, override via MIND_MAP_DEEPSEEK_MODEL)
```

temperature/timeout unchanged (0.7 / 120).

**Approval required before this task** — this changes the production default reasoning provider. (All preceding tasks are inert without it: `deepseek` is wired but MiniMax remains the default until this flip.)

**Step 1: Commit**

```bash
git add config.yaml
git commit -m "feat: default reasoning provider to deepseek (deepseek-v4-flash)"
```

---

### Task 6: Docs

**Files:**
- Modify: `README.md` — memo "Explicit Memo Model Paths" section is unaffected; update the reasoning-LLM mention if any (check current text first)
- Modify: `CLAUDE.md` — LLM Configuration table: Reasoning row → `deepseek-direct` → `deepseek-v4-flash`; update fallback prose
- Modify: `docs/MIND_MAP_HERMES_INTEGRATION.md` — only if it names the reasoning provider (check first)

Commit:

```bash
git add README.md CLAUDE.md docs/MIND_MAP_HERMES_INTEGRATION.md
git commit -m "docs: document deepseek as default reasoning provider"
```

---

### Task 7: Full verification (mock level)

```bash
ulimit -n 4096 && poetry run pytest -q
```

Expected: all pass (284+ existing + new deepseek tests). `poetry run ruff check src/mind_map/rag/reasoning_llm.py src/mind_map/rag/llm_status.py tests/test_deepseek_chat_llm.py` — no NEW findings in touched files.

---

### Task 8: ⛔ OWNER GATE — Live E2E (paid API call + writes one Q&A node to production graph)

Requires approval; consumes DeepSeek API credits and back-feeds a Q&A node into `/Users/gwansun/mind-map/data` (CLI ask back-feed).

```bash
cd /Users/gwansun/Desktop/projects/mind-map
poetry run mind-map ask "What do I know about DeepSeek?" --data-dir /Users/gwansun/mind-map/data
```

Expected output markers:
- `Using DeepSeek API (deepseek-v4-flash)` printed
- Response panel generated (not "Reasoning LLM not available")
- `Stored: 1 Q&A node` line

Verify: `poetry run mind-map stats --data-dir /Users/gwansun/mind-map/data` — concept node count +1.

Note: Q&A back-feed extraction uses the *processing* LLM (ollama `phi3.5`; if Ollama is down it degrades to heuristic extraction and the answer is still stored). This does not block the reasoning-LLM verification.

Restart the running backend after the config flip so `POST /ask`/health pick it up (uvicorn autoreload covers source changes; `config.yaml` load happens per-call so a restart is only needed if the process cached state — verify via `GET /health` → `reasoning_llm` should report `provider: deepseek`, `status: online`).

---

## Risks & Notes

| Risk | Mitigation |
|---|---|
| `requests` is not a declared main dependency (transitive only) | Already relied on by `MiniMaxChatLLM`/`cli_executor`; no change in pattern. Optional hygiene: add `requests = "^2"` to pyproject — separate decision, not in this plan |
| DeepSeek `max_tokens=8192` may truncate extremely long ask answers | Matches ecosystem quality guidance; if a real truncation appears, revisit as a follow-up (do not preemptively enlarge) |
| Config flip changes behavior for all ask surfaces (CLI, API, MCP) at once | That is the intent. MiniMax stays in fallback chain (Decision 5) |
| `.env` key exists but empty/misconfigured at runtime | Factory returns `None` → fallback chain engages; ask never hard-fails |
| `MIND_MAP_DEEPSEEK_MODEL` is shared with memo path | Same env var controls both; changing it affects memo extraction model too — document in Task 6 |

## File-Level Diff Summary (predicted)

```
src/mind_map/rag/reasoning_llm.py   | +130 lines  (DeepSeekChatLLM + factory + dispatch + fallback entry)
src/mind_map/rag/llm_status.py      | +6 / -2     (import, default, deepseek branch)
config.yaml                         | ±2 lines    (provider/model flip)
tests/test_deepseek_chat_llm.py     | +150 lines  (NEW)
tests/test_processing_llm_providers.py | +20 lines (NEW test in TestGetLLMStatus)
README.md / CLAUDE.md               | small doc deltas (Task 6)
```
