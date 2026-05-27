# Replace OpenClaw Agent with Direct MiniMax API

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace all `openclaw agent --agent minimax --message` subprocess calls with direct HTTP POST to `https://api.minimax.io/v1/chat/completions` using MiniMax-M2.5.

**Architecture:** Two distinct openclaw usage paths need replacement. Path A is the `OpenClawAgentLLM` LangChain wrapper in `reasoning_llm.py` (reasoning/response generation). Path B is the `OpenClawTarget` → `build_openclaw_command()` chain in `cli_executor.py` (memo extraction/filtering). Both become direct `requests.post()` HTTP calls, eliminating the `openclaw` binary dependency entirely.

**Tech Stack:** `requests` (already in project deps), `os.getenv("MINIMAX_API_KEY")`, `MiniMax-M2.5` model, `https://api.minimax.io/v1/chat/completions`, `<think>` tag stripping, JSON-first response parsing.

**Provider migration order** (per batch-ordering reference):
1. Freeze RED contract tests
2. Add new transport behind existing seam
3. Switch callers
4. Remove obsolete subprocess/orchestration
5. Rename artifacts/fields
6. Naming cleanup last

---

## Current Code Reality

### Two distinct openclaw call paths:

| Path | File | Class/Function | What it does | Callers |
|------|------|---------------|--------------|---------|
| **A** | `reasoning_llm.py` | `OpenClawAgentLLM._generate()` | `subprocess.run(["openclaw", "agent", "--agent", model, "--message", prompt])` | `get_reasoning_llm()`, direct instantiation |
| **B** | `cli_executor.py` | `build_openclaw_command()` → `run_cli_json()` | Builds shell template, `subprocess.run()` with `shlex.split()` | `KnowledgeProcessor.extract_with_references()`, `FilterAgent` |

### Path A detail — `reasoning_llm.py`:
- `_find_openclaw_cli()` (line 23): `shutil.which("openclaw")`
- `_is_openclaw_error_output()` (line 223): detects error payloads in agent output
- `OpenClawAgentLLM` (line 280): LangChain BaseChatModel wrapper, `_llm_type = "openclaw-agent"`
- `check_openclaw_agent_installed()` (line 358): binary existence check
- `check_openclaw_agent_available()` (line 363): test prompt with `openclaw agent --agent main --message "respond with only: OK"`
- `get_openclaw_agent_llm()` (line 385): factory function
- `get_reasoning_llm()` (line 505): orchestrator — uses `openclaw-agent` as default AND as fallback (lines 557-559)

### Path B detail — `cli_executor.py`:
- `OpenClawTarget` dataclass (line 30): `message: str`, `agent: str | None`
- `_OPENCLAW_TEMPLATE = "openclaw agent --message"` (line 50)
- `_OPENCLAW_AGENT_TEMPLATE = "openclaw agent --agent {agent} --message"` (line 51)
- `build_openclaw_command()` (line 54): returns the shell command string
- `build_cli_template()` (line 121): dispatches to `build_openclaw_command()` if `isinstance(target, OpenClawTarget)`
- `run_cli_json()` (line 128): runs the shell command via `subprocess.run()`, parses stdout JSON

### Test files using OpenClawTarget:
- `tests/test_pipeline.py`: `openclaw_target()` fixture returns `OpenClawTarget(agent="minimax")`, used in 3 tests
- `tests/test_filter_fallback.py`: 4 `FilterAgent(target=OpenClawTarget())` instantiations

### Config:
- `config.yaml` line 16: `provider: openclaw-agent`

---

## Target State

### New class: `MiniMaxChatLLM` (Path A replacement)

A LangChain `BaseChatModel` that POSTs to `https://api.minimax.io/v1/chat/completions`:

```python
class MiniMaxChatLLM(BaseChatModel):
    model: str = Field(default="MiniMax-M2.5")
    timeout: int = Field(default=120)
    api_key: str = Field(default_factory=lambda: os.getenv("MINIMAX_API_KEY", ""))
    base_url: str = Field(default="https://api.minimax.io")

    def _generate(self, messages, stop, run_manager, **kwargs) -> ChatResult:
        # Build OpenAI-compatible payload
        payload = {
            "model": self.model,
            "messages": [{"role": _map_role(m), "content": m.content} for m in messages],
            "max_tokens": 124000,
        }
        resp = requests.post(f"{self.base_url}/v1/chat/completions",
                             headers={"Authorization": f"Bearer {self.api_key}"},
                             json=payload, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        content = _strip_think_tags(content)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
```

### New transport: `MiniMaxTarget` (Path B replacement)

Replace `OpenClawTarget` in `cli_executor.py` with a new target that carries the API call info directly:

```python
@dataclass(frozen=True)
class MiniMaxTarget:
    """Direct MiniMax API target for memo extraction/filtering."""
    api_key: str
    base_url: str = "https://api.minimax.io"
    model: str = "MiniMax-M2.5"
```

Then `build_cli_template()` dispatches to a `build_minimax_curl()` that generates a `requests`-based Python one-liner (or better: refactor `run_cli_json()` itself to support direct HTTP calls natively).

### Key design decisions:
- **Model:** `MiniMax-M2.5` — M2.7 puts ALL content inside `<think>` tags and is unusable for structured JSON
- **Endpoint:** `/v1/chat/completions` — NOT `/v1/text/chatcompletion` (returns auth 1004)
- **Host:** `api.minimax.io` — NOT `api.minimax.chat`
- **Think tags:** Strip with `re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)`
- **max_tokens:** 124000 for all extraction/filtering/reasoning — full M2.5 128K context window eliminates all truncation risk regardless of `` reasoning tokens
- **API key:** `os.getenv("MINIMAX_API_KEY")` — loaded from `~/.vault/secret.md` at runtime
- **Timeout:** 120s (matches current `OpenClawAgentLLM` timeout)

---

## Implementation Plan

### Phase 0: Freeze RED Contract Tests

#### Task 0.1: Write failing RED tests for new MiniMax classes

**Files:**
- Create: `tests/test_minimax_chat_llm.py`
- Create: `tests/test_minimax_target.py`

**Test `test_minimax_chat_llm.py`:**

```python
import os
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage
from mind_map.rag.reasoning_llm import MiniMaxChatLLM

class TestMiniMaxChatLLM:
    def test_llm_type_is_minimax_direct(self):
        llm = MiniMaxChatLLM()
        assert llm._llm_type == "minimax-direct"

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-123"}):
            llm = MiniMaxChatLLM()
            assert llm.api_key == "sk-test-123"

    def test_missing_api_key_returns_none_from_factory(self):
        with patch.dict(os.environ, {}, clear=True):
            llm = MiniMaxChatLLM()
            assert llm.api_key == ""

    def test_messages_converted_to_openai_format(self):
        llm = MiniMaxChatLLM(api_key="sk-test")
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

            result = llm._generate(messages)

            call_args = mock_post.call_args
            assert call_args[1]["json"]["messages"] == [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]

    def test_strips_think_tags(self):
        llm = MiniMaxChatLLM(api_key="sk-test")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "<think>reasoning...</think>\n\n{\"summary\":\"test\"}"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = llm._generate([HumanMessage(content="test")])
            content = result.generations[0].message.content
            assert "<think>" not in content
            assert "reasoning" not in content
            assert "{\"summary\":\"test\"}" in content
```

**Test `test_minimax_target.py`:**

```python
from mind_map.processor.cli_executor import MiniMaxTarget, build_minimax_http_command

class TestMiniMaxTarget:
    def test_target_has_default_model(self):
        target = MiniMaxTarget(api_key="sk-test")
        assert target.model == "MiniMax-M2.5"

    def test_target_has_default_base_url(self):
        target = MiniMaxTarget(api_key="sk-test")
        assert target.base_url == "https://api.minimax.io"

    def test_build_http_command_uses_correct_endpoint(self):
        target = MiniMaxTarget(api_key="sk-test")
        cmd = build_minimax_http_command(target)
        assert "/v1/chat/completions" in cmd
        assert "api.minimax.io" in cmd
```

**Step 1: Run tests to verify RED** — all should fail because classes don't exist yet.

Run: `cd /Users/gwansun/Desktop/projects/mind-map && poetry run pytest tests/test_minimax_chat_llm.py tests/test_minimax_target.py -v`

Expected: all FAIL (class/function not defined).

**Step 2: Commit**

```bash
git add tests/test_minimax_chat_llm.py tests/test_minimax_target.py
git commit -m "test: add RED tests for MiniMaxChatLLM and MiniMaxTarget replacement"
```

---

### Phase 1: Add New Transport Behind Existing Seam

#### Task 1.1: Add `MiniMaxTarget` and `build_minimax_http_command()` to `cli_executor.py`

**File:** `src/mind_map/processor/cli_executor.py`

Add after the `LocalTarget` definition (around line 42):

```python
@dataclass(frozen=True)
class MiniMaxTarget:
    """Direct MiniMax API target for memo extraction/filtering.

    Replaces the deprecated OpenClawTarget. Uses the MiniMax M2.5 model
    via direct HTTP calls instead of shelling out to the openclaw CLI.
    """
    api_key: str
    base_url: str = "https://api.minimax.io"
    model: str = "MiniMax-M2.5"
    max_tokens: int = 124000
```

Update `MemoTarget` type alias:

```python
MemoTarget = OpenClawTarget | LocalTarget | MiniMaxTarget
```

Add `build_minimax_http_command()` and `_strip_think_tags()` after `build_local_command()` (around line 118):

```python
def _strip_think_tags(text: str) -> str:
    """Strip MiniMax M-series <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def build_minimax_http_command(target: MiniMaxTarget) -> str:
    """Build a Python one-liner that POSTs to MiniMax API directly.

    This replaces the old `openclaw agent --agent minimax --message` pattern
    with a direct HTTP call to the MiniMax chat completions endpoint.
    The prompt is injected via command-line argument.
    """
    return (
        "python3 -c "
        + shlex.quote(
            "import json, sys, urllib.request; "
            f"api_key={target.api_key!r}; "
            f"base_url={target.base_url.rstrip('/')!r}; "
            f"model={target.model!r}; "
            f"max_tokens={target.max_tokens!r}; "
            "prompt=sys.argv[1]; "
            "body=json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens}).encode(); "
            "req=urllib.request.Request(base_url + '/v1/chat/completions', data=body, "
            "headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}); "
            "resp=urllib.request.urlopen(req, timeout=60); "
            "raw=resp.read().decode(); "
            "data=json.loads(raw); "
            "content=data['choices'][0]['message']['content']; "
            "import re; "
            "cleaned=re.sub(r'<think>.*?</think>\\\\s*', '', content, flags=re.DOTALL).strip(); "
            "sys.stdout.write(cleaned)"
        )
    )
```

Update `build_cli_template()` (line 121-124) to add the new dispatch:

```python
def build_cli_template(target: MemoTarget) -> str:
    """Build the exact CLI template for a resolved memo target."""
    if isinstance(target, OpenClawTarget):
        return build_openclaw_command(target)
    if isinstance(target, MiniMaxTarget):
        return build_minimax_http_command(target)
    return build_local_command(target)
```

**Step 1: Run the new RED tests** — should still fail or the MiniMax tests should now pass:

Run: `poetry run pytest tests/test_minimax_target.py -v`
Expected: PASS (the target class and builder now exist).

**Step 2: Run existing tests** to verify no regressions:

Run: `poetry run pytest tests/test_pipeline.py tests/test_filter_fallback.py -v`
Expected: PASS (existing OpenClawTarget tests still work).

**Step 3: Commit**

```bash
git add src/mind_map/processor/cli_executor.py
git commit -m "feat: add MiniMaxTarget and build_minimax_http_command transport"
```

#### Task 1.2: Add `MiniMaxChatLLM` class to `reasoning_llm.py`

**File:** `src/mind_map/rag/reasoning_llm.py`

Add new section after the OpenClaw Agent section (after line 403, before the Cloud Provider section):

```python
# ============== MiniMax Direct LLM ==============


def _strip_think_tags(text: str) -> str:
    """Strip MiniMax M-series <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


class MiniMaxChatLLM(BaseChatModel):
    """LangChain-compatible wrapper for direct MiniMax API calls.

    Uses POST to https://api.minimax.io/v1/chat/completions with
    the MiniMax-M2.5 model. Replaces the deprecated OpenClawAgentLLM.
    """

    model: str = Field(default="MiniMax-M2.5", description="MiniMax model name")
    timeout: int = Field(default=120, description="Timeout in seconds for API calls")
    api_key: str = Field(
        default_factory=lambda: os.getenv("MINIMAX_API_KEY", ""),
        description="MiniMax API key (from MINIMAX_API_KEY env var)",
    )
    base_url: str = Field(
        default="https://api.minimax.io",
        description="MiniMax API base URL",
    )
    max_tokens: int = Field(default=124000, description="Maximum completion tokens")

    @property
    def _llm_type(self) -> str:
        return "minimax-direct"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response via direct MiniMax API call."""
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY not set. Set it in your environment or .env file."
            )

        try:
            import requests
        except ImportError:
            raise RuntimeError("requests is required for MiniMax API calls")

        # Convert LangChain messages to OpenAI format
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
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.Timeout:
            raise RuntimeError(f"MiniMax API timed out after {self.timeout}s")
        except requests.RequestException as e:
            raise RuntimeError(f"MiniMax API request failed: {e}")

        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"MiniMax API returned no choices: {body}")

        response_text = choices[0].get("message", {}).get("content", "")
        if not response_text:
            raise RuntimeError("MiniMax API returned empty content")

        # Strip <think> tags (M-series models wrap reasoning in them)
        response_text = _strip_think_tags(response_text)

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=response_text))
            ]
        )
```

Add factory function and checks:

```python
def check_minimax_api_available() -> bool:
    """Check if MiniMax API key is configured."""
    return bool(os.getenv("MINIMAX_API_KEY"))


def get_minimax_llm(model: str = "MiniMax-M2.5", timeout: int = 120) -> Any:
    """Get MiniMax Direct LLM for response generation.

    Args:
        model: MiniMax model name (default: MiniMax-M2.5)
        timeout: Timeout in seconds

    Returns:
        MiniMaxChatLLM instance or None if API key not configured
    """
    if not check_minimax_api_available():
        console.print("[yellow]MINIMAX_API_KEY not set. Cannot use MiniMax API.[/yellow]")
        return None

    api_key = os.getenv("MINIMAX_API_KEY", "")
    return MiniMaxChatLLM(
        model=model,
        timeout=timeout,
        api_key=api_key,
    )
```

**Step 1: Run the new RED tests** — MiniMaxChatLLM tests should now pass:

Run: `poetry run pytest tests/test_minimax_chat_llm.py -v`
Expected: PASS (4 tests).

**Step 2: Run existing test suite** to verify no regressions:

Run: `poetry run pytest -v`
Expected: existing tests still pass (new code is additive, not yet wired).

**Step 3: Commit**

```bash
git add src/mind_map/rag/reasoning_llm.py
git commit -m "feat: add MiniMaxChatLLM class for direct MiniMax API calls"
```

---

### Phase 2: Switch Callers

#### Task 2.1: Wire `minimax-direct` into `get_reasoning_llm()` as default provider

**File:** `src/mind_map/rag/reasoning_llm.py` — `get_reasoning_llm()` function

Replace the `openclaw-agent` path (lines 527-532) with `minimax-direct`:

```python
    # Try configured provider first
    if provider == "minimax-direct":
        llm = get_minimax_llm(model, timeout)
        if llm:
            console.print(f"[dim]Using MiniMax API ({model})[/dim]")
            return llm
        console.print("[yellow]MiniMax API not available, trying fallback...[/yellow]")
    elif provider == "openclaw-agent":
        # Legacy support — redirect to minimax-direct
        console.print("[dim]openclaw-agent is deprecated, using minimax-direct instead[/dim]")
        llm = get_minimax_llm("MiniMax-M2.5", timeout)
        if llm:
            console.print(f"[dim]Using MiniMax API ({model})[/dim]")
            return llm
        console.print("[yellow]MiniMax API not available, trying fallback...[/yellow]")
    elif provider == "claude-cli":
```

Update the default provider and fallback chain (lines 521, 557-562):

```python
    provider = reasoning_config.get("provider", "minimax-direct")
```

Remove the `openclaw-agent` fallback (lines 557-562):
```python
    # Fallback chain: Claude CLI → Gemini → Anthropic → OpenAI
    # (MiniMax is the primary; no need to fallback to it)

    if provider != "claude-cli" and check_claude_cli_installed():
```

**Step 1: Update config.yaml to match**

```yaml
reasoning_llm:
  provider: minimax-direct  # Options: minimax-direct, claude-cli, gemini, anthropic, openai
  model: MiniMax-M2.5       # For minimax-direct: model name
  temperature: 0.7
  timeout: 120
```

**Step 2: Run tests**

Run: `poetry run pytest -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add src/mind_map/rag/reasoning_llm.py config.yaml
git commit -m "feat: switch default reasoning provider to minimax-direct API"
```

#### Task 2.2: Switch memo CLI from `OpenClawTarget` to `MiniMaxTarget`

**File:** `src/mind_map/app/cli/main.py` — `memo` command (lines 404-470)

The CLI now only accepts `--local`. We need to add `--minimax` as the default/primary mode, with `--local` as fallback.

Currently the flow is:
1. `--local` is required
2. Creates `LocalTarget(model=model_name)`
3. `build_cli_template(target)` → generates Python one-liner curl to local 11435

New flow:
1. Default: use `MiniMaxTarget` if `MINIMAX_API_KEY` is set
2. `--local` as explicit fallback
3. `build_cli_template(target)` dispatches correctly

**Implementation:**

```python
@app.command()
def memo(
    text: Annotated[str, typer.Argument(help="Text to ingest into the knowledge graph")],
    source: Annotated[
        str | None, typer.Option("--source", "-s", help="Source identifier")
    ] = None,
    data_dir: Annotated[
        Path, typer.Option("--data-dir", "-d", help="Directory for database storage")
    ] = get_data_dir(),
    local: Annotated[
        str | None,
        typer.Option(
            "--local",
            help='Use explicit local OpenAI-compatible path. Omit value to auto-resolve first /models entry, or pass a model id'
        ),
    ] = None,
) -> None:
    """Ingest a note or thought into the knowledge graph."""
    from mind_map.app.pipeline import ingest_memo_cli
    from mind_map.processor.cli_executor import (
        CLIExecutionError,
        LocalTarget,
        MiniMaxTarget,
        build_cli_template,
        resolve_local_model,
    )
    from mind_map.rag.graph_store import GraphStore

    if not data_dir.exists():
        console.print("[red]Database not initialized. Run 'mind-map init' first.[/red]")
        raise typer.Exit(1)

    # Determine target: MiniMax API (default) or local
    if local is not None:
        # Explicit local mode
        try:
            model_name = resolve_local_model(model=local or None)
            target = LocalTarget(model=model_name)
        except CLIExecutionError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        # Default: MiniMax API
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            console.print("[red]MINIMAX_API_KEY not set.[/red]")
            console.print("[dim]Set MINIMAX_API_KEY in your environment or pass --local for local mode.[/dim]")
            raise typer.Exit(1)
        target = MiniMaxTarget(api_key=api_key)

    shared_cli = build_cli_template(target)

    store = GraphStore(data_dir)
    store.initialize()

    console.print(f"[dim]CLI: {shared_cli}[/dim]")
    console.print("[yellow]Processing memo...[/yellow]")

    success, message, node_ids = ingest_memo_cli(
        text,
        store,
        target=target,
        source_id=source,
    )

    if message.startswith("Memo rejected:"):
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)

    if success:
        console.print(f"[green]{message}[/green]")
        if node_ids:
            ids_display = ", ".join(node_ids[:3]) + ("..." if len(node_ids) > 3 else "")
            console.print(f"[dim]Node IDs: {ids_display}[/dim]")
    else:
        console.print(f"[yellow]{message}[/yellow]")
```

**Step 1: Update tests** — `tests/test_memo_cli_modes.py` needs updating:

- The `--local` requirement test should now check that `--local` is no longer required by default
- Add test for MiniMax API key missing error
- Add test for MiniMax default path

**Step 2: Run tests**

Run: `poetry run pytest tests/test_memo_cli_modes.py -v`
Expected: updated tests PASS.

**Step 3: Commit**

```bash
git add src/mind_map/app/cli/main.py tests/test_memo_cli_modes.py
git commit -m "feat: switch memo CLI default to MiniMax API, --local as explicit fallback"
```

#### Task 2.3: Update tests that use `OpenClawTarget` to use `MiniMaxTarget`

**Files:** `tests/test_pipeline.py`, `tests/test_filter_fallback.py`

Replace `OpenClawTarget(agent="minimax")` with `MiniMaxTarget(api_key="sk-test")`.

In `tests/test_pipeline.py`:
```python
# Replace:
from mind_map.processor.cli_executor import OpenClawTarget
# With:
from mind_map.processor.cli_executor import MiniMaxTarget

@pytest.fixture
def minimax_target() -> MiniMaxTarget:
    return MiniMaxTarget(api_key="sk-test")
```

Update the 3 test functions to use `minimax_target` instead of `openclaw_target`.

In `tests/test_filter_fallback.py`:
```python
# Replace 4 occurrences of:
agent = FilterAgent(target=OpenClawTarget())
# With:
agent = FilterAgent(target=MiniMaxTarget(api_key="sk-test"))
```

**Step 1: Run tests**

Run: `poetry run pytest tests/test_pipeline.py tests/test_filter_fallback.py -v`
Expected: PASS.

**Step 2: Commit**

```bash
git add tests/test_pipeline.py tests/test_filter_fallback.py
git commit -m "test: migrate test fixtures from OpenClawTarget to MiniMaxTarget"
```

---

### Phase 3: Remove Obsolete openclaw Code

#### Task 3.1: Remove `OpenClawAgentLLM`, `OpenClawTarget`, and related functions

**File:** `src/mind_map/rag/reasoning_llm.py`

Remove:
- `_find_openclaw_cli()` (lines 23-25)
- `_is_openclaw_error_output()` (lines 223-277)
- `OpenClawAgentLLM` class (lines 280-355)
- `check_openclaw_agent_installed()` (lines 358-360)
- `check_openclaw_agent_available()` (lines 363-382)
- `get_openclaw_agent_llm()` (lines 385-403)

Also remove the legacy `elif provider == "openclaw-agent":` block from `get_reasoning_llm()` and any remaining `openclaw-agent` fallback logic.

**File:** `src/mind_map/processor/cli_executor.py`

Remove:
- `OpenClawTarget` dataclass (lines 29-35)
- `_OPENCLAW_TEMPLATE` (line 50)
- `_OPENCLAW_AGENT_TEMPLATE` (line 51)
- `build_openclaw_command()` (lines 54-58)

Update `MemoTarget`:
```python
MemoTarget = LocalTarget | MiniMaxTarget
```

Update `build_cli_template()`:
```python
def build_cli_template(target: MemoTarget) -> str:
    if isinstance(target, MiniMaxTarget):
        return build_minimax_http_command(target)
    return build_local_command(target)
```

Remove `import shutil` if no longer needed.

**Step 1: Run full test suite**

Run: `poetry run pytest -v`
Expected: PASS. Any test that still references deleted symbols will fail — those should have been updated in Phase 2.

**Step 2: Commit**

```bash
git add src/mind_map/rag/reasoning_llm.py src/mind_map/processor/cli_executor.py
git commit -m "refactor: remove obsolete OpenClawAgentLLM, OpenClawTarget, and CLI helpers"
```

---

### Phase 4: Naming Cleanup

#### Task 4.1: Rename `_llm_type` if desired (cosmetic)

The `_llm_type` for the new class is already `"minimax-direct"` — no rename needed.

#### Task 4.2: Update `config.yaml` comments

Remove `openclaw-agent` from the options list entirely:

```yaml
reasoning_llm:
  provider: minimax-direct  # Options: minimax-direct, claude-cli, gemini, anthropic, openai
  model: MiniMax-M2.5       # MiniMax model name (e.g., MiniMax-M2.5, MiniMax-M2.7)
  temperature: 0.7
  timeout: 120
```

**Step 1: Commit**

```bash
git add config.yaml
git commit -m "docs: remove openclaw-agent from config options"
```

#### Task 4.3: Update CLAUDE.md and documentation

Update `CLAUDE.md`:
- Change `openclaw agent --agent minimax --message "..."` references to MiniMax API
- Update "OpenClaw MiniMax primary path" section to "MiniMax API primary path"
- Update config table to show `minimax-direct` instead of `openclaw-agent`

**Step 1: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for MiniMax API migration"
```

---

### Phase 5: Verify E2E

#### Task 5.1: Live memo ingestion test with MiniMax API

```bash
cd /Users/gwansun/Desktop/projects/mind-map
MINIMAX_API_KEY=$(cat ~/.vault/secret.md | grep MINIMAX_API_KEY | ...) poetry run mind-map memo "I am testing the new MiniMax API integration for knowledge graph memo ingestion" --data-dir /Users/gwansun/mind-map/data
```

Expected: "Processing memo..." → green success message.

#### Task 5.2: Live reasoning test

```bash
poetry run mind-map ask "What is in my knowledge graph?"
```

Expected: Uses MiniMax API for reasoning, returns results.

#### Task 5.3: Run full test suite one final time

```bash
poetry run pytest -v
```

Expected: all PASS.

---

## Rollout Strategy

**Direct replacement** — no feature flag. The old `openclaw-agent` provider config value maps to `minimax-direct` with a deprecation warning for one release cycle, then removed.

## Contract Changes

- **Config key:** `reasoning_llm.provider` now defaults to `minimax-direct` (was `openclaw-agent`)
- **New env var required:** `MINIMAX_API_KEY` for default operation
- **CLI:** `memo` command no longer requires `--local` — defaults to MiniMax API
- **MemoTarget type:** `OpenClawTarget` removed, `MiniMaxTarget` added

## Files Changed

| File | Change |
|------|--------|
| `src/mind_map/rag/reasoning_llm.py` | Remove OpenClawAgentLLM, add MiniMaxChatLLM, update get_reasoning_llm() |
| `src/mind_map/processor/cli_executor.py` | Remove OpenClawTarget, add MiniMaxTarget + builder |
| `src/mind_map/app/cli/main.py` | Default to MiniMax API, --local as explicit fallback |
| `config.yaml` | Change provider default to minimax-direct |
| `CLAUDE.md` | Update docs |
| `tests/test_minimax_chat_llm.py` | New: RED tests for MiniMaxChatLLM |
| `tests/test_minimax_target.py` | New: RED tests for MiniMaxTarget |
| `tests/test_pipeline.py` | Migrate OpenClawTarget → MiniMaxTarget |
| `tests/test_filter_fallback.py` | Migrate OpenClawTarget → MiniMaxTarget |
| `tests/test_memo_cli_modes.py` | Update --local behavior tests |
