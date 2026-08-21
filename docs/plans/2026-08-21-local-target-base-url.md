# Configurable `--local` Base URL + API Key (DeepSeek-ready)

> **Status:** PLAN ONLY — no code changed yet. Owner gate: approve before execution.

**Goal:** Let `mind-map memo --local <model>` target any OpenAI-compatible endpoint (e.g. DeepSeek `https://api.deepseek.com/v1`) via env vars, instead of the hardcoded `http://127.0.0.1:11435/v1`.

**Architecture:** Extend `LocalTarget` with an optional `api_key` field and read two new env vars at target-resolution time. Downstream pipeline (`FilterAgent`, `KnowledgeProcessor`) already receives the target object untouched — zero changes there.

## Current Reality (verified in code today)

| Site | File:Line | Behavior |
|---|---|---|
| Hardcoded default | `src/mind_map/processor/cli_executor.py:26` | `_DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11435/v1"` |
| Target dataclass | `cli_executor.py:30-34` | `LocalTarget(model, base_url=…)` — no auth field |
| Model resolver | `cli_executor.py:52` | `resolve_local_model(model, base_url=…)` **already accepts** `base_url` — callers just never pass it |
| Local HTTP command | `cli_executor.py:88-107` | Sends **no Authorization header** → remote endpoints would 401 even with correct base_url |
| Resolution site 1 | `src/mind_map/app/services.py:91-99` (`parse_memo_target`) | Builds `LocalTarget` without threading base_url |
| Resolution site 2 | `src/mind_map/app/services.py:148-156` (`memo_ingest`) | Duplicated resolution block (kept for test-patch compatibility) |
| Pipeline threading | `src/mind_map/app/pipeline.py:85,124` | `FilterAgent(target=…)`, `KnowledgeProcessor(target=…)` — no changes needed |

## Decisions

1. **Env-var config only** — `MIND_MAP_LOCAL_BASE_URL`, `MIND_MAP_LOCAL_API_KEY`. No `config.yaml` schema change this pass (YAGNI; can layer later).
2. **Zero behavior change when unset** — defaults stay `11435`, no header sent.
3. **Auth header conditional** — sent only when `MIND_MAP_LOCAL_API_KEY` is set; localhost Ollama/LM Studio unaffected.
4. **No new CLI flags** — `--local <model-id>` semantics unchanged (value = model id).
5. **Env ownership:** only `cli_executor.py` reads these vars via two new getters.

## Non-goals

- Not touching `MiniMaxTarget` (default MiniMax path stays as-is).
- Not touching `processing_llm.py` / `reasoning_llm.py` (separate seams).
- Not changing MCP tool signatures (env inherits into MCP server process automatically).

## Tasks (TDD, bite-sized)

### Task 1: RED — env-driven base URL & auth header tests
**Files:** Create `tests/test_local_target_env.py`

```python
import pytest
from mind_map.processor import cli_executor as ce


def test_default_base_url_unchanged(monkeypatch):
    monkeypatch.delenv("MIND_MAP_LOCAL_BASE_URL", raising=False)
    cmd = ce.build_cli_template(ce.LocalTarget(model="m"))
    assert "127.0.0.1:11435" in cmd
    assert "Authorization" not in cmd          # negative assertion


def test_env_base_url_used(monkeypatch):
    monkeypatch.setenv("MIND_MAP_LOCAL_BASE_URL", "https://api.deepseek.com/v1")
    t = ce.LocalTarget(model="deepseek-chat")
    cmd = ce.build_cli_template(t)
    assert "api.deepseek.com/v1" in cmd


def test_env_api_key_sends_bearer(monkeypatch):
    monkeypatch.setenv("MIND_MAP_LOCAL_API_KEY", "sk-test")
    t = ce.LocalTarget(model="m", api_key=ce.get_local_api_key())
    cmd = ce.build_cli_template(t)
    assert "Authorization" in cmd and "Bearer sk-test" in cmd
```

Run: `poetry run pytest tests/test_local_target_env.py -v` → **FAIL** (`get_local_api_key` missing).

### Task 2: GREEN — extend `LocalTarget` + getters + header
**Files:** Modify `src/mind_map/processor/cli_executor.py`

```python
_DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11435/v1"

def get_local_base_url() -> str:
    return os.getenv("MIND_MAP_LOCAL_BASE_URL", _DEFAULT_LOCAL_BASE_URL)

def get_local_api_key() -> str | None:
    return os.getenv("MIND_MAP_LOCAL_API_KEY") or None

@dataclass(frozen=True)
class LocalTarget:
    model: str
    base_url: str = _DEFAULT_LOCAL_BASE_URL   # default literal kept (frozen-dataclass compat)
    api_key: str | None = None
```

In `build_local_command`: build headers dict conditionally —

```
headers={'Content-Type': 'application/json'} + ({'Authorization': 'Bearer ' + api_key} if api_key else {})
```

(keep the stdlib one-liner pattern per repo convention). Run Task 1 suite → **PASS**, then `tests/test_minimax_target.py tests/test_memo_cli_modes.py` → still green.

### Task 3: Wire resolution sites in `services.py`
Both `parse_memo_target` (line ~96) and `memo_ingest` (line ~155):

```python
from mind_map.processor.cli_executor import get_local_base_url, get_local_api_key
base_url = get_local_base_url()
model_name = _resolve_local_model(model=local or None, base_url=base_url)
return LocalTarget(model=model_name, base_url=base_url, api_key=get_local_api_key())
```

**RED first:** add test asserting `monkeypatch.setenv("MIND_MAP_LOCAL_BASE_URL", …)` reaches `resolve_local_model(base_url=…)` (patch it, capture kwargs). Existing lazy-import patch comments must keep working — do not remove the lazy imports.

### Task 4: Docs
- `.env.example`: add the two vars with comments (DeepSeek example).
- `CLAUDE.md` LLM table: one row noting `--local` now honors `MIND_MAP_LOCAL_BASE_URL/_API_KEY`.
- Update Hermes skill `mind-map-memo` §"`--local` flag hardcodes base URL" after merge (post-execution step, not this repo).

### Task 5: Full verification + commit
```bash
ulimit -n 4096 && poetry run pytest -q        # whole suite, expect all green
poetry run ruff check .
git add -A && git commit -m "feat: configurable --local base URL/API key via env"
```

## ⛔ Owner Gate — Live E2E (needs your approval + secret)

Real DeepSeek validation requires `DEEPSEEK_API_KEY` (→ `~/.vault/secret.md`, exported per-shell, never committed/logged):
```bash
export MIND_MAP_LOCAL_BASE_URL=https://api.deepseek.com/v1
export MIND_MAP_LOCAL_API_KEY=$DEEPSEEK_API_KEY
mind-map memo "E2E smoke: configurable local endpoint" --local deepseek-chat
mind-map stats   # node count +1
unset MIND_MAP_LOCAL_BASE_URL MIND_MAP_LOCAL_API_KEY
```
Without a key: mock-level tests above are the acceptance bar; live probe deferred.

## Risks / Notes

- `max_tokens=1200` in local command is modest for remote models (MiniMax path uses 124k). Kept unchanged — flag for tuning if extractions truncate.
- DeepSeek supports `response_format: json_object` (already sent by local command) ✓.
- Frozen-dataclass default must stay a literal (can't reference function call) — hence getter pattern.
