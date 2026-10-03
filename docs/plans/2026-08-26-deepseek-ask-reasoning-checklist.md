# DeepSeek Ask Reasoning — Execution Checklist

> Companion checklist for `docs/plans/2026-08-26-deepseek-ask-reasoning.md`.
> Mark items `[x]` as they complete. Items marked **OWNER GATE** require explicit owner approval before execution — they are NOT part of the default autonomous run.
> Repo: `/Users/gwansun/Desktop/projects/mind-map`. Test runner: `poetry run pytest` (full suite needs `ulimit -n 4096`).

## Pre-flight (read-only)

- [x] Confirm `config.yaml` still has `reasoning_llm.provider: minimax-direct` (Tasks 1–4 must NOT change it)
- [x] Confirm `.env` has `DEEPSEEK_API_KEY` (presence only — never print the value)
- [x] Baseline: `ulimit -n 4096 && poetry run pytest -q` — all green BEFORE any change (record count: 284 passed, 4 skipped — verified 2026-08-26)
- [x] `git status --short` clean; branch is `main` (4 ahead of origin) — only untracked plan docs present

## Task 1 — RED: DeepSeekChatLLM contract tests

- [x] Create `tests/test_deepseek_chat_llm.py` with the 8 tests from the plan (llm_type, key-from-env, defaults, model override env, message conversion, missing-key raise, factory None, check_available)
- [x] Run `poetry run pytest tests/test_deepseek_chat_llm.py -v` → FAIL (ImportError: DeepSeekChatLLM) — RED confirmed
- [x] Commit: `8f8cef3` — "test: add RED tests for DeepSeekChatLLM reasoning provider"

## Task 2 — GREEN: DeepSeekChatLLM + factory

- [x] Add `DeepSeekChatLLM` + `check_deepseek_available()` + `get_deepseek_llm()` to `src/mind_map/rag/reasoning_llm.py` (insert after MiniMax section, before Cloud Providers)
  - [x] `_llm_type == "deepseek"`
  - [x] model default `deepseek-v4-flash`, override via `MIND_MAP_DEEPSEEK_MODEL`
  - [x] `base_url = "https://api.deepseek.com/v1"`, endpoint built as `{base}/chat/completions`
  - [x] `max_tokens = 8192`, `timeout = 120`
  - [x] messages → OpenAI roles (system/human/ai)
  - [x] `RuntimeError("DEEPSEEK_API_KEY not set...")` when key empty
- [x] Run `poetry run pytest tests/test_deepseek_chat_llm.py -v` → PASS (8)
- [x] Regression: `poetry run pytest tests/test_minimax_chat_llm.py tests/test_claude_cli_llm.py -q` → PASS
- [x] Commit: `30e9a2a` — "feat: add DeepSeekChatLLM reasoning provider (direct API transport)"

## Task 3 — Wire `deepseek` into `get_reasoning_llm()`

- [x] RED first: add `TestDeepSeekProviderDispatch` (2 tests) to `tests/test_deepseek_chat_llm.py` → first test FAILS ("Unknown reasoning_llm provider: deepseek")
- [x] Add configured-provider branch `elif provider == "deepseek":` (after minimax-direct block)
- [x] Add fallback-chain entry (FIRST in chain, skipped when provider == "deepseek"): `get_deepseek_llm("deepseek-v4-flash", timeout)`
- [x] Run `poetry run pytest tests/test_deepseek_chat_llm.py -v` → PASS (10)
- [x] Regression: 4-file provider set → PASS (72)
- [x] Commit: `5d35f7a` — "feat: wire deepseek provider into get_reasoning_llm dispatch and fallback chain"

## Task 4 — `llm_status.py` deepseek branch + default alignment

- [x] RED first: add `test_reports_deepseek_online_when_key_present` to `TestGetLLMStatus` (NO `_clean_env` fixture — it doesn't clear DEEPSEEK_API_KEY; use `patch.dict`) → FAILS
- [x] Import `check_deepseek_available` in `src/mind_map/rag/llm_status.py`
- [x] Change default `reasoning_config.get("provider", "claude-cli")` → `"deepseek"`
- [x] Add `if reasoning_provider == "deepseek":` branch before claude-cli branch
- [x] Run `poetry run pytest tests/test_processing_llm_providers.py -q` → PASS (43)
- [x] Commit: `b5c54f9` — "feat: report deepseek reasoning provider status in llm_status"

## Task 5 — ⛔ OWNER GATE: flip `config.yaml` default

- [x] OWNER APPROVAL obtained (2026-08-26)
- [x] `config.yaml`: `provider: minimax-direct` → `deepseek`, `model: MiniMax-M2.5` → `deepseek-v4-flash` (temp/timeout unchanged)
- [x] Commit: `2081bda` — "feat: default reasoning provider to deepseek (deepseek-v4-flash)"

## Task 6 — Docs

- [x] README.md — checked: no reasoning-provider mentions, no changes needed
- [x] CLAUDE.md LLM table: Reasoning row → `deepseek-direct` / `deepseek-v4-flash`; fallback prose updated
- [x] `docs/MIND_MAP_HERMES_INTEGRATION.md` — checked: does not name the reasoning provider, no changes needed
- [x] Commit: `7a7511d` — "docs: document deepseek as default reasoning provider"

## Task 7 — Full mock-level verification

- [x] `ulimit -n 4096 && poetry run pytest -q` → 295 passed, 4 skipped (baseline 284 + 11 new)
- [x] `poetry run ruff check` on touched files → 0 NEW findings (6 introduced findings fixed in commit `4790610`; remaining 9 are pre-existing baseline)
- [x] `git status --short` — only plan/checklist docs untracked; no stray changes
- [x] Git log review — 1 commit per task + 1 ruff hygiene commit, config.yaml only changed in Task 5

## Task 8 — ⛔ OWNER GATE: Live E2E (paid DeepSeek call + writes 1 Q&A node to production graph)

- [x] OWNER APPROVAL obtained (2026-08-26)
- [x] `poetry run mind-map ask "What do I know about DeepSeek?" --data-dir /Users/gwansun/mind-map/data`
  - [x] Output shows `Using DeepSeek API (deepseek-v4-flash)`
  - [x] Response panel generated (NOT "Reasoning LLM not available") — grounded answer citing 2 context nodes
  - [x] `Stored: 1 Q&A node (linked to 2 context nodes)` line present
- [x] `poetry run mind-map stats --data-dir /Users/gwansun/mind-map/data` → 427→434 nodes, 83→84 concepts (Q&A back-feed +7 nodes via heuristic extraction; Ollama offline)
- [x] Running backend verified: `GET /health` → `reasoning_llm: {provider: deepseek, status: online}`

## Post-completion

- [x] All checklist items marked; owner-gated items approved and completed (2026-08-26)
- [x] Consider pushing branch (now 11 ahead of origin) — owner decision → pushed; verified 2026-10-03 while preparing this commit: origin/main level at `04ab101`, 0 commits ahead
- [x] Hermes `mind-map-memo` skill review — provider docs changed (deepseek reasoning provider); note for next skill maintenance pass
