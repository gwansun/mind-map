# Plan: MCP Server → Follows CLI Logic

## Goal

Eliminate the silent divergence between `mind-map` CLI commands and the MCP
server in `src/mind_map/mcp/server.py`. Today the same user-facing operation
(`memo`, `retrieve`, `stats`, `prune`) runs **different code paths** depending
on whether the caller is a shell or an MCP client. This plan extracts the shared
business logic into a `services.py` layer and turns both CLI commands and MCP
tools into thin wrappers around it — so behavior, defaults, and error
contracts are identical.

**Additionally, the MCP server is not currently registered with Hermes (see
`~/.hermes/config.yaml` — only `cdi-pipeline`, `gbrain`, `minimax` are wired).
After the refactor, this plan adds the `mind-map` MCP entry so the tools
become callable from any Hermes session.

**Current MCP tool inventory** (verified from `server.py`):
- `mind_map_retrieve(query, n_results=5, workspace_id=None)` ✓
- `mind_map_memo(text, workspace_id=None)` ✓ — but missing `local`/`source`/`data_dir` params and silently uses `get_processing_llm()` instead of CLI parity
- `mind_map_stats(workspace_id=None)` ✓ — but missing `data_dir` param
- `mind_map_report(workspace_id=None)` ✓ — but missing `data_dir` param
- `mind_map_prune(workspace_id=None)` ✓ — but missing `--percent` and `data_dir` params
- `mind_map_health(workspace_id=None)` ✓ — but missing `data_dir` param
- `mind_map_ask` ✗ — **only tool missing entirely**

**Planned MCP tool inventory** (post-refactor, CLI parity):
- `mind_map_memo(text, source=None, local=None, data_dir=None, workspace_id=None)` — adds `source`, `local`, `data_dir` for CLI parity
- `mind_map_retrieve(query, n_results=5, show_context=True, max_context_per_node=3, data_dir=None, workspace_id=None)` — adds `show_context`, `max_context_per_node`, `data_dir`
- `mind_map_ask(query, depth=2, n_results=5, data_dir=None, model=None, back_feed=False, workspace_id=None)` — NEW tool; `depth` is dead flag for parity, `back_feed=False` default (read-only)
- `mind_map_stats(data_dir=None, workspace_id=None)` — adds `data_dir`
- `mind_map_prune(percent=0.1, data_dir=None, workspace_id=None)` — adds `percent`, `data_dir`
- `mind_map_report(data_dir=None, workspace_id=None)` — adds `data_dir`
- `mind_map_health(data_dir=None, workspace_id=None)` — adds `data_dir`

**M3 clarified**: adding `local` to `mind_map_memo` is a **new capability**, not
just parity — MCP callers have never been able to choose a local model
target. This deserves a callout in the commit message and the PR description.

---

## Current State (verified from source)

### CLI commands — `src/mind_map/app/cli/main.py`

| Command | Body lines | Key deps |
|---|---|---|
| `memo` (405–475) | `MemoTarget` resolution (LocalTarget / MiniMaxTarget) → `ingest_memo_cli` → status print | `cli_executor.resolve_local_model`, `app.pipeline.ingest_memo_cli` |
| `retrieve` (478–531) | `store.query_similar` → `enrich_context_nodes` → `get_connected_context` → formatted lines | `GraphStore` |
| `ask` (534–626) | retrieve → `get_reasoning_llm` → `ResponseGenerator.generate_sync` → `update_interaction` → `ingest_memo_internal` (Q&A back-feed) → edge linking | `rag.reasoning_llm`, `rag.response_generator`, `pipeline.ingest_memo_internal` |
| `stats` (629–657) | `store.get_stats` → Rich Table | `GraphStore` |
| `prune` (660–751) | collect candidates → sort by importance → delete edges/nodes | `GraphStore`, `core.schemas.NodeType` |

### MCP tools — `src/mind_map/mcp/server.py`

| Tool | Body lines | Divergence from CLI |
|---|---|---|
| `mind_map_memo` (88–106) | `get_processing_llm()` + `ingest_memo_internal` | **Wrong pipeline**: uses `internal` (legacy LangChain path) instead of CLI's `cli` (direct MiniMax / local) — **silent behavioral drift** |
| `mind_map_retrieve` (59–86) | `query_similar` → `enrich_context_nodes` → flat bullet output | **Missing**: `--show-context`, `--max-context-per-node` (returns no neighbor expansion) |
| `mind_map_ask` | **(does not exist)** | **Missing entirely** — biggest gap |
| `mind_map_stats` (108–124) | `get_stats` → text summary | Missing `--data-dir` |
| `mind_map_prune` (217–348) | Same algorithm as CLI prune | **Missing `--percent`** (hardcoded 0.1); missing `--data-dir` |
| `mind_map_report` (126–214) | Top-5 importance + JSON | MCP-only; no CLI equivalent. Keep as-is. |
| `mind_map_health` (351–569) | Ollama + ChromaDB + SQLite + LLM + 3 integration tests | MCP-only; no CLI equivalent. Keep as-is. |

### Duplication matrix

`prune` algorithm — **identical logic, duplicated** between CLI lines 682–747
and MCP lines 234–333. ~65 lines of pure duplication.

`stats` — `get_stats()` is shared; only the presentation differs (Rich table
vs plain text).

`retrieve` algorithm — same `query_similar` → `enrich_context_nodes` →
`get_connected_context` flow, but MCP drops the context-enrichment step.

`memo` — **different pipeline entry point** (`internal` vs `cli`), different
default target resolution.

### Locked contracts (from tests)

These behaviors MUST be preserved exactly. Refactor is correct only if all of
these tests still pass without modification.

1. **`test_cli_retrieve.py`** — locks CLI `retrieve` output format:
   - Header `### Relevant Context from Mind Map:`
   - `- [<type>] (Relevance: <score>): <doc>` lines
   - `└─ related [<type>] via <relation>: <doc>` indentation (2-space indent + `└─`)
   - `--no-context` suppresses related lines
   - `--max-context-per-node` truncates neighbors to top-N by weight desc
   - Exit code 0 always (even on "no relevant info")
   - "No relevant information found" stdout on empty result

2. **`test_memo_cli_modes.py`** — locks CLI `memo` semantics:
   - `--data-dir` honored
   - `MINIMAX_API_KEY` env required when `--local` omitted (exit 1)
   - `--openclaw` option rejected (must NOT appear in help)
   - `--local ""` resolves to first model via `resolve_local_model`
   - `--local <model>` uses explicit model id
   - `target.base_url == "http://127.0.0.1:11435/v1"` for LocalTarget
   - `ingest_memo_cli` called with `target=...` kwarg

3. **`test_mcp_health.py`** — locks MCP `mind_map_health` contract:
   - JSON output with keys: `status`, `checks`, `timestamp`, `workspace`
   - `status` ∈ `{"healthy", "degraded", "unhealthy"}`
   - `checks` contains: `ollama_connection`, `chromadb_connection`, `sqlite_connection`, `processing_llm`, `integration_tests`
   - `integration_tests` contains: `similarity_search`, `memo_ingestion`, `data_persistence`
   - Integration tests MUST clean up after themselves (node count unchanged)
   - `mind_map.mcp.server.DEFAULT_DATA_DIR` is patchable (test mocks it)
   - `mind_map.mcp.server.ingest_memo` alias must exist (back-compat for
     `patch("mind_map.mcp.server.ingest_memo", ...)`) — see line 322 of test
   - `mind_map.mcp.server.get_store` and `mind_map.mcp.server.stores` are
     module-level globals that tests patch directly
   - Error path returns valid JSON (not raise)

4. **`test_mcp_prune.py`** — locks MCP `mind_map_prune` contract:
   - JSON output keys: `deleted_nodes`, `deleted_tags`, `deleted_edges_count`, `summary`
   - `summary` contains workspace name + counts
   - `deleted_node` shape: `{id, document, type}`
   - `deleted_tag` shape: `{id, document}`
   - Tags only pruned if ALL edges connect to pruned nodes
   - Entities are eligible; tags are not direct candidates
   - `summary_mentions_counts` requires `"default"` literal string in summary
   - `mind_map.mcp.server.DEFAULT_DATA_DIR` patchable
   - `mind_map.mcp.server.get_store` patchable

5. **`test_mcp_report.py`** — locks MCP `mind_map_report` contract:
   - JSON output keys: `summary`, `top_nodes`
   - `summary` shape: `{workspace, total_nodes, total_edges, concepts, entities, tags, avg_connections}`
   - Top 5 nodes by importance, with `{id, document, type, importance_score, connection_count, edges, tags}`
   - Multi-workspace isolation

6. **`test_pipeline.py`** — locks `ingest_memo_cli` requires `target` kwarg at
   call boundary (TypeError if missing). Back-compat alias `ingest_memo` for
   older test patches lives in `server.py` line 19.

### `workspace_id` — reclassification

Originally proposed: **drop** `workspace_id` from MCP (CLI has no analog).
**Revised: keep** it. Reasoning:
- All 3 MCP test suites (health/prune/report) assert workspace semantics
- `test_mcp_health.py` line 269 asserts `reasoning_llm` is NOT in `checks` — confirms we read the same code as tests
- Tests patch `mind_map.mcp.server.DEFAULT_DATA_DIR` and `mind_map.mcp.server.get_store` directly
- `stores` global dict is part of the public MCP module surface

**However**, `workspace_id` is genuinely MCP-only. To follow "CLI is source of
truth", we should ALSO expose workspace selection on the CLI. Recommendation:
add `--workspace/-w` to `memo`, `retrieve`, `ask`, `stats`, `prune` as an
optional override that maps to `data_dir/<workspace>` subfolder. This makes
the two surfaces truly symmetric.

---

## Proposed Architecture

```
                ┌─────────────────────────────────────┐
                │     src/mind_map/app/services.py    │  ← NEW
                │                                     │
                │  Pure logic functions:              │
                │   • memo_ingest(...)                │
                │   • retrieve_context(...)           │
                │   • ask_question(...)               │
                │   • graph_stats(...)                │
                │   • prune_graph(...)                │
                │   • report_graph(...)               │
                │   • health_check(...)               │
                │   • resolve_store(workspace_id,     │
                │                     data_dir)       │
                │                                     │
                │  No Typer, no Rich, no FastMCP.     │
                │  All return plain data + status.    │
                └──────────────┬──────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
   ┌────────▼─────────┐                ┌──────────▼──────────┐
   │ CLI main.py      │                │ MCP server.py        │
   │ (Typer commands) │                │ (FastMCP tools)      │
   │                  │                │                      │
   │ Wraps services   │                │ Wraps services       │
   │ + Rich output    │                │ + JSON/str return    │
   └──────────────────┘                └──────────────────────┘
```

---

## Step-by-Step Plan

### Phase 1 — Extract `services.py` (RED → GREEN refactor, behavior-preserving)

**Goal**: Create `src/mind_map/app/services.py` with pure logic functions.
Both CLI and MCP still call their own code in this phase. The point is to
land the new module + tests in isolation, then migrate callers one by one.

1. **Create `src/mind_map/app/services.py`** with these functions (signatures
   revised after review):

   ```python
   def resolve_store(
       *,
       data_dir: Path | str | None = None,  # CLI passes --data-dir; MCP can override or use workspace
       workspace_id: str | None = None,     # MCP-only; CLI ignores
   ) -> tuple[GraphStore, Path]:
       """Resolve data dir + workspace, initialize + return store.
       Returns (store, effective_data_dir).
       Precedence:
         1. data_dir provided → use it directly (ignore workspace_id)
         2. workspace_id provided → DEFAULT_DATA_DIR/workspaces/<id>/
         3. neither → DEFAULT_DATA_DIR (default workspace)
       Workspace subfolder layout is MCP-only. CLI passes data_dir and
       never sets workspace_id."""

   def parse_memo_target(
       *,
       local: str | None,
       api_key: str | None,
   ) -> MemoTarget:
       """Pure helper. Resolves LocalTarget (model id or auto via /models)
       or MiniMaxTarget (api_key required). Raises CLIExecutionError on
       local resolution failure."""

   def memo_ingest(
       text: str,
       store: GraphStore,
       *,
       local: str | None = None,        # "" = auto-resolve first model
       source: str | None = None,
   ) -> tuple[bool, str, list[str]]:
       """Mirrors `mind-map memo` exactly.
       Reads MINIMAX_API_KEY from env when `local` is None.
       Raises if neither local nor MINIMAX_API_KEY is available.
       Returns (success, message, node_ids).
       Caller (CLI/MCP) is responsible for the `data_dir.exists()` check
       BEFORE calling this — preserves CLI exit code 1 behavior."""

   def retrieve_context(
       query: str,
       store: GraphStore,
       *,
       n_results: int = 5,
       show_context: bool = True,
       max_context_per_node: int = 3,
   ) -> list[str]:
       """Returns formatted lines (one per match + zero or more `└─ related`
       lines). Caller joins with newline. Mirrors CLI `retrieve` exactly.
       Returns ["No relevant information found in the knowledge graph."]
       when query_similar returns nothing.
       Caller (CLI/MCP) is responsible for the `data_dir.exists()` check
       BEFORE calling this — CLI prints that message and exits 0."""

   def ask_question(
       query: str,
       store: GraphStore,
       *,
       depth: int = 2,                  # unused, kept for CLI parity
       model: str | None = None,        # processing LLM override for back_feed
       back_feed: bool = False,         # CLI ask always back-feeds; MCP defaults off
   ) -> dict[str, Any]:
       """Mirrors `mind-map ask` when back_feed=True.
       When back_feed=False: pure read — retrieve + LLM answer, no writes.
       Returns dict: {response, context_nodes, qa_node_ids, status}.
       When no reasoning LLM available: response = "Reasoning LLM not
       available. Raw context:\\n[bulleted nodes]". status = "no_llm".
       Caller formats."""

   def graph_stats(store: GraphStore) -> dict[str, Any]:
       """Returns normalized stats dict: total_nodes, total_edges,
       concept_nodes, entity_nodes, tag_nodes, avg_connections."""

   def format_stats_text(stats: dict[str, Any], *, workspace_id: str = "default") -> str:
       """Returns the text block format that MCP `mind_map_stats` currently
       produces. Locked by test contracts."""

   def prune_graph(
       store: GraphStore,
       *,
       percent: float = 0.1,
   ) -> dict[str, Any]:
       """Returns {deleted_nodes, deleted_tags, deleted_edges_count, summary}.
       Mirrors MCP prune JSON contract exactly."""

   def report_graph(store: GraphStore) -> dict[str, Any]:
       """Returns {summary, top_nodes} matching MCP report contract exactly."""

   def health_check(
       store: GraphStore,
       *,
       workspace_id: str,
   ) -> dict[str, Any]:
       """Returns the full health-check dict. Mirrors MCP health JSON exactly.
       Keeps the 3 integration tests inline.
       Uses `ingest_memo` re-exported from `mind_map.app.services` (which
       points to `ingest_memo_internal`) for back-compat with test patches."""

   # Back-compat re-export for `patch("mind_map.mcp.server.ingest_memo", ...)`
   ingest_memo = ingest_memo_internal  # from mind_map.app.pipeline
   ```

2. **Write `tests/test_services.py`** with **unit tests** for each function
   (no CLI invocation, no MCP server — just direct function calls). At
   minimum, port the existing assertions from `test_cli_retrieve.py` and
   `test_mcp_prune.py` into `test_services.py::test_retrieve_context_*` and
   `test_services.py::test_prune_graph_*`. These become the single contract
   both surfaces must satisfy.

3. **Run `poetry run pytest`** — expect existing CLI/MCP tests to still
   pass (we haven't touched callers yet).

### Phase 2 — Migrate CLI commands to `services.py`

For each CLI command in `src/mind_map/app/cli/main.py`:

1. Replace the inline body with a thin wrapper that:
   - Calls the corresponding `services.py` function
   - Formats output with Rich
   - Handles `typer.Exit` codes
2. **Do not change** `--data-dir`, `--local`, `--source`, `--show-context`,
   `--max-context-per-node`, `--percent`, `--model` flag names or defaults
3. Run `poetry run pytest tests/test_cli_retrieve.py tests/test_memo_cli_modes.py`
   — must pass unchanged

Specifically:
- `memo` (lines 405–475) → 20 lines: validate env → call `memo_ingest` → print
- `retrieve` (478–531) → 15 lines: call `retrieve_context` → join → print
- `ask` (534–626) → 25 lines: call `ask_question` → render 2 Rich panels
- `stats` (629–657) → 10 lines: call `graph_stats` → render Rich Table
- `prune` (660–751) → 15 lines: call `prune_graph` → render status

### Phase 3 — Migrate MCP tools to `services.py`

For each `@mcp.tool()` in `src/mind_map/mcp/server.py`:

1. Replace the inline body with a thin wrapper that:
   - Resolves store via `services.resolve_store(...)` — **but keep the
     `stores` global dict + `get_store()` helper for back-compat with tests**
   - Calls the corresponding `services.py` function
   - Returns string (or JSON string where contract requires)
2. **Add new MCP tools for CLI parity**:
   - `mind_map_ask(query, depth: int = 2, n_results: int = 5, data_dir: str | None = None, model: str | None = None, back_feed: bool = False, workspace_id: str | None = None)` — uses `services.ask_question`. **Read-only by default** (no Q&A write-back, no edge linking, no `update_interaction`). Tool description must state this. `back_feed=True` opts into CLI-style write behavior. `depth` is a dead flag kept for CLI parity (unused in CLI implementation).
3. Update existing tools to accept new params:
   - `mind_map_memo(text, source: str | None = None, local: str | None = None, data_dir: str | None = None, workspace_id: str | None = None)` — uses `services.memo_ingest`. **Requires `MINIMAX_API_KEY` when `local` is None** (CLI parity). Closes the silent drift to `get_processing_llm()` cloud-auto.
   - `mind_map_retrieve(query, n_results: int = 5, show_context: bool = True, max_context_per_node: int = 3, data_dir: str | None = None, workspace_id: str | None = None)` — uses `services.retrieve_context`
   - `mind_map_stats(data_dir: str | None = None, workspace_id: str | None = None)` — uses `services.graph_stats` + `services.format_stats_text`
   - `mind_map_prune(percent: float = 0.1, data_dir: str | None = None, workspace_id: str | None = None)` — uses `services.prune_graph`
   - `mind_map_report(data_dir: str \| None = None, workspace_id: str \| None = None)` — uses `services.report_graph`
   - `mind_map_health(data_dir: str | None = None, workspace_id: str | None = None)` — uses `services.health_check`
4. **`data_dir` precedence in MCP tools** (CLI parity): when `data_dir` is provided, use it directly and ignore `workspace_id`. When only `workspace_id` is provided, use `DEFAULT_DATA_DIR/workspaces/<id>/` (existing MCP behavior preserved). When neither, use `DEFAULT_DATA_DIR` (default workspace). Encapsulate this in `services.resolve_store(data_dir=, workspace_id=)`.
4. **Preserve the `ingest_memo` alias** (line 19): `ingest_memo = ingest_memo_internal`.
   `services.health_check` will reference `services.ingest_memo` (which is the
   same alias), and tests patch `mind_map.mcp.server.ingest_memo`. The cleanest
   approach is to **also expose `ingest_memo` as a module-level re-export** in
   `server.py` that points to the alias in `pipeline.py`. This means the test
   patch `patch("mind_map.mcp.server.ingest_memo", side_effect=...)` continues
   to work even though the actual call site moved to `services.py`.
5. Keep `DEFAULT_DATA_DIR`, `stores`, `get_store()` exactly as they are —
   they are public test surface
6. Run `poetry run pytest tests/test_mcp_health.py tests/test_mcp_prune.py tests/test_mcp_report.py`
   — must pass unchanged

### Phase 4 — Register MCP server with Hermes

Add to `~/.hermes/config.yaml` under `mcp_servers:` (lines 715–740):

```yaml
  mind-map:
    command: /Users/gwansun/Desktop/projects/mind-map/.venv/bin/python
    args:
      - -m
      - mind_map.mcp.server
    workdir: /Users/gwansun/Desktop/projects/mind-map
    env:
      MINIMAX_API_KEY: ${MINIMAX_API_KEY}
```

(Shape mirrors `cdi-pipeline` entry above it.)

**Verification**:
```bash
hermes mcp list                    # should show mind-map: ✓ enabled
hermes mcp inspect mind-map        # should list 7 tools
```

### Phase 5 — Final validation

1. **Full test suite**: `cd ~/Desktop/projects/mind-map && poetry run pytest`
   — must pass (all 15 files, ~3.5k LOC). No test edits allowed except adding
   `test_services.py` in Phase 1.
2. **Manual smoke test**: rebuild & install the wheel
   (`poetry build --format wheel && uv tool uninstall mind-map && uv tool install ...`)
   per the existing `references/runtime-installation-drift.md` pattern, then
   verify `mind-map memo`, `mind-map retrieve`, `mind-map ask`, `mind-map stats`,
   `mind-map prune` all behave identically to pre-refactor.
3. **MCP end-to-end**: from this Discord session, call
   `mcp_mind_map__mind_map_retrieve(query="...", n_results=3)` and confirm
   the output matches CLI `mind-map retrieve --n-results 3`.
4. **Commit**: one commit for code (`PLAN.McpFollowsCli.md`,
   `src/mind_map/app/services.py`, refactored `main.py` + `server.py`), one
   commit for tests (`tests/test_services.py`), one commit for docs
   (`README.md`, `CLAUDE.md` updates if any).

---

## Risks & Open Questions

### Risks

| Risk | Mitigation |
|---|---|
| `services.ask_question` triggers `update_interaction` + Q&A back-feed — both are **write operations**. A misconfigured MCP caller could spam the KG with Q&A pairs. | Per decision #2: `back_feed: bool = False` default (read-only by default). CLI `ask` keeps its write behavior. Surface the difference in the tool description. |
| When `services.ask_question` is called and no reasoning LLM is available, CLI `ask` falls back to printing raw context nodes (lines 571–579 of main.py). MCP must return a string instead. | Define a return contract: when no LLM available, return `"Reasoning LLM not available. Raw context:\n[bulleted nodes]"` — string form of CLI's fallback path. No error, just an honest signal. |
| Health check integration tests in MCP pollute the store on transient failures if cleanup throws. Existing code handles this with try/except cleanup — preserve exactly. | Don't refactor the cleanup logic — only the dispatch. |
| The `depth` flag on CLI `ask` is declared but unused (lines 537, 544 in main.py). Refactor shouldn't "fix" this — keep parity. | Leave dead flag in service signature with `# unused, kept for CLI parity` comment. |
| The `ingest_memo` back-compat alias in `server.py:19` is patched by tests. If `services.py` imports `ingest_memo_internal` directly and tests patch only `mind_map.mcp.server.ingest_memo`, the patch won't take. | Re-export in `server.py`: `from mind_map.app.services import ingest_memo as ingest_memo`. This makes the test patch `patch("mind_map.mcp.server.ingest_memo", side_effect=...)` continue to work even though the actual call site moved to `services.py`. |
| `parse_memo_target` and `memo_ingest` both could read `MINIMAX_API_KEY` from env. If both do, error message is duplicated. If neither does, callers must pass it explicitly. | **Decision**: `memo_ingest` reads the env var (CLI parity). `parse_memo_target` is a pure helper that takes `api_key` as a param. CLI `memo` passes nothing → service reads env. MCP `memo` does the same. |
| `resolve_store(workspace_basedir=...)` parameter is invented — not used by current MCP code which hardcodes `DEFAULT_DATA_DIR` as the workspace parent. | **Drop the param.** `resolve_store` uses `DEFAULT_DATA_DIR` from `core.config` as the parent. Matches current MCP behavior. CLI passes `data_dir` directly without workspace concept. |
| `services.health_check(workspace=...)` and `services.resolve_store(workspace_id=...)` use different naming for the same concept. | **Standardize on `workspace_id`** everywhere. Update `health_check` signature. |
| `mind_map_stats` MCP return format (text block) is a test contract, but `services.graph_stats` returns a dict. Both CLI (Rich Table) and MCP (text block) need to format differently from the same dict. | Define `services.format_stats_text(stats_dict) -> str` helper. CLI uses Rich Table from the dict directly; MCP uses the text helper. Add unit tests for the text helper. |
| Wheel rebuild happens at end of Phase 5 but pytest runs earlier. There's a window where pytest passes against repo but global `mind-map` binary still points at old code. | **Accepted** per your decision. Document the gap window in the PR description. Final commit includes both the refactor and the wheel rebuild commands. |

### Decisions (confirmed)

| # | Decision | Resolution |
|---|---|---|
| 1 | MCP `memo` default target | **Require `MINIMAX_API_KEY`** (CLI parity). If unset AND no `--local` passed, error out — never silently fall through to `get_processing_llm()` cloud-auto. This closes the silent behavioral drift. |
| 2 | MCP `ask` writes back to graph | **Off by default.** `mind_map_ask` will be a **pure read** — no Q&A back-feed, no edge linking, no `update_interaction`. CLI `ask` keeps its existing write-back behavior (untouched). Surface this difference in the tool description. Add a `back_feed: bool = False` param for explicit opt-in. |
| 3 | Workspace scope | **Keep MCP-only.** Do NOT add `--workspace` to CLI. Justification: the `mind-map` binary version is what users interact with, and adding `--workspace` to it would be premature until there's a real multi-workspace use case. The MCP-only subfolder layout (`data_dir/workspaces/<id>/`) is internal MCP plumbing. |
| 4 | Installed wheel drift | **Confirmed in sync** (verified Jun 26 19:53). `mind-map` (uv tool) and `poetry run mind-map` produce byte-identical help output for top-level + `memo`. Default data dir `/Users/gwansun/mind-map/data` is the same in both. Error message for `--data-dir /nonexistent` is identical. Wheel was built May 27, matches `dist/` build timestamp. **No reinstall needed at start; will rebuild at the very end of Phase 5, accepting a gap window where pytest passes against the repo but the global `mind-map` binary still points at the old code.** |

### Non-goals

- No new features beyond CLI parity (e.g., no batch memo, no streaming ask)
  - **Exception**: adding `local` parameter to `mind_map_memo` is a **new capability** (MCP callers previously had no way to choose a local model target). This is intentional and gets its own commit message callout.
- No changes to the ingestion pipeline internals (`build_memo_cli_ingestion_pipeline`,
  `build_legacy_ingestion_pipeline`) — just the dispatch
- No changes to GraphStore, ChromaDB, SQLite, embeddings, or scoring
- No new FastMCP server features (auth, HTTP transport, etc.)
- No changes to the `serve` (FastAPI) command — `app/api/routes.py` keeps its own `get_store` import. If desired later, that's a separate refactor.
- No changes to the `model` subcommands (`model list/get/set/pull/select`) — CLI-only by design, no MCP analog.

---

## File-Level Diff Summary (predicted)

```
src/mind_map/app/services.py         | +480 lines  (NEW)
src/mind_map/app/cli/main.py         | -220 / +90  (5 commands shrink to wrappers)
src/mind_map/mcp/server.py           | -300 / +80  (7 tools shrink to wrappers)
tests/test_services.py               | +320 lines  (NEW — covers all 8 functions + format_stats_text)
PLAN.McpFollowsCli.md                | +500 lines  (this file, NEW)
~/.hermes/config.yaml                | + 10 lines  (mind-map MCP entry)
```

**Net**: ~70 lines removed from CLI/MCP duplication, ~480 lines added in
focused services module, ~10 lines in Hermes config. Same behavior, half the
code paths, one source of truth. No README/CLAUDE.md changes needed (workspaces stay MCP-only, no user-facing changes beyond tool addition).

## Working branch & rollback

- **Branch**: `refactor/mcp-follows-cli` off `main` (or current default branch)
- **Commit order**:
  1. `feat(mind-map): add services.py with pure logic functions` (services + tests, both pass)
  2. `refactor(cli): migrate commands to services.py` (CLI tests pass)
  3. `feat(mcp): migrate tools to services.py + add mind_map_ask + local param` (MCP tests pass)
  4. `chore(hermes): register mind-map MCP server` (config change)
  5. `chore(mind-map): rebuild and reinstall wheel` (closes the gap window)
- **Rollback**: `git revert HEAD~5..HEAD` reverts all 5 commits in dependency order (last to first). Or per-phase: `git revert HEAD` on the specific phase commit. Services.py removal is safe — CLI/MCP keep their inline logic as fallback if we revert the migration commits but keep services.py.
- **Per CLAUDE.md convention**: code (`services.py`, refactored `main.py`, refactored `server.py`) in one commit; tests (`test_services.py`) in a separate commit; docs (`PLAN.McpFollowsCli.md`, config) in a third commit. Adjust the 5-commit plan above to merge adjacent pure-code commits if they're tiny.

---

## Validation Checklist (sign-off criteria)

- [ ] All 15 existing test files pass without modification
- [ ] `tests/test_services.py` covers each services function with ≥3 cases
- [ ] `mind-map memo --help` shows same flags as before (no `--openclaw`)
- [ ] `mind-map memo "test" --data-dir /tmp/x --local ""` works (auto-resolve)
- [ ] `mind-map retrieve "q" --show-context --max-context-per-node 2` matches pre-refactor output
- [ ] `mind-map ask "q"` writes a Q&A node back (verify `mind-map stats` increments)
- [ ] `mind-map prune --percent 0.2` honors custom percent
- [ ] `hermes mcp list` shows `mind-map` as enabled
- [ ] From Discord: `mcp_mind_map__mind_map_retrieve(query="test", n_results=3, data_dir="/tmp/x")` returns expected format (CLI parity on data_dir)
- [ ] From Discord: `mcp_mind_map__mind_map_retrieve(query="test", workspace_id="alice")` works (MCP-only workspace layout)
- [ ] From Discord: `mcp_mind_map__mind_map_ask(query="test")` returns LLM-generated answer **and does NOT increment node count** (read-only default)
- [ ] From Discord: `mcp_mind_map__mind_map_ask(query="test", back_feed=True)` returns LLM-generated answer **and DOES increment node count** (CLI parity opt-in)
- [ ] From Discord: `mcp_mind_map__mind_map_memo(text="test")` with no `MINIMAX_API_KEY` returns error string (no silent fallback)
- [ ] From Discord: `mcp_mind_map__mind_map_prune(percent=0.2)` honors custom percent (CLI parity)
- [ ] `mcp_mind_map__mind_map_health()` returns JSON with `status` field
- [ ] No new deprecation warnings from FastMCP or Typer
- [ ] `mind-map` installed wheel rebuilt + reinstalled to match repo (`poetry build --format wheel && uv tool uninstall mind-map && uv tool install dist/mind_map-*.whl`)
