# Revised Code Plan for OpenClaw v2026.4.9

> Purpose: Rebase the existing `CODING_PLAN.md` onto the current OpenClaw `v2026.4.9` code structure before implementation.
> Status: Compatibility-adjusted plan for the cloned repo at `custom_openclaw_v2026.4.9/openclaw`

---

## 1. Goal

Carry forward the same two features from the original plan, but align them to the current repo structure:

1. **Language restriction**
   - Hardcode a response-language restriction to English and Korean.
   - Reinforce it in both the system prompt and per-message body.

2. **Mind map pre-fetch**
   - Automatically retrieve relevant mind-map context for non-trivial inbound messages before the model runs.
   - Inject the retrieved context into the message body, not the system prompt.

---

## 2. Current Repo Findings (v2026.4.9)

These were verified in the cloned repo.

### 2.1 System prompt builder still exists and is still the right place

File:
- `src/agents/system-prompt.ts`

Still true:
- `buildAgentSystemPrompt()` is the central builder.
- `promptMode === "none"` still has an early return.
- Tool-call narration guidance is still embedded directly in this file.
- Runtime section is still appended near the end with `lines.push(...)`.

### 2.2 Reply pipeline still supports message preprocessing hooks

File:
- `src/auto-reply/reply/get-reply.ts`

Current order relevant to this work:
1. `finalizeInboundContext(ctx)`
2. `applyMediaUnderstandingIfNeeded(...)` inside `!isFastTestEnv`
3. `applyLinkUnderstandingIfNeeded(...)` inside `!isFastTestEnv`
4. `emitPreAgentMessageHooks(...)`
5. session/directive/model flow

This means the correct insertion point for mind-map enrichment is still:
- after link understanding
- before `emitPreAgentMessageHooks(...)`

### 2.3 Reference pattern still valid

File:
- `src/link-understanding/apply.ts`

Still true:
- mutates `ctx.Body`
- calls `finalizeInboundContext(ctx, { forceBodyForAgent: true, forceBodyForCommands: true })`

This remains the best reference shape for message-body enrichment.

### 2.4 `finalizeInboundContext` is still compatible

File:
- `src/auto-reply/reply/inbound-context.ts`

Still true:
- supports `forceBodyForAgent`
- supports `forceBodyForCommands`
- normalizes `Body`, `BodyForAgent`, and command fields

### 2.5 There is no existing `src/mind-map/` directory

So this feature can still be introduced as a new module family without colliding with an upstream implementation.

---

## 3. Required Adjustments from the Old Plan

The original `CODING_PLAN.md` is directionally right, but these details must change for `v2026.4.9`:

1. **Prompt wording changed**
   - Current base string is:
     - `You are a personal assistant operating inside OpenClaw.`
   - Do not patch against the older `running inside OpenClaw` wording.

2. **Tool-call narration string changed location/wording stability assumptions**
   - The line still exists, but implementation should patch the current exact line:
     - `Use plain human language for narration unless in a technical context.`

3. **Runtime hint insertion must use the current final runtime push block**
   - The old plan is right conceptually, but should be applied to the current `lines.push("## Runtime", ...)` section.

4. **Testing must be included in implementation**
   - Current OpenClaw has a broader test surface than the older plan assumes.
   - Any implementation should add or update tests for:
     - system prompt output
     - message preprocessing path
     - duplicate-prevention logic
     - fail-open behavior

---

## 4. Revised Implementation Plan

## 4.1 New module family: `src/mind-map/`

Create these files:

### `src/mind-map/retrieve.ts`
Purpose:
- Provide `retrieveMindMapContext(query, dataDir, count?)`
- Use `execFile` / `execFileSync`-style child-process execution without shell interpolation
- Call:

```bash
mind-map retrieve --data-dir <path> --n-results <count> <query>
```

Behavior:
- 10s timeout
- return stdout string on success
- return `undefined` on error / timeout / empty output
- fail-open only

### `src/mind-map/format.ts`
Purpose:
- `formatMindMapBody(body, context)`
- `isSkippableQuery(body)`

Behavior:
- wrap retrieved payload inside:

```text
<mind_map_context>
...
</mind_map_context>
```

- preserve original body below or above consistently
- skip if body is:
  - shorter than 3 chars
  - starts with `/`
  - matches simple greeting/ack regex

Suggested starter regex from original plan is still acceptable:

```text
/^(hi|ok|hello|thanks|thank you|hey|yo|sup|bye|goodbye|yes|no|sure|k|okay|안녕|ㅎㅇ|하이)\b/i
```

### `src/mind-map/apply.ts`
Purpose:
- `applyMindMapContext({ ctx, cfg })`

Behavior:
1. cached PATH check for `mind-map`
2. skip if body is trivial
3. skip if `<mind_map_context>` already exists in `ctx.Body`
4. run retrieval on full `ctx.Body`
5. if no result, return silently
6. inject formatted body into `ctx.Body`
7. call:

```ts
finalizeInboundContext(ctx, {
  forceBodyForAgent: true,
  forceBodyForCommands: true,
});
```

Note:
- Use both `forceBodyForAgent` and `forceBodyForCommands` to match the current enrichment style used by `link-understanding/apply.ts`.
- This is slightly better aligned to `v2026.4.9` than the older plan’s `forceBodyForAgent`-only wording.

### `src/mind-map/language-restriction.ts`
Purpose:
- export constants:

```ts
export const LANGUAGE_DIRECTIVE = "[Respond in English or Korean only.]";
export const LANGUAGE_NARRATION_HINT = "Use specifically English or Korean for narration unless in a technical context.";
```

- export:

```ts
applyLanguageRestriction({ ctx })
```

Behavior:
- append `LANGUAGE_DIRECTIVE` to `ctx.Body` only if not already present
- then run:

```ts
finalizeInboundContext(ctx, {
  forceBodyForAgent: true,
  forceBodyForCommands: true,
});
```

Rationale:
- In current OpenClaw, keeping command/body variants aligned is safer than updating only `BodyForAgent`.

### `src/mind-map/index.ts`
Purpose:
- re-export all public functions/constants

---

## 4.2 System prompt changes in `src/agents/system-prompt.ts`

Add imports from:
- `../mind-map/language-restriction.js`

### Change A: `promptMode === "none"`
Current code returns:

```ts
return "You are a personal assistant operating inside OpenClaw.";
```

Revised target:

```ts
return [
  "You are a personal assistant operating inside OpenClaw.",
  LANGUAGE_DIRECTIVE,
].join("\n");
```

### Change B: Tool Call Style narration line
Replace current literal:

```ts
"Use plain human language for narration unless in a technical context."
```

with:

```ts
LANGUAGE_NARRATION_HINT
```

### Change C: Runtime tail hints
At the final runtime section append both:

```ts
LANGUAGE_DIRECTIVE
```

and:

```ts
"If <mind_map_context> is already present in the user message, skip the initial retrieve (context was pre-fetched)."
```

Apply this in the current runtime `lines.push(...)` block near the end of `buildAgentSystemPrompt()`.

---

## 4.3 Reply pipeline changes in `src/auto-reply/reply/get-reply.ts`

Add new imports/runtime loader pattern as needed for the new module.

### Current correct insertion point
Inside:

```ts
if (!isFastTestEnv) {
  await applyMediaUnderstandingIfNeeded(...)
  await applyLinkUnderstandingIfNeeded(...)
}
emitPreAgentMessageHooks(...)
```

Revised target:

```ts
if (!isFastTestEnv) {
  await applyMediaUnderstandingIfNeeded(...)
  await applyLinkUnderstandingIfNeeded(...)
  await applyMindMapContext({ ctx: finalized, cfg })
}

applyLanguageRestriction({ ctx: finalized })

emitPreAgentMessageHooks(...)
```

Why this ordering:
- mind-map retrieval is external I/O, so keep it inside `!isFastTestEnv`
- language restriction is pure string mutation, so always apply it
- hooks should observe the final enriched body, not the pre-enrichment body

This is a small improvement over the old plan, which placed `applyLanguageRestriction()` after the guarded block but did not explicitly account for current hook timing.

---

## 5. Hardcoded Data / Environment Assumptions

Use the same currently intended hardcoded data dir unless changed deliberately:

```text
/Users/gwansun/.openclaw/workspace/projects/mind-map/data
```

Binary lookup policy remains:
- resolve `mind-map` from PATH
- no auto-install in request path
- cache presence result for process lifetime

---

## 6. Revised Verification Checklist for v2026.4.9

## 6.1 Source compatibility

Verify before implementation:
- `src/agents/system-prompt.ts`
  - `promptMode === "none"` early return still exists
  - current narration string still exists
  - runtime section still uses final `lines.push(...)`
- `src/auto-reply/reply/get-reply.ts`
  - `applyMediaUnderstandingIfNeeded(...)` still exists
  - `applyLinkUnderstandingIfNeeded(...)` still exists
  - `emitPreAgentMessageHooks(...)` still follows those calls
- `src/link-understanding/apply.ts`
  - still uses `finalizeInboundContext(... forceBodyForAgent/Commands ... )`

## 6.2 Functional verification

### Language restriction
Expected:
- system prompt contains `[Respond in English or Korean only.]`
- `promptMode === "none"` output includes the directive
- tool narration line uses the English/Korean-specific hint
- message body gets the directive only once

### Mind-map prefetch
Expected:
- non-trivial message + binary present => injected `<mind_map_context>` block
- trivial body => no retrieval
- body already containing `<mind_map_context>` => no duplicate injection
- binary missing => no error, feature silently skipped
- CLI timeout/error => no error thrown upstream

### Hook visibility
Expected:
- `emitPreAgentMessageHooks(...)` sees final enriched text after:
  - link understanding
  - mind-map injection
  - language directive append

This should be verified because the current repo has explicit message preprocessing hooks.

---

## 7. Recommended Tests to Add or Update

At minimum, add tests for:

1. **System prompt**
   - `LANGUAGE_DIRECTIVE` appears in full mode
   - `LANGUAGE_DIRECTIVE` appears in none mode
   - narration hint uses `LANGUAGE_NARRATION_HINT`

2. **Message hook ordering**
   - verify pre-agent hooks receive enriched `BodyForAgent`

3. **Mind-map apply logic**
   - skip on trivial query
   - skip on duplicate `<mind_map_context>`
   - successful injection updates `Body`, `BodyForAgent`, and `BodyForCommands`
   - failure is silent

4. **Language apply logic**
   - appends once only
   - updates command/body variants consistently

---

## 8. Final Compatibility Verdict

For `v2026.4.9`:
- **Architecture:** compatible
- **Original wording/patch targets:** stale
- **Implementation path:** still good after rebasing

Practical guidance:
- Use the original `CODING_PLAN.md` as the historical rationale
- Use this revised plan as the actual implementation checklist for `v2026.4.9`

---

## 9. Next Step

After this revised plan, the next safe step is:
1. implement against the cloned repo using this rebased checklist
2. add/update tests immediately with the source changes
3. verify prompt output and message-preprocess behavior before any gateway rollout
