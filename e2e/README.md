# Mind Map E2E Tests

Playwright-based end-to-end/integration tests for the mind-map frontend, focusing on the node deletion flow.

## What Gets Covered

| Test | What it verifies |
|------|-----------------|
| `confirm delete → node disappears` | Click node → inspector opens → click Delete → confirm → node removed, inspector closes |
| `cancel delete does not remove node` | Click node → click Delete → click Cancel → node still in graph, inspector still open |
| `correct confirmation message` | Delete confirmation shows the node label + connection count + both action buttons |
| `canvas click deselects node` | Click canvas bg → inspector closes, no delete triggered |

## Prerequisites

Both servers must be **running before** executing the tests:

```bash
# Terminal 1: Backend API (FastAPI, port 8000)
cd /Users/gwansun/Desktop/projects/mind-map
mind-map serve

# Terminal 2: Frontend dev server (Angular, port 4200)
cd /Users/gwansun/Desktop/projects/mind-map/frontend
npm start
```

> The test suite seeds its own nodes, so no pre-existing data is required.

## Install

```bash
cd e2e
npm install
```

This installs `@playwright/test` locally in the `e2e/` folder. Chromium is already cached system-wide.

## Run

```bash
# From the e2e directory
cd e2e
npm run test

# Or directly via Playwright
node_modules/.bin/playwright test --config=playwright.config.js

# Run with headed browser (see the browser)
npm run test:headed

# View the HTML report
npm run report
```

## File Layout

```
e2e/
├── package.json            # Local @playwright/test install
├── playwright.config.js    # Test runner config (CommonJS, no ts-node required)
├── node-deletion.spec.js   # The actual test cases
└── README.md               # This file
```

## Browser

The tests use the Google Chrome for Testing binary bundled with Playwright 1.59.1 at:
```
~/.cache/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing
```

This is an arm64 macOS Chromium — no extra downloads needed on this machine.

## Gaps / Future Work

- **Cannot run without live servers** — a future `webServer` block or mock backend would enable fully offline execution
- **No edge deletion test** — edges are cascade-deleted when a node is deleted; a dedicated edge-delete test would require adding a direct edge delete button (not currently in the UI)
- **D3 simulation timing** — the test waits for circles to appear and for graph to settle; on very slow machines these timeouts may need to increase
- **Inspector close via X button** — the "×" close button in the inspector header is not covered (simple extension to the cancel test)
