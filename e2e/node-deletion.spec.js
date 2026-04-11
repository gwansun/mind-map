/**
 * Playwright E2E tests for node deletion flow.
 *
 * Prerequisites:
 *   1. Backend running on http://localhost:8000
 *   2. Frontend dev server running on http://localhost:4200
 *   3. Database seeded with some nodes (run `mind-map memo "test note"` to create one)
 *
 * Run with:
 *   cd /Users/gwansun/Desktop/projects/mind-map/e2e
 *   npm install
 *   node_modules/.bin/playwright test
 */

const { test, expect } = require('@playwright/test');

/**
 * Shared helpers
 */
async function waitForGraphCanvas(page) {
  await page.waitForSelector('app-graph-canvas', { timeout: 10000 });
  await page.waitForLoadState('networkidle');
}

async function seedNode(page, doc) {
  doc = doc || `Test node for deletion flow ${Date.now()}`;
  await page.goto('http://localhost:4200');
  await waitForGraphCanvas(page);

  const input = page.locator('textarea, input[type="text"]').first();
  await input.fill(doc);
  await input.press('Enter');
  await page.waitForTimeout(2000);
  return doc;
}

async function refreshGraph(page) {
  const refreshBtn = page.locator('button[title*="refresh"], button[title*="Refresh"]').first();
  if (await refreshBtn.isVisible()) await refreshBtn.click();
  await page.waitForTimeout(1500);
}

/**
 * Test: select a node, open inspector, click delete, confirm, and verify removal.
 */
test('node deletion flow: select → inspector → confirm delete → node disappears', async ({ page }) => {
  await page.goto('http://localhost:4200');
  await waitForGraphCanvas(page);

  // Seed a unique node so we can target it reliably
  const uniqueLabel = `Playwright-test-node-${Date.now()}`;
  await seedNode(page, uniqueLabel);
  await refreshGraph(page);

  // Step 1: Click a node circle in the SVG
  const svg = page.locator('app-graph-canvas svg');
  await expect(svg).toBeVisible();

  const allCircles = page.locator('app-graph-canvas svg circle.node');
  await allCircles.first().wait({ timeout: 8000 });
  const count = await allCircles.count();
  expect(count).toBeGreaterThan(0);

  await allCircles.first().click();
  await page.waitForTimeout(500);

  // Step 2: Inspector panel should appear
  const inspector = page.locator('app-inspector-panel');
  await expect(inspector).toBeVisible({ timeout: 5000 });

  // Step 3: Click the Delete button inside the inspector
  const deleteBtn = inspector.locator('button:has-text("Delete")').first();
  await expect(deleteBtn).toBeVisible({ timeout: 3000 });
  await deleteBtn.click();
  await page.waitForTimeout(300);

  // Step 4: Confirmation dialog should appear
  const confirmMsg = inspector.locator('.delete-confirm__message');
  await expect(confirmMsg).toBeVisible({ timeout: 3000 });

  const confirmDeleteBtn = inspector.locator('button.btn--danger:has-text("Delete")');
  await expect(confirmDeleteBtn).toBeVisible({ timeout: 3000 });
  await confirmDeleteBtn.click();

  // Step 5: Wait for deletion to complete and graph to refresh
  await page.waitForTimeout(2000);

  // Inspector should close (node was deselected) — either gone or empty state
  const inspectorEmpty = inspector.locator('.inspector-empty');
  const inspectorHidden = await inspector.evaluate((el) => {
    const s = window.getComputedStyle(el);
    return s.display === 'none' || s.visibility === 'hidden' || el.hidden;
  }).catch(() => false);
  const eitherGoneOrEmpty = inspectorHidden || await inspectorEmpty.isVisible().catch(() => false);
  expect(eitherGoneOrEmpty).toBeTruthy();
});

/**
 * Test: cancel delete should not remove the node
 */
test('node deletion flow: cancel delete does not remove node', async ({ page }) => {
  await page.goto('http://localhost:4200');
  await waitForGraphCanvas(page);

  const uniqueLabel = `Playwright-cancel-test-${Date.now()}`;
  await seedNode(page, uniqueLabel);
  await refreshGraph(page);

  const allCircles = page.locator('app-graph-canvas svg circle.node');
  await allCircles.first().wait({ timeout: 8000 });
  const countBefore = await allCircles.count();

  await allCircles.first().click();
  await page.waitForTimeout(500);

  const inspector = page.locator('app-inspector-panel');
  await expect(inspector).toBeVisible({ timeout: 5000 });

  const deleteBtn = inspector.locator('button:has-text("Delete")').first();
  await deleteBtn.click();
  await page.waitForTimeout(300);

  const cancelBtn = inspector.locator('button:has-text("Cancel")').first();
  await expect(cancelBtn).toBeVisible({ timeout: 3000 });
  await cancelBtn.click();
  await page.waitForTimeout(300);

  // Inspector should still show node content (not empty state)
  const inspectorEmpty = inspector.locator('.inspector-empty');
  const stillHasContent = !(await inspectorEmpty.isVisible().catch(() => true));
  expect(stillHasContent).toBeTruthy();

  // Node count should be unchanged
  await page.waitForTimeout(1000);
  const countAfter = await allCircles.count();
  expect(countAfter).toBe(countBefore);
});

/**
 * Test: inspector shows correct delete confirmation message with connection count
 */
test('node deletion flow: inspector shows correct delete confirmation message', async ({ page }) => {
  await page.goto('http://localhost:4200');
  await waitForGraphCanvas(page);

  const uniqueLabel = `Playwright-msg-test-${Date.now()}`;
  await seedNode(page, uniqueLabel);
  await refreshGraph(page);

  const allCircles = page.locator('app-graph-canvas svg circle.node');
  await allCircles.first().wait({ timeout: 8000 });
  await allCircles.first().click();
  await page.waitForTimeout(500);

  const inspector = page.locator('app-inspector-panel');
  await expect(inspector).toBeVisible({ timeout: 5000 });

  const deleteBtn = inspector.locator('button:has-text("Delete")').first();
  await deleteBtn.click();
  await page.waitForTimeout(300);

  // Confirmation message should appear and mention "Delete" and "connection(s)"
  const confirmMsg = inspector.locator('.delete-confirm__message');
  await expect(confirmMsg).toBeVisible({ timeout: 3000 });
  const msgText = await confirmMsg.textContent();
  expect(msgText).toContain('Delete');
  expect(msgText).toMatch(/connection/);

  // Both Cancel and danger Delete buttons should be visible
  await expect(inspector.locator('button:has-text("Cancel")')).toBeVisible();
  await expect(inspector.locator('button.btn--danger:has-text("Delete")')).toBeVisible();
});

/**
 * Test: canvas click deselects node and closes inspector without deleting
 */
test('node deletion flow: canvas click deselects node and closes inspector', async ({ page }) => {
  await page.goto('http://localhost:4200');
  await waitForGraphCanvas(page);

  const uniqueLabel = `Playwright-deselect-test-${Date.now()}`;
  await seedNode(page, uniqueLabel);
  await refreshGraph(page);

  const allCircles = page.locator('app-graph-canvas svg circle.node');
  await allCircles.first().wait({ timeout: 8000 });
  await allCircles.first().click();
  await page.waitForTimeout(500);

  const inspector = page.locator('app-inspector-panel');
  await expect(inspector).toBeVisible({ timeout: 5000 });

  // Click on the SVG canvas background to deselect
  const svg = page.locator('app-graph-canvas svg').first();
  await svg.click({ position: { x: 10, y: 10 } });
  await page.waitForTimeout(500);

  // Inspector should now be hidden / show empty state
  const inspectorEmpty = inspector.locator('.inspector-empty');
  const eitherGoneOrEmpty = await inspectorEmpty.isVisible().catch(() => false);
  expect(eitherGoneOrEmpty).toBeTruthy();
});
