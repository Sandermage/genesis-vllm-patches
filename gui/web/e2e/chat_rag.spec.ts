import { test, expect } from "@playwright/test";

// The chat's Project-RAG path retrieves from /api/v1/chat/retrieve (read-only
// project knowledge) BEFORE it streams from the engine, so grounding works and
// source citations render even when no engine is running in CI/sandbox.
test("chat: Project RAG grounds an answer in real project sources", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

  await page.goto("/");
  await page.click('.side-nav button:has-text("Chat")');
  await expect(page.locator(".chat2")).toBeVisible();

  // Turn on Project RAG.
  const ragToggle = page.locator(".chat-rag-toggle");
  await expect(ragToggle).toBeVisible();
  await ragToggle.click();
  await expect(ragToggle).toHaveClass(/on/);

  // Empty-state switches to project-flavored suggestions.
  const suggestion = page.locator('.chat-suggest button:has-text("PN95")');
  await expect(suggestion).toBeVisible();
  await suggestion.click();

  // Retrieval grounds the answer — source citations appear (engine stream may
  // fail, but the RAG sources are attached before streaming).
  const sources = page.locator(".chat-sources").first();
  await expect(sources).toBeVisible({ timeout: 10000 });
  await expect(sources.locator(".chat-sources-label")).toContainText(/project source/i);

  // A patch source chip should be present and reference PN95.
  const chips = sources.locator(".chat-src");
  await expect(chips.first()).toBeVisible();
  await expect(sources).toContainText(/PN95|patch:|preset:/i);

  // Clicking a chip expands its snippet detail.
  await chips.first().click();
  await expect(sources.locator(".chat-src-detail")).toBeVisible();

  // No horizontal overflow, no console errors (the engine-down error is shown
  // in-UI via .chat-error, not the console).
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
  );
  expect(overflow).toBe(false);
  expect(errors, errors.join("\n")).toHaveLength(0);
});

// Connecting an external notes folder (Obsidian vault) and grounding in it.
test("chat: connect a notes vault and ground an answer in it", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

  await page.goto("/");
  await page.click('.side-nav button:has-text("Chat")');
  await expect(page.locator(".chat2")).toBeVisible();

  // Open Params → Knowledge sources, connect the demo vault created by the test runner.
  await page.click('.ghost-button:has-text("Params")');
  const knowledge = page.locator(".chat-knowledge");
  await expect(knowledge).toBeVisible();
  await knowledge.locator(".chat-vault-add input").fill("/tmp/obsidian_demo");
  await knowledge.locator('.ghost-button:has-text("Connect")').click();

  // The folder is validated server-side and shown as a connected source.
  const vaultRow = page.locator(".chat-vault").filter({ hasText: "obsidian_demo" });
  await expect(vaultRow).toBeVisible({ timeout: 10000 });

  // Turn RAG on and ask something only the vault knows.
  await page.locator(".chat-rag-toggle").click();
  await page.locator(".chat-composer textarea").fill("how do I deploy, and what about the warmup gate?");
  await page.keyboard.press("Enter");

  // A note-kind source citation from the vault should appear.
  const noteChip = page.locator(".chat-src.src-note").first();
  await expect(noteChip).toBeVisible({ timeout: 10000 });
  await expect(noteChip).toContainText(/note:/i);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
  );
  expect(overflow).toBe(false);
  expect(errors, errors.join("\n")).toHaveLength(0);
});
