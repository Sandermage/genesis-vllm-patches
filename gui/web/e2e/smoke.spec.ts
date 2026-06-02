import { test, expect } from "@playwright/test";

const SECTIONS = [
  "Overview", "Setup", "Fleet", "Hosts", "Models", "Configs", "Presets",
  "Planner", "Copilot", "Launch Plan", "Services", "Doctor", "Patches", "Benchmarks",
  "Evidence", "Clients", "Chat", "Reports", "Operations", "Advanced"
];

test("control center loads, navigates every section, no console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

  await page.goto("/");
  await expect(page.locator(".sidebar")).toBeVisible();
  await expect(page.locator(".side-nav")).toBeVisible();
  // Data loaded from the daemon: footer shows a connected pill.
  await expect(page.locator(".status-footer")).toBeVisible();

  for (const section of SECTIONS) {
    await page.click(`.side-nav button:has-text("${section}")`);
    await expect(page.locator(".main-shell")).toBeVisible();
    await page.waitForTimeout(200);
  }

  // Catch real app/JS errors (TypeError, React, pageerror) — but ignore
  // network/resource noise (a probed host being down, or transient HTTP from
  // other tests mutating shared backend state in parallel). Same filter the
  // host-wiring spec uses; the engine/daemon probes legitimately fail-fast.
  const appErrors = errors.filter((e) => !/Failed to load resource|ERR_CONNECTION_REFUSED|ERR_NETWORK|net::/.test(e));
  expect(appErrors, appErrors.join("\n")).toHaveLength(0);
});

test("no horizontal overflow at tablet width", async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 900 });
  await page.goto("/");
  await page.waitForTimeout(800);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
  );
  expect(overflow).toBe(false);
});
