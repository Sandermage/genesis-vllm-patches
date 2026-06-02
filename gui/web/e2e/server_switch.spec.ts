import { test, expect } from "@playwright/test";

// The daemon connection switcher is a reflection of the host registry (single
// source of truth): "This host (local daemon)" plus every host profile. There
// is no separate saved-server list.
test("connection switcher is driven by the host registry", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto("/");
  await expect(page.locator(".server-current")).toBeVisible();
  // The local daemon is always the anchor connection.
  await expect(page.locator(".server-current-label")).toHaveText(/local|127\.0\.0\.1|host/i);

  // Open the switcher: it lists "This host" + host profiles, with a manage link.
  await page.locator(".server-current").click();
  await expect(page.locator(".server-menu")).toBeVisible();
  await expect(page.locator(".server-item").filter({ hasText: /local daemon/i })).toBeVisible();
  await expect(page.locator(".server-add")).toHaveText(/manage hosts/i);

  // "Add / manage hosts" navigates to the Hosts section (single creation flow).
  await page.locator(".server-add").click();
  await expect(page.locator(".section-hosts, .section-workspace.section-hosts")).toBeVisible();

  // Picking the local daemon keeps the GUI connected (apiBase = local).
  await page.locator(".server-current").click();
  await page.locator(".server-item").filter({ hasText: /local daemon/i }).locator(".server-pick").click();
  await page.waitForTimeout(800);
  const apiBase = await page.evaluate(() => localStorage.getItem("sndr.gui.apiBase"));
  expect(apiBase === null || /127\.0\.0\.1:8765|localhost:8765/.test(apiBase)).toBeTruthy();

  expect(errors.filter((e) => !/ERR_CONNECTION_REFUSED|Failed to load resource/.test(e)), errors.join("\n")).toHaveLength(0);
});
