// SPDX-License-Identifier: Apache-2.0
// Hermetic E2E + accessibility gate for CI. Unlike smoke/host_wiring/server_switch
// (which need the live daemon and run on the dev box), this spec mocks every
// /api/** call (see fixtures.ts) and runs against `vite preview` of the
// production bundle. It proves two things no unit test can:
//   1. the built bundle boots without a runtime/JS crash, and
//   2. the real, CSS-composed DOM is free of structural accessibility
//      violations across every navigable section.
//
// Scan policy: fail on critical/serious axe violations — including color-contrast.
// Every navigable section is scanned in the default (light) theme here; the
// "every theme is WCAG AA" test below boots each of the four themes natively and
// scans them too. All four (light/dark/carbon/lime) are contrast-clean.
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mockApi } from "./fixtures";

const THEMES = ["light", "dark", "carbon", "lime"] as const;

// Freeze CSS transitions/animations so contrast is measured at final colours,
// not mid-fade (entrance animations briefly render text at reduced opacity,
// which would make the contrast scan flaky). Call before goto().
async function freezeUi(page: import("@playwright/test").Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const css = "*,*::before,*::after{transition:none!important;animation:none!important}";
    const apply = () => {
      const style = document.createElement("style");
      style.textContent = css;
      document.head.appendChild(style);
    };
    if (document.head) apply();
    else document.addEventListener("DOMContentLoaded", apply);
  });
}

async function scan(page: import("@playwright/test").Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter(
    (v) => v.impact === "critical" || v.impact === "serious",
  );
  const report = serious
    .map((v) => `[${v.impact}] ${v.id}: ${v.help} (${v.nodes.length} node(s))`)
    .join("\n");
  expect(serious, report).toHaveLength(0);
}

test("production bundle boots and the shell renders without API", async ({ page }) => {
  await mockApi(page);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

  await page.goto("/");
  // Stable, always-sized landmarks: the sidebar chrome and its nav buttons
  // render regardless of data-load state (the main-shell can be momentarily
  // zero-height while async content fills in, so we don't gate on it here).
  await expect(page.locator(".sidebar")).toBeVisible();
  await expect(page.locator('.side-nav button:has-text("Overview")')).toBeVisible();

  expect(errors, errors.join("\n")).toHaveLength(0);
});

test("Overview is accessible (axe, real DOM)", async ({ page }) => {
  await mockApi(page);
  await freezeUi(page);
  await page.goto("/");
  await expect(page.locator('.side-nav button:has-text("Overview")')).toBeVisible();
  await scan(page);
});

test("every nav entry routes to real content and stays accessible", async ({ page }) => {
  await mockApi(page);
  await freezeUi(page);
  await page.goto("/");
  await expect(page.locator(".side-nav")).toBeVisible();

  // Drive the ACTUAL nav (no hardcoded list to drift): every button the sidebar
  // renders must route to a non-empty workspace — a button that goes nowhere
  // (unrouted section) would leave the content area blank. Also scans a11y.
  const labels = await page.locator(".side-nav button").allInnerTexts();
  expect(labels.length).toBeGreaterThan(10);

  for (const label of labels) {
    const button = page.locator(".side-nav button", { hasText: label.trim() }).first();
    await button.click();
    // Routing + robustness invariant: the clicked entry becomes active and its
    // workspace renders its real heading (NOT the error boundary). Every section
    // must survive the empty hermetic data and show its empty state — a panel
    // that crashes on empty/partial data would fall back to .error-boundary and
    // fail here. (Fixtures in e2e/fixtures.ts supply type-complete empty shapes.)
    await expect(button).toHaveClass(/active/);
    const workspace = page.locator(".main-shell .section-workspace").first();
    await expect(workspace).toBeVisible();
    await expect(workspace.locator(".section-heading").first()).toBeVisible();
    await expect(workspace.locator(".error-boundary")).toHaveCount(0);
    await scan(page);
  }
});

test("every theme is WCAG AA (color-contrast across all sections)", async ({ page }) => {
  test.setTimeout(180_000); // 4 themes x 20 sections of axe scans
  await mockApi(page);
  await freezeUi(page);
  // Boot each theme natively via persisted settings (no dataset injection, so no
  // mid-transition colour artefacts), then contrast-scan every section.
  const SECTIONS = ["Overview", "Setup", "Fleet", "Hosts", "Models", "Configs", "Presets", "Planner", "Copilot", "Launch Plan", "Services", "Doctor", "Patches", "Benchmarks", "Evidence", "Clients", "Chat", "Reports", "Operations", "Advanced"];
  for (const theme of THEMES) {
    await page.addInitScript((t) => {
      window.localStorage.setItem("sndr.gui.settings", JSON.stringify({ theme: t, density: "comfortable", accent: "teal" }));
    }, theme);
    await page.goto("/");
    await expect(page.locator('.side-nav button:has-text("Overview")')).toBeVisible();
    await page.waitForTimeout(300);
    for (const label of SECTIONS) {
      const button = page.locator(".side-nav button", { hasText: label }).first();
      if ((await button.count()) === 0) continue;
      await button.click();
      await page.waitForTimeout(180);
      const results = await new AxeBuilder({ page }).withRules(["color-contrast"]).analyze();
      const nodes = results.violations.flatMap((v) =>
        v.nodes.map((n) => `[${theme}/${label}] ${n.html.slice(0, 80)} — ${(n.any?.[0]?.data as { contrastRatio?: number } | undefined)?.contrastRatio}`),
      );
      expect(nodes, nodes.join("\n")).toHaveLength(0);
    }
  }
});

test("Setup subtabs render without crashing", async ({ page }) => {
  // The 'every nav entry' test only checks each section's DEFAULT tab; the Setup
  // sub-tabs (Guided / Install onto host / Deploy a model) host their own panels
  // (SetupWizard / InstallWizard / DeploymentConsole) that must also survive the
  // empty hermetic data instead of falling back to the error boundary.
  await mockApi(page);
  await freezeUi(page);
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Setup")').first().click();
  for (const tab of ["Guided setup", "Install onto host", "Deploy a model"]) {
    await page.locator(".section-tabs button", { hasText: tab }).first().click();
    const workspace = page.locator(".main-shell .section-workspace").first();
    await expect(workspace.locator(".error-boundary")).toHaveCount(0);
    await scan(page);
  }
});

test("no horizontal overflow at tablet width", async ({ page }) => {
  await mockApi(page);
  await page.setViewportSize({ width: 768, height: 900 });
  await page.goto("/");
  await page.waitForTimeout(400);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
  );
  expect(overflow).toBe(false);
});
