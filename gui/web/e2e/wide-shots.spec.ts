// SPDX-License-Identifier: Apache-2.0
// Visual-review screenshot sweep — renders the production bundle with the
// hermetic API mock at an ultra-wide (3440) dark-theme viewport and captures
// each navigable section. NOT a CI gate: a developer aid for the wide-screen
// layout work (see e2e/fixtures.ts for the mocked data shapes).
//   Run: npx playwright test --config playwright.shots.config.ts
import { test } from "@playwright/test";
import { mockApi } from "./fixtures";
import { RESPONSES, URL_TABLE } from "../tests/test/fixtures-data";

const OUT = process.env.SHOT_OUT || "/tmp/sndr-wide";

// The card-heavy sections under review plus a few known-good references.
const SECTIONS = [
  "Overview", "Models", "Presets", "Configs", "Planner",
  "Fleet", "Hosts", "Containers", "Virtualization", "Hardware",
  "Launch Plan", "Doctor", "Patches", "Benchmarks", "Routing", "Advanced",
];

async function bootDark(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "sndr.gui.settings",
      JSON.stringify({ theme: "dark", density: "comfortable", accent: "teal" }),
    );
  });
}

test("wide dark-theme section sweep (3440)", async ({ page }) => {
  test.setTimeout(180_000);
  await mockApi(page);
  await bootDark(page);
  // Force the true ultra-wide viewport: the project's `devices["Desktop Chrome"]`
  // pins 1280x720, which would otherwise override the config viewport.
  await page.setViewportSize({ width: 3440, height: 1440 });
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Overview")').first().waitFor();
  await page.waitForTimeout(500);

  for (const label of SECTIONS) {
    const button = page.locator(".side-nav button", { hasText: label }).first();
    if ((await button.count()) === 0) {
      console.log(`skip (no nav entry): ${label}`);
      continue;
    }
    await button.click();
    await page.waitForTimeout(700);
    const slug = label.toLowerCase().replace(/\s+/g, "-");
    await page.screenshot({ path: `${OUT}/${slug}.png`, fullPage: false });
    console.log(`shot: ${slug}`);
  }
});

// Overview with realistic catalog data (mirrors the live daemon numbers) so the
// enriched layout AND any data duplication across cards can be judged for real,
// not against the empty hermetic fixtures.
test("overview with rich data", async ({ page }) => {
  const base = RESPONSES as Record<string, any>;
  const overview = {
    capabilities: {
      ...base.overview.capabilities,
      platform: {
        ...base.overview.capabilities.platform,
        public_brand: "Genesis", package_name: "SNDR Core",
        sndr_core_version: "12.0.0.dev0", os_name: "Darwin", machine: "arm64",
        python_version: "3.11.9", engine_installed: true,
      },
    },
    catalog: {
      ...base.overview.catalog,
      models_count: 10, hardware_count: 4, profiles_count: 12,
      presets_count: 23, preset_cards_count: 23, unannotated_presets_count: 0,
      workload_counts: { free_chat: 5, code_gen: 4, tool_calls: 6, structured_json: 3, summarization: 5 },
      family_counts: { qwen3: 6, llama: 2, deepseek: 2 },
    },
  };
  const presets = {
    ...base.presets, total: 23, matched: 23,
    presets: Array.from({ length: 23 }, (_, i) => ({
      id: `preset-${i}`, model: "qwen3.6-35b", hardware: "a5000x2", profile: "fp8",
      has_card: true, card: { primary_metric: { value: i < 6 ? 120 : 0 }, title: `Preset ${i}` },
    })),
  };
  const patches = {
    ...base.patches, total: 252, matched: 252,
    patches: Array.from({ length: 252 }, (_, i) => ({ id: `patch-${i}`, default_on: i < 52 })),
  };
  const doctor = {
    ...base.doctor,
    findings: Array.from({ length: 16 }, (_, i) => ({
      id: `f-${i}`, severity: i < 2 ? "blocked" : i < 7 ? "warning" : "info",
      title: `Finding ${i}`, category: "runtime", detail: "",
    })),
  };
  const environment = {
    ...base.environment, sndr_core_version: "12.0.0.dev0", engine_name: "vLLM",
    engine_version: "0.20.2", engine_installed: true, os_name: "Darwin", machine: "arm64", python_version: "3.11.9",
  };
  const hosts = { hosts: [{ id: "h1", host: "127.0.0.1", label: "This host", port: 8765, engine_port: 8000 }] };
  const rich: Record<string, any> = { ...base, overview, presets, patches, doctor, environment, hosts };

  await page.route("**/api/**", (route) => {
    const pathname = new URL(route.request().url()).pathname;
    const hit = URL_TABLE.find(([suffix]) => pathname.includes(suffix));
    const body = hit ? rich[hit[1]] : {};
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body ?? {}) });
  });
  await bootDark(page);
  for (const width of [1920, 3440]) {
    await page.setViewportSize({ width, height: width >= 2880 ? 1440 : 1000 });
    await page.goto("/");
    await page.locator('.side-nav button:has-text("Overview")').first().click();
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${OUT}/rich-overview-${width}.png`, fullPage: false });
    console.log(`shot: rich-overview-${width}`);
  }
});

// High-fidelity element clip of the Overview hero tiles, for inspecting tile
// internals (value alignment, click affordance, text clamp) up close.
test("overview hero tile clip", async ({ page }) => {
  await mockApi(page);
  await bootDark(page);
  for (const width of [1920, 3440]) {
    await page.setViewportSize({ width, height: width >= 2880 ? 1440 : 900 });
    await page.goto("/");
    // The app boots on launch-plan; navigate to Overview so `.ov-hero` exists.
    await page.locator('.side-nav button:has-text("Overview")').first().click();
    await page.waitForTimeout(500);
    const hero = page.locator(".ov-hero").first();
    await hero.screenshot({ path: `${OUT}/hero-${width}.png` });
    console.log(`shot: hero-${width}`);
  }
});

// Verify the Overview sub-tabs (Summary / Environment / Coverage) all route to
// real content and capture each for review.
test("overview sub-tabs", async ({ page }) => {
  await mockApi(page);
  await bootDark(page);
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Overview")').first().click();
  await page.waitForTimeout(400);
  for (const tab of ["Summary", "Environment", "Coverage"]) {
    await page.locator('.section-tabs button', { hasText: tab }).first().click();
    await page.waitForTimeout(300);
    await page.screenshot({ path: `${OUT}/ov-tab-${tab.toLowerCase()}.png`, fullPage: false });
    console.log(`shot: ov-tab-${tab.toLowerCase()}`);
  }
});

// Focused responsive check: capture the sections under active work at three
// real widths so the adaptive (viewport-tier) layout can be reviewed side by side.
test("responsive widths for sections under work", async ({ page }) => {
  test.setTimeout(180_000);
  await mockApi(page);
  await bootDark(page);
  const WIDTHS = [1280, 1920, 3440];
  const FOCUS = ["Overview", "Containers"];
  for (const width of WIDTHS) {
    await page.setViewportSize({ width, height: width >= 2880 ? 1440 : 900 });
    await page.goto("/");
    await page.locator('.side-nav button:has-text("Overview")').first().waitFor();
    await page.waitForTimeout(400);
    for (const label of FOCUS) {
      const button = page.locator(".side-nav button", { hasText: label }).first();
      if ((await button.count()) === 0) continue;
      await button.click();
      await page.waitForTimeout(600);
      const slug = label.toLowerCase().replace(/\s+/g, "-");
      await page.screenshot({ path: `${OUT}/${slug}-${width}.png`, fullPage: false });
      console.log(`shot: ${slug}-${width}`);
    }
  }
});
