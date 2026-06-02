import { defineConfig, devices } from "@playwright/test";

// Smoke E2E for the SNDR Control Center. Assumes the read-only daemon
// (`sndr gui-api`, :8765) and the Vite dev server (:5173/:5174) are running.
// Override the UI origin with PLAYWRIGHT_BASE_URL when Vite falls back ports.
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:5174",
    headless: true,
    viewport: { width: 1440, height: 900 }
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }]
});
