import { defineConfig, devices } from "@playwright/test";

// Wide-screen visual-review config (NOT a CI gate). Renders the production
// build with the hermetic API mock at a 3440 dark-theme viewport so the
// ultra-wide layout can be inspected via screenshots. Reuses the same
// `vite preview` webServer wiring as the hermetic config.
export default defineConfig({
  testDir: "./e2e",
  testMatch: /wide-shots\.spec\.ts/,
  timeout: 200_000,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
    viewport: { width: 3440, height: 1440 },
    deviceScaleFactor: 1,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run preview -- --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
