/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import path from "node:path";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // ``--mode=carbon`` builds the new Carbon-based Control Center from
  // index.carbon.html → main.carbon.tsx → CarbonApp. Anything else
  // continues to build the legacy App.tsx for v11-line operators.
  const carbonMode = mode === "carbon";
  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src"),
      },
    },
    server: {
      host: "127.0.0.1",
      port: 5173,
    },
    build: {
      // xterm (Terminal) is lazy-loaded; Carbon-mode also splits heavy chunks.
      chunkSizeWarningLimit: 900,
      rollupOptions: carbonMode
        ? { input: path.resolve(__dirname, "index.carbon.html") }
        : undefined,
    },
    test: {
      environment: "jsdom",
      globals: true,
      include: ["src/**/*.{test,spec}.{ts,tsx}"],
      exclude: ["e2e/**", "node_modules/**"],
      coverage: {
        provider: "v8",
        reporter: ["text-summary", "html", "lcov"],
        reportsDirectory: "coverage",
        // Measure the modular library; exclude entry points, type-only and
        // generated files, and the lazy-loaded heavy panels covered by E2E.
        include: ["src/**/*.{ts,tsx}"],
        exclude: [
          "src/**/*.{test,spec}.{ts,tsx}",
          "src/main.tsx",
          "src/main.carbon.tsx",
          "src/api.ts",
          "src/**/*.d.ts",
        ],
        // Regression floor locked just under the measured level. Raised after
        // the App shell test (App.shell.test.tsx) began exercising the boot
        // orchestration + SectionWorkspace routing — lines 30->36, branches
        // 24->29. Bump again as more shell paths land.
        thresholds: { lines: 35, functions: 28, statements: 32, branches: 28 },
      },
    },
  };
});
