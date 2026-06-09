import { defineConfig } from "vite";
import path from "node:path";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
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
      // xterm (Terminal) is lazy-loaded into its own chunk.
      chunkSizeWarningLimit: 900,
    },
  };
});
