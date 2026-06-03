import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173
  },
  build: {
    // xterm (Terminal) is lazy-loaded, so it splits into its own on-demand chunk
    // automatically — keeping the initial bundle smaller and the app fast to open.
    chunkSizeWarningLimit: 900
  }
});
