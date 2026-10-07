import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev: `npm run dev` serves the UI on :5173 and proxies API calls to FastAPI on :8000.
// Build: output goes straight into the FastAPI static folder (committed, so Render only needs pip).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/graphql": "http://127.0.0.1:8000",
      "/healthz": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "../app/static",
    emptyOutDir: true,
  },
});
