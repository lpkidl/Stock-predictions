import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Frontend always calls relative /api URLs; dev proxies to uvicorn.
      "/api": "http://127.0.0.1:8000",
    },
  },
});
