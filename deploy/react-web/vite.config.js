import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const root = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, root, "HUSH_");
  const proxy = { "/influx": { target: environment.HUSH_INFLUX_PROXY_TARGET || "http://81.71.150.26", changeOrigin: true } };
  return {
  base: "./",
  plugins: [react()],
  server: { host: "127.0.0.1", proxy },
  preview: { host: "127.0.0.1", proxy },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        trace: resolve(root, "index.html"),
        tutorial: resolve(root, "tutorial.html")
      }
    }
  }
  };
});
