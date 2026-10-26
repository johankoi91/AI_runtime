import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { csvFor, fixtureEvents } from "./fixtures.js";

const root = fileURLToPath(new URL("../../web/", import.meta.url));
const types = { ".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8" };
const server = createServer(async (request, response) => {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Headers", "authorization, content-type, accept");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  if (request.method === "OPTIONS") { response.writeHead(204).end(); return; }
  const url = new URL(request.url, "http://127.0.0.1");
  if (url.pathname.endsWith("/api/v2/query")) {
    let body = "";
    for await (const chunk of request) body += chunk;
    if (body.includes('bucket: "fixture-error"')) { response.writeHead(503).end("fixture unavailable"); return; }
    const kind = body.includes('bucket: "fixture-empty"') ? "empty" : body.includes('bucket: "fixture-incomplete"') ? "incomplete" : "complete";
    response.writeHead(200, { "Content-Type": "application/csv" }).end(csvFor(fixtureEvents(kind)));
    return;
  }
  const filename = resolve(root, "." + (url.pathname === "/" ? "/index.html" : decodeURIComponent(url.pathname)));
  if (!filename.startsWith(root)) { response.writeHead(403).end(); return; }
  try {
    const contents = await readFile(filename);
    response.writeHead(200, { "Content-Type": types[extname(filename)] || "application/octet-stream" }).end(contents);
  } catch { response.writeHead(404).end(); }
});
server.listen(5174, "127.0.0.1", () => console.log("Read-only legacy baseline + deterministic Trace fixture: http://127.0.0.1:5174"));
