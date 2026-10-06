import express from "express";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const app = express();
const publicDir = join(dirname(fileURLToPath(import.meta.url)), "public");

app.disable("x-powered-by");
app.get("/healthz", (_req, res) => res.type("text").send("ok"));

app.use(
  express.static(publicDir, {
    setHeaders(res, path) {
      if (path.endsWith(".wasm")) res.type("application/wasm");
      // vendored runtime files are version-pinned; HTML should revalidate
      if (path.includes("/vendor/")) res.set("Cache-Control", "public, max-age=31536000, immutable");
      else if (path.endsWith(".html")) res.set("Cache-Control", "no-cache");
      // mic access is needed by this origin only
      res.set("Permissions-Policy", "microphone=(self)");
    },
  })
);

const port = process.env.PORT || 3000;
app.listen(port, "0.0.0.0", () => console.log(`listening on ${port}`));
