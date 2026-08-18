import type { NextConfig } from "next";

// Backend REST is proxied under /api so the frontend never hardcodes the
// backend URL. WebSockets are NOT proxied by Next rewrites — see
// src/shared/api/ws.ts, which connects to the backend directly.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Standalone output traces the minimal set of files/deps a production
  // server needs into .next/standalone (incl. a self-contained server.js) —
  // required for the lean multi-stage `frontend/Dockerfile` production
  // image; unused by `pnpm dev` / `make dev-frontend`.
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },
  experimental: {
    // Default is 10MB; strategy-spec PDFs (with embedded images/diagrams)
    // routinely exceed that. Applies to requests proxied through the /api
    // rewrite above, e.g. POST /api/ai/pdf-strategy/upload.
    proxyClientMaxBodySize: "50mb",
    // Next's dev proxy kills any /api rewrite after 30s by default. PDF
    // extraction and code generation both run a real LLM call and routinely
    // take longer than that, so the proxy was resetting the socket
    // (ECONNRESET) while the backend kept working in the background — the
    // draft would show up on refresh even though the request itself errored
    // client-side. code_generation via the claude_code provider can run up
    // to its own 480s adapter timeout (backend/src/ai/adapters/claude_code.py,
    // TB_CLAUDE_CODE_TIMEOUT_S) — this must stay above that or the proxy cuts
    // the connection before the backend's own timeout/response has a chance.
    proxyTimeout: 540_000,
  },
};

export default nextConfig;
