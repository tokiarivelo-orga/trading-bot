# syntax=docker/dockerfile:1
#
# Production image for the Next.js frontend, built from `.next/standalone`
# (next.config.ts: output: "standalone"). Multi-stage: `deps` installs with
# a frozen pnpm lockfile, `builder` runs the production build, `runner` is a
# minimal Alpine image that only holds the standalone server + static assets.
# Runs as a non-root user.
#
# Backend REST is proxied under /api at runtime via BACKEND_URL (see
# next.config.ts) — set it in the container environment (docker-compose.prod.yml),
# not baked in at build time.
#
# Build:  docker build -f frontend/Dockerfile -t tradingbot/frontend:local frontend
# Run:    see docker-compose.prod.yml (pulls a published image instead)

FROM node:24-alpine AS base

# ─── deps: install with a frozen lockfile ──────────────────────────────────
FROM base AS deps
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# ─── builder: production build ─────────────────────────────────────────────
FROM base AS builder
WORKDIR /app
RUN corepack enable
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm build

# ─── runner: minimal standalone runtime ────────────────────────────────────
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
# .next/standalone already contains a pruned node_modules + server.js;
# .next/static is not included in standalone and must be copied separately
# (Next.js standalone-output convention).
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider "http://127.0.0.1:${PORT}/" || exit 1

CMD ["node", "server.js"]
