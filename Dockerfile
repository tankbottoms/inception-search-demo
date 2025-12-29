# Inception ONNX - CPU Build
# TypeScript/Bun inference backend with ONNX Runtime
# Uses Debian for glibc compatibility with onnxruntime-node

FROM oven/bun:1-debian AS base

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies stage
FROM base AS deps
COPY package.json bun.lockb* ./
RUN bun install --frozen-lockfile || bun install

# Build stage
FROM base AS build
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN bun run build || true

# Production stage
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8005
ENV EXECUTION_PROVIDER=cpu
ENV MODEL_REGISTRY=/models/registry.json
ENV LOG_LEVEL=info

# Create non-root user
RUN groupadd --system --gid 1001 nodejs && \
    useradd --system --uid 1001 --gid nodejs inference

# Copy built application with ownership
COPY --from=deps --chown=inference:nodejs /app/node_modules ./node_modules
COPY --from=build --chown=inference:nodejs /app/src ./src
COPY --from=build --chown=inference:nodejs /app/package.json ./

# Create models directory
RUN mkdir -p /models && chown -R inference:nodejs /models /app

USER inference

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8005/health || exit 1

CMD ["bun", "run", "src/index.ts"]
