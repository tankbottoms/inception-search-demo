# Demo Client Dockerfile
FROM oven/bun:1-alpine

WORKDIR /app

# Install curl for health checks
RUN apk add --no-cache curl

# Copy package files
COPY package.json ./
COPY bun.lock* ./

# Install dependencies
RUN bun install --frozen-lockfile || bun install

# Copy source
COPY src/ ./src/
COPY files/ ./files/

# Create output directory
RUN mkdir -p output logs

# Default command runs the demo
CMD ["bun", "run", "src/index.ts", "demo", "--pdf-count", "0"]
