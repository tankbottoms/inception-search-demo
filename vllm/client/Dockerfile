# vLLM Hydra Client
# Full-featured demo client with OCR, embeddings, and citation generation
#
# Build: docker build -t vllm-hydra-client .
# Run:   docker run --rm -it --network host -v ./files:/app/files -v ./output:/app/output vllm-hydra-client demo:full

FROM ubuntu:latest AS base

# Install system dependencies and Bun in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    unzip \
    graphicsmagick \
    ghostscript \
    poppler-utils \
    && curl -fsSL https://bun.sh/install | bash \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.bun/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy package files
COPY package.json bun.lock* ./

# Install dependencies
RUN bun install --frozen-lockfile

# Copy source code
COPY src/ ./src/

# Create directories for files and output
RUN mkdir -p /app/files /app/output

# Set environment variables
ENV EMBEDDINGS_URL=http://localhost:8001
ENV OCR_URL=http://localhost:8003
ENV INFERENCE_URL=http://localhost:8004
ENV FILES_DIR=/app/files
ENV OUTPUT_DIR=/app/output

# Default command
ENTRYPOINT ["bun", "run"]
CMD ["demo:full"]
