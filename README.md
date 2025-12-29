# Inception Search Demo

> Multi-platform document processing with OCR, embeddings, and semantic search

## Overview

A comprehensive document processing pipeline featuring:

- **OCR**: Extract text from PDFs using HunyuanOCR (GPU) or Tesseract (CPU)
- **Embeddings**: Generate semantic embeddings using ModernBERT
- **Inference**: Chain-of-thought reasoning with GPT-OSS 20B
- **Search**: Cosine similarity semantic search

Two deployment options are available:

| Option | Use Case | Stack | Documentation |
|--------|----------|-------|---------------|
| **vLLM Hydra** | Production GPU inference | vLLM + Docker | [vllm/README.md](vllm/README.md) |
| **ONNX Backend** | Development / CPU | TypeScript + ONNX | This file |

**Target Platforms**:

| Platform | Provider | Notes |
|----------|----------|-------|
| NVIDIA DGX Spark | vLLM + CUDA | Production recommended (ARM64 + GB200) |
| Apple Silicon (M1-M5) | CPUExecutionProvider | ARM64 CPU inference |
| Generic ARM64/x64 | CPUExecutionProvider | Fallback CPU |

## Models

| Model | Type | Status | Provider |
|-------|------|--------|----------|
| `freelawproject/modernbert-embed-base_finetune_512` | Embedding | Active | vLLM |
| `tencent/HunyuanOCR` | OCR | Active | vLLM |
| `openai/gpt-oss-20b` | Inference | Active | vLLM |
| Tesseract | OCR | Active | CPU fallback |

## Directory Structure

```
/
├── src/                        # TypeScript/Bun inference backend
│   ├── index.ts                # Hono server entry
│   ├── config.ts               # Settings (ENV + JSON)
│   ├── cli.ts                  # CLI: --check, --benchmark
│   ├── routes/
│   │   ├── embed.ts            # /api/v1/embed/*
│   │   ├── ocr.ts              # /api/v1/ocr/*
│   │   └── health.ts           # /health, /metrics
│   ├── services/
│   │   ├── model-registry.ts   # Model resolution logic
│   │   ├── model-loader.ts     # ONNX session management
│   │   ├── embedding.ts        # Embedding generation
│   │   └── ocr/
│   │       ├── index.ts        # OCR router
│   │       ├── mistral.ts      # Mistral OCR API
│   │       ├── tesseract.ts    # Tesseract CPU fallback
│   │       └── pdf-utils.ts    # PDF processing utilities
│   └── instrumentation/
│       ├── metrics.ts          # Prometheus metrics
│       └── logger.ts           # Structured logging
│
├── vllm/                       # vLLM Hydra (production GPU stack)
│   ├── README.md               # Comprehensive vLLM documentation
│   ├── Makefile                # All operations via make commands
│   ├── docker-compose.yml      # Main stack (spark-1)
│   ├── docker-compose.spark2.yml  # Multi-node (spark-2)
│   ├── traefik/                # Load balancer configuration
│   │   ├── traefik.yml
│   │   └── dynamic.yml
│   ├── client/                 # TypeScript demo client
│   │   ├── src/
│   │   │   ├── index.ts        # CLI commands
│   │   │   ├── cot-demo.ts     # Chain-of-thought demo
│   │   │   ├── verify-demo.ts  # Verification suite
│   │   │   ├── stress-test.ts  # Load testing
│   │   │   └── ocr-pipeline.ts # OCR pipeline
│   │   └── package.json
│   ├── scripts/
│   │   ├── hydra-start.sh      # Smart startup
│   │   ├── sync-spark2.sh      # Multi-node sync
│   │   └── verify-services.sh  # Health monitoring
│   ├── CHANGELOG.md
│   └── TODO.md
│
├── llm-model-server/           # Python embedding server
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   └── src/
│       └── main.py             # FastAPI server
│
├── demo/                       # Demo client
│   ├── files/                  # Sample PDFs
│   ├── output/                 # Generated outputs
│   └── logs/                   # Benchmark sessions
│
├── models/                     # Model cache (mounted volume)
│
├── scripts/
│   ├── startup.sh              # Main entry (auto-detect platform)
│   ├── detect-platform.sh      # Platform detection
│   ├── benchmark-cpu-gpu.sh    # CPU vs GPU comparison
│   └── verify-pipeline.sh      # Pipeline verification
│
├── Dockerfile                  # CPU build
├── docker-compose.yml          # ONNX backend services
├── Makefile                    # Simplified commands
├── package.json
├── README.md                   # This file
├── CHANGELOG.md
└── TODO.md
```

## Quick Start

### Option 1: vLLM Hydra (Production GPU - Recommended)

For NVIDIA DGX Spark or other CUDA GPUs:

```bash
cd vllm

# Start all services (embeddings + OCR + inference + load balancer)
make up-all

# Check service health
make health

# Run verification tests (14 tests)
make test

# Run chain-of-thought demo
make demo-cot-quick

# Stop all services
make down
```

See [vllm/README.md](vllm/README.md) for comprehensive documentation.

### Option 2: ONNX Backend (Development / CPU)

For local development or CPU-only environments:

```bash
# Prerequisites: Bun >= 1.0, Docker

# Install dependencies
bun install

# Start server
bun run dev

# Run demo
cd demo && bun run demo
```

### Platform Detection

The system auto-detects your platform:

```bash
# Show platform info
./scripts/detect-platform.sh all

# Or via make
make status
```

## Model Resolution Flow

```
1. Check local cache (/models/*.onnx)
   └── Found? → Load → Ready

2. Check HuggingFace for ONNX files
   └── Found? → Download → Cache → Ready

3. Fallback: Python conversion service
   └── Download .safetensors → Convert → Cache → Ready
```

## Output Files

The demo client processes PDFs and generates:

| File | Description |
|------|-------------|
| `output/{hash}.ocr.md` | OCR extracted text in markdown |
| `output/{hash}.bert.json` | Embedding vectors with metadata |
| `logs/{timestamp}.json` | Benchmark session data |

## API Endpoints

### vLLM Hydra (OpenAI-compatible)

All services expose OpenAI-compatible APIs:

```bash
# Embeddings (port 8001, or 8000 for load balanced)
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "freelawproject/modernbert-embed-base_finetune_512",
    "input": "Your text to embed"
  }'

# OCR via HunyuanOCR (port 8003, or 8010 for load balanced)
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tencent/HunyuanOCR",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract all text from this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }],
    "max_tokens": 4096
  }'

# Inference via GPT-OSS (port 8004, or 8020 for load balanced)
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is 15 + 28?"}],
    "max_tokens": 512
  }'

# Health check (all services)
curl http://localhost:8001/health
curl http://localhost:8003/health
curl http://localhost:8004/health
```

### ONNX Backend (Development)

```bash
# Embedding
curl -X POST http://localhost:8005/api/v1/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "search query here"}'

# Health check
curl http://localhost:8005/health
```

## Instrumentation

All operations are timed and logged:

```typescript
interface TimingMetrics {
  operation: string;
  model_id: string;
  provider: string;        // CPU, CUDA
  input_size: number;      // tokens or bytes
  output_size: number;     // vectors or characters
  latency_ms: number;
  tokens_per_second?: number;
  memory_mb?: number;
}
```

### Benchmark Report

```json
{
  "system": {
    "platform": "linux",
    "arch": "arm64",
    "cpu": "Nvidia Grace",
    "gpu": "Nvidia Blackwell",
    "memory_gb": 128,
    "provider": "CUDAExecutionProvider"
  },
  "models": {
    "modernbert-embed": {
      "load_time_ms": 1234,
      "inference": {
        "avg_latency_ms": 12.5,
        "p50_latency_ms": 11.2,
        "p95_latency_ms": 18.7,
        "tokens_per_second": 4521
      }
    }
  },
  "comparison": {
    "cpu_baseline_ms": 89.2,
    "gpu_accelerated_ms": 12.5,
    "speedup": "7.1x"
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8005` | Server port |
| `EXECUTION_PROVIDER` | `auto` | `auto`, `cpu`, `cuda` |
| `MODEL_REGISTRY` | `/models/registry.json` | Model config path |
| `CONVERTER_URL` | `http://converter:8010` | Python converter service |
| `MISTRAL_OCR_API_KEY` | - | Mistral API key for OCR |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error` |
| `ENABLE_METRICS` | `true` | Prometheus metrics |

### Model Registry (registry.json)

```json
{
  "version": "1.0",
  "cache_dir": "/models",
  "models": [
    {
      "id": "modernbert-embed",
      "name": "freelawproject/modernbert-embed-base_finetune_512",
      "type": "embedding",
      "enabled": true,
      "config": {
        "max_tokens": 512,
        "embedding_dim": 768,
        "pooling": "mean",
        "normalize": true,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: "
      }
    },
    {
      "id": "deepseek-ocr",
      "name": "deepseek-ai/DeepSeek-OCR",
      "type": "ocr",
      "enabled": true
    },
    {
      "id": "hunyuan-ocr",
      "name": "tencent/HunyuanOCR",
      "type": "ocr",
      "enabled": true
    }
  ]
}
```

## Docker Compose Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| `cpu` | backend-cpu | Apple Silicon, generic ARM64 |
| `gpu` | backend-gpu | DGX Spark, CUDA GPUs |
| `demo` | backend + demo client | Full demo |
| `convert` | converter | Model conversion only |
| `llm-cpu` | llm-model-server-cpu | Python LLM server (CPU) |
| `llm-gpu` | llm-model-server-gpu | Python LLM server (GPU) |

## vLLM Hydra Cluster

For production GPU deployments, the vLLM Hydra stack provides:

- **Multi-model inference**: Embeddings, OCR, and GPT-OSS on shared GPU
- **Load balancing**: Traefik distributes requests across replicas
- **Multi-node**: Deploy across spark-1 and spark-2 for high availability

**Quick start**:

```bash
cd vllm

# Start all services
make up-all

# Check cluster health
make health-all

# Run 14-test verification suite
make test

# Run chain-of-thought demo
make demo-cot
```

**Service Ports**:

| Service | spark-1 | spark-2 | Load Balanced |
|---------|---------|---------|---------------|
| Embeddings | 8001, 8002 | 8001, 8002 | 8000 |
| OCR | 8003 | 8003 | 8010 |
| Inference | 8004 | -- | 8020 |

See [vllm/README.md](vllm/README.md) for comprehensive documentation.

## Development

### Project Setup

```bash
# Clone and checkout branch
git clone <repo>
git checkout feature/onnx-typescript-backend

# Install dependencies
bun install
cd demo && bun install
cd ../converter && pip install -e .

# Run tests
bun test

# Type check
bun run typecheck
```

### Testing

```bash
# Unit tests
bun test

# Integration tests (requires running server)
bun test:integration

# Benchmark tests
bun test:benchmark
```

## Resources

- [ONNX Runtime](https://onnxruntime.ai/)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [Hono](https://hono.dev/)
- [Free Law Project Inception](https://github.com/freelawproject/inception)
- [ModernBERT Model](https://huggingface.co/freelawproject/modernbert-embed-base_finetune_512)

## License

MIT
