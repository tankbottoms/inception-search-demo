# Inception ONNX - TypeScript/Bun Inference Backend

> Multi-platform ONNX inference service with ARM64 CPU and CUDA GPU acceleration

## Overview

This branch refactors the inception-demo from Python/PyTorch to a unified TypeScript/Bun stack using ONNX Runtime. The goal is a single codebase that auto-detects hardware and leverages GPU acceleration when available.

**Target Platforms**:

| Platform | Provider | Notes |
|----------|----------|-------|
| Apple Silicon (M1-M5) | CPUExecutionProvider | ARM64 CPU inference |
| Nvidia DGX Spark | CUDAExecutionProvider | ARM64 + CUDA acceleration |
| Generic ARM64/x64 | CPUExecutionProvider | Fallback CPU |

## Approved Models

| Model | Type | Status |
|-------|------|--------|
| `freelawproject/modernbert-embed-base_finetune_512` | Embedding | Primary |
| `deepseek-ai/DeepSeek-OCR` | OCR | Enabled |
| `tencent/HunyuanOCR` | OCR | Enabled |
| `gpt-oss:20b` | General Inference | Planned |

## Directory Structure

```
/
├── backup/                     # Original files (delete after testing)
│   ├── client/
│   ├── inception/
│   ├── doctor/
│   └── tests/
│
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
│   │   ├── huggingface.ts      # HF API client
│   │   ├── tokenizer.ts        # Transformers.js tokenization
│   │   ├── embedding.ts        # Embedding generation
│   │   ├── pooling.ts          # Mean pooling + L2 norm
│   │   ├── chunking.ts         # Text chunking
│   │   └── ocr/
│   │       ├── mistral.ts      # Mistral OCR API
│   │       ├── deepseek.ts     # DeepSeek-OCR local
│   │       └── hunyuan.ts      # HunyuanOCR local
│   ├── providers/
│   │   └── provider-factory.ts # CPU/CUDA detection
│   └── instrumentation/
│       ├── metrics.ts          # Prometheus metrics
│       ├── timing.ts           # Operation timing
│       └── logger.ts           # Structured logging
│
├── converter/                  # Python ONNX conversion (fallback)
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── convert.py              # CLI conversion
│   └── server.py               # HTTP conversion API
│
├── demo/                       # Demo client
│   ├── src/
│   │   ├── index.ts            # CLI commands
│   │   ├── api.ts              # Backend API client
│   │   ├── benchmark.ts        # Benchmark analyzer
│   │   └── types.ts
│   ├── files/                  # Sample PDFs (existing)
│   ├── output/                 # Generated outputs
│   │   ├── *.ocr.md            # OCR markdown
│   │   └── *.bert.json         # Embeddings
│   ├── logs/                   # Benchmark sessions
│   └── package.json
│
├── models/                     # ONNX model cache (mounted volume)
│   └── registry.json           # Model definitions
│
├── vllm/                       # vLLM distributed inference (alternative)
│   ├── README.md
│   ├── docker-compose.spark-1.yml
│   ├── docker-compose.spark-2.yml
│   └── scripts/
│       ├── start-cluster.sh
│       ├── stop-cluster.sh
│       └── monitor.sh
│
├── test/                       # Tests
│   ├── embedding.test.ts
│   ├── ocr.test.ts
│   └── benchmark.test.ts
│
├── scripts/
│   ├── startup.sh              # Main entry
│   ├── check-models.sh         # Model validation
│   └── benchmark.sh            # CPU vs GPU comparison
│
├── Dockerfile                  # CPU build
├── Dockerfile.cuda             # GPU build
├── docker-compose.yml
├── package.json
├── tsconfig.json
├── README.md                   # This file
├── CHANGELOG.md
└── TODO.md
```

## Quick Start

### Prerequisites

- Bun >= 1.0
- Docker & Docker Compose
- (Optional) NVIDIA GPU with CUDA 12.x for GPU acceleration

### Run with Docker

```bash
# Auto-detect platform (CPU or GPU)
./scripts/startup.sh

# Explicit CPU mode
./scripts/startup.sh --profile cpu

# GPU mode (DGX Spark)
./scripts/startup.sh --profile gpu

# Check/download models only
./scripts/startup.sh --check

# Run benchmarks
./scripts/startup.sh --benchmark
```

### Run Locally (Development)

```bash
# Install dependencies
bun install

# Check models
bun run cli --check

# Start server
bun run dev

# Run demo
cd demo && bun run demo
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

### Embedding

```bash
# Query embedding (fast, single vector)
curl -X POST http://localhost:8005/api/v1/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "search query here"}'

# Document embedding (chunked, multiple vectors)
curl -X POST http://localhost:8005/api/v1/embed/text \
  -H "Content-Type: text/plain" \
  -d "Full document text here..."

# Batch embedding
curl -X POST http://localhost:8005/api/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"id": 1, "text": "..."}]}'
```

### OCR

```bash
# OCR with Mistral (default)
curl -X POST http://localhost:8005/api/v1/ocr \
  -F "file=@document.pdf" \
  -F "provider=mistral"

# OCR with DeepSeek
curl -X POST http://localhost:8005/api/v1/ocr \
  -F "file=@document.pdf" \
  -F "provider=deepseek"

# OCR with HunyuanOCR
curl -X POST http://localhost:8005/api/v1/ocr \
  -F "file=@document.pdf" \
  -F "provider=hunyuan"
```

### Health & Metrics

```bash
# Health check
curl http://localhost:8005/health

# Prometheus metrics
curl http://localhost:8005/metrics
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
| `legacy` | inception-cpu (Python) | Comparison/fallback |

## vLLM Distributed Inference

For production deployments on dual DGX Spark nodes, see `vllm/README.md`.

**Quick start**:

```bash
# Start Ray cluster on spark-1 and spark-2
cd vllm && ./scripts/start-cluster.sh

# Serve model distributed across both nodes
./scripts/start-cluster.sh --model tencent/HunyuanOCR

# Monitor cluster
./scripts/monitor.sh
```

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

## Implementation Status

### Phase 1: Branch Setup

- [x] Create feature branch
- [x] Move existing files to backup/
- [x] Create directory structure
- [x] Initialize package.json, tsconfig.json

### Phase 2: Core Backend

- [ ] Hono server setup
- [ ] Provider detection (CPU/CUDA)
- [ ] Model registry loader
- [ ] HuggingFace API client

### Phase 3: Python Converter

- [ ] Dockerfile
- [ ] CLI conversion tool
- [ ] HTTP API for on-demand conversion

### Phase 4: Inference Services

- [ ] Tokenization (Transformers.js)
- [ ] ONNX model loading
- [ ] Mean pooling + normalization
- [ ] Text chunking

### Phase 5: OCR Integration

- [ ] Mistral OCR API
- [ ] DeepSeek-OCR local inference
- [ ] HunyuanOCR local inference

### Phase 6: Demo Client

- [ ] PDF processing pipeline
- [ ] OCR -> Embedding workflow
- [ ] Search with similarity
- [ ] Benchmark reporting

### Phase 7: Docker & Scripts

- [ ] Dockerfile (CPU)
- [ ] Dockerfile.cuda (GPU)
- [ ] docker-compose.yml
- [ ] startup.sh scripts

### Phase 8: vLLM Alternative

- [ ] spark-1 docker-compose
- [ ] spark-2 docker-compose
- [ ] Cluster management scripts

### Phase 9: Benchmarking

- [ ] CPU vs GPU comparison
- [ ] vLLM distributed benchmarks
- [ ] Generate comparison reports

## Migration Notes

The `backup/` folder contains the original Python implementation:

- `backup/inception/` - Original FastAPI embedding service
- `backup/client/` - Original TypeScript demo client
- `backup/doctor/` - Django document processing

These will be deleted after the new implementation is validated. The legacy Python backend remains available via `--profile legacy` for comparison.

## Resources

- [ONNX Runtime](https://onnxruntime.ai/)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [Hono](https://hono.dev/)
- [Free Law Project Inception](https://github.com/freelawproject/inception)
- [ModernBERT Model](https://huggingface.co/freelawproject/modernbert-embed-base_finetune_512)

## License

MIT
