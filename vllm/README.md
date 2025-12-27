# vLLM Hydra Cluster

Multi-model GPU inference stack for document processing with vLLM. Provides embeddings, OCR, and general inference services in a single Docker Compose stack with optional load balancing.

## Quick Start

### Using Make (Recommended)

```bash
cd vllm

# Start all services (embeddings + OCR + GPT-OSS + load balancer)
make up-all

# Check service health
make health

# Run verification tests (14 tests across OCR, embeddings, search)
make test

# Run chain-of-thought demo
make demo-cot-quick

# Stop all services
make down
```

### Using Docker Compose Directly

```bash
cd vllm

# Core only (embeddings + OCR)
docker compose up -d

# With GPT-OSS inference
docker compose --profile gpt-oss up -d

# With scaled embeddings (2x replicas + load balancer)
docker compose --profile embeddings-scaled up -d

# All services
docker compose --profile embeddings-scaled --profile gpt-oss up -d
```

## Make Commands Reference

```bash
# Service Management
make up            # Start core services (embeddings + OCR)
make up-all        # Start all services (core + GPT-OSS + LB)
make up-scaled     # Start with 2x embeddings + load balancer
make up-gpt        # Start with GPT-OSS inference
make down          # Stop all services
make restart       # Restart all services
make status        # Show container status
make logs          # Follow service logs

# Testing & Verification
make health        # Quick health check of all services
make test          # Run full test suite (14 tests)
make verify        # Verify OCR, embeddings, and search

# Demos
make demo          # Run embeddings similarity demo
make demo-cot      # Run chain-of-thought demo (4 examples)
make demo-cot-quick # Quick CoT demo (2 examples)
make demo-pipeline # Run full OCR pipeline with Blue Book citations

# Benchmarking
make benchmark     # Run performance benchmark
make stress        # Stress test embedding load balancer
make stress-100    # Stress test with 100 concurrent requests

# Development
make install       # Install client dependencies
make build         # Build Docker images
make clean         # Clean output files
```

## Multi-Node Deployment (spark-1 + spark-2)

### Network Configuration

| Node | Role | Internal IP | Tailscale IP |
|------|------|-------------|--------------|
| spark-1 | Primary (vLLM services) | 192.168.1.76 | 100.70.220.58 |
| spark-2 | Worker (Ray workers) | 192.168.1.63 | 100.87.229.92 |

### Deploy to spark-1

```bash
# On spark-1
cd ~/Developer/inception-search-demo/vllm

# Deploy full stack with progress monitoring
make deploy

# Or manually with startup script
./scripts/hydra-start.sh
```

### Start Workers on spark-2

```bash
# From spark-1 (via SSH)
make workers

# Or directly on spark-2
cd ~/Developer/inception-search-demo/vllm
docker compose -f docker-compose.spark2.yml up -d
```

### Sync Files to spark-2

```bash
# Sync model configs and scripts
make sync-spark2

# Or manually
./scripts/sync-spark2.sh
```

### Verify Cluster

```bash
# Check all nodes
make verify-cluster

# Manual verification
curl http://192.168.1.76:8001/health  # spark-1 embeddings
curl http://192.168.1.76:8004/health  # spark-1 inference
curl http://192.168.1.63:8001/health  # spark-2 embeddings (if running)
```

## Architecture

```
                            vLLM Hydra Stack
+-------------------------------------------------------------------------+
|                                                                          |
|  CORE SERVICES (always started)                                          |
|  +---------------------------+  +---------------------------+            |
|  | vllm-freelaw-modernbert   |  |    vllm-hunyuanOCR        |            |
|  | (Embeddings)              |  |    (OCR)                  |            |
|  |                           |  |                           |            |
|  | Port: 8001                |  | Port: 8003                |            |
|  | Model: ModernBERT 512     |  | Model: Tencent HunyuanOCR |            |
|  | GPU: 10%                  |  | GPU: 40%                  |            |
|  +---------------------------+  +---------------------------+            |
|                                                                          |
|  LOAD BALANCING (--profile embeddings-scaled)                            |
|  +---------------------------+  +---------------------------+            |
|  | vllm-freelaw-modernbert-2 |  |    embeddings-lb          |            |
|  | (Replica 2)               |  |    (Traefik LB)           |            |
|  | Port: 8002                |  | Port: 8000 (unified)      |            |
|  +---------------------------+  | Dashboard: 8088           |            |
|                                 +---------------------------+            |
|                                                                          |
|  INFERENCE OPTIONS (choose one profile)                                  |
|  +---------------------------+  +---------------------------+            |
|  | vllm-gpt-oss-20b          |  |    vllm-llama             |            |
|  | (--profile gpt-oss)       |  |    (--profile llama)      |            |
|  |                           |  |                           |            |
|  | Port: 8004                |  | Port: 8004                |            |
|  | Model: GPT-OSS 20B        |  | Model: Llama 3.1 8B       |            |
|  | GPU: 45%                  |  | GPU: 45%                  |            |
|  | Features: Chain-of-Thought|  | Load: ~30s                |            |
|  +---------------------------+  +---------------------------+            |
|                                                                          |
+-------------------------------------------------------------------------+
```

## Services

### Core Services (always started)

| Service | Container | Port | Model | GPU | Purpose |
|---------|-----------|------|-------|-----|---------|
| Embeddings | `vllm-freelaw-modernbert` | 8001 | FreeLaw ModernBERT | 10% | Semantic search embeddings |
| OCR | `vllm-hunyuanOCR` | 8003 | Tencent HunyuanOCR | 40% | Document text extraction |

### Scaling Services (profile: embeddings-scaled)

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| Embeddings Replica 2 | `vllm-freelaw-modernbert-2` | 8002 | Second embedding instance |
| Load Balancer | `embeddings-lb` | 8000 | Traefik round-robin LB |
| LB Dashboard | - | 8088 | Traefik monitoring UI |

### Inference Options (choose one profile)

| Service | Container | Port | Model | Profile | Load Time |
|---------|-----------|------|-------|---------|-----------|
| GPT-OSS | `vllm-gpt-oss-20b` | 8004 | openai/gpt-oss-20b | `gpt-oss` | ~100-165s |
| GPT-OSS Eager | `vllm-gpt-oss-20b` | 8004 | openai/gpt-oss-20b | `gpt-oss-eager` | ~80s |
| Llama | `vllm-llama` | 8004 | meta-llama/Llama-3.1-8B | `llama` | ~30s |

## Client Commands

```bash
cd client && bun install

# Health & Testing
bun run health              # Check all services
bun run test                # Run service tests
bun run verify              # Run verification suite (14 tests)
bun run verify:ocr          # OCR verification only
bun run verify:embedding    # Embedding verification only
bun run verify:search       # Search ranking verification

# Demos
bun run demo                # Embeddings similarity demo
bun run cot                 # Chain-of-thought demo (4 examples)
bun run cot:quick           # Quick CoT demo (2 examples)
bun run cot:test            # CoT verification tests
bun run ocr-pipeline        # Full OCR pipeline with Blue Book citations

# Benchmarking
bun run benchmark           # Performance benchmark
bun run stress              # Stress test (50 concurrent, 30s)
bun run stress:compare      # Compare single vs load balanced
bun run stress --concurrency 100 --duration 60  # Custom stress test
```

## Benchmark Results

### Verification Tests

```
OCR             PASS (5/5 tests)
  ✓ Simple Text: 100% match
  ✓ Numbers: 91% match
  ✓ Legal Citation: 93% match
  ✓ Mixed Case: 86% match
  ✓ Multi-line Document: 4/4 phrases

Embeddings      PASS (5/5 tests)
  ✓ Semantic Similarity: 86% (min: 70%)
  ✓ Legal Synonyms: 75% (min: 60%)
  ✓ Identical Text: 100% (min: 99%)
  ✓ Topic Distinction: 21% (max: 50%)
  ✓ Domain Separation: 32% (max: 40%)

Search          PASS (4/4 tests)
  ✓ Contract Query → Document #1
  ✓ Patent Query → Document #2
  ✓ Criminal Law Query → Document #3
  ✓ Antitrust Query → Document #4
```

### Stress Test Results

| Configuration | Concurrency | Throughput | p95 Latency |
|---------------|-------------|------------|-------------|
| Single Replica | 20 | ~640 req/s | 38ms |
| Single Replica | 50 | ~850 req/s | 65ms |
| Load Balanced (2x) | 50 | ~1,400 req/s | 40ms |
| Load Balanced (2x) | 100 | ~1,800 req/s | 60ms |

### Chain-of-Thought Demo

| Example | Category | Duration | Tokens/sec |
|---------|----------|----------|------------|
| 1 | Mathematical Reasoning | 8.1s | 45 tok/s |
| 2 | Logical Deduction | 12.6s | 45 tok/s |
| 3 | Code Analysis | 13.2s | 44 tok/s |
| 4 | Legal Reasoning | 6.3s | 44 tok/s |

*Benchmarks run on NVIDIA GB200 with ModernBERT-512 and GPT-OSS 20B*

## API Usage

All services expose OpenAI-compatible APIs.

### Embeddings API

```bash
# Generate embeddings
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "freelawproject/modernbert-embed-base_finetune_512",
    "input": "Your text to embed"
  }'

# Load balanced endpoint
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "freelawproject/modernbert-embed-base_finetune_512",
    "input": ["Document 1", "Document 2", "Document 3"]
  }'
```

### OCR API

```bash
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
```

### Inference API (GPT-OSS with Chain-of-Thought)

```bash
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "What is 15 + 28? Show your reasoning."}
    ],
    "max_tokens": 512
  }'

# Response includes reasoning_content field with chain-of-thought
```

## Configuration

### Environment Variables (.env)

```bash
# Model Configuration
EMBEDDINGS_MODEL=freelawproject/modernbert-embed-base_finetune_512
EMBEDDINGS_GPU_MEMORY_UTIL=0.10

HUNYUAN_OCR_MODEL=tencent/HunyuanOCR
HUNYUAN_OCR_GPU_MEMORY_UTIL=0.40

GPT_OSS_20B_MODEL=openai/gpt-oss-20b
GPT_OSS_GPU_MEMORY_UTIL=0.45
GPT_OSS_20B_MAX_MODEL_LEN=8192

# GPU Configuration
TORCH_CUDA_ARCH_LIST=12.1a

# vLLM Docker image (NVIDIA 25.12+ recommended for GPT-OSS)
VLLM_IMAGE=nvcr.io/nvidia/vllm:25.12-py3

# Cache Directory
HF_CACHE_DIR=/home/rooot/.cache/huggingface

# Multi-node Configuration
SPARK1_IP=192.168.1.76
SPARK2_IP=192.168.1.63
```

## File Structure

```
vllm/
├── Makefile                    # Primary command interface
├── docker-compose.yml          # Main stack configuration
├── docker-compose.spark2.yml   # Multi-node worker config
├── traefik/                    # Load balancer configuration
│   ├── traefik.yml
│   └── dynamic.yml
├── client/                     # TypeScript demo client
│   ├── src/
│   │   ├── index.ts           # CLI commands
│   │   ├── cot-demo.ts        # Chain-of-thought demo
│   │   ├── verify-demo.ts     # OCR/embedding verification
│   │   ├── stress-test.ts     # Load balancer stress test
│   │   └── ocr-pipeline.ts    # Full OCR pipeline
│   └── package.json
├── scripts/
│   ├── hydra-start.sh         # Smart startup with progress
│   ├── hydra-down.sh          # Stop services
│   ├── hydra-monitor.sh       # Health monitoring
│   ├── sync-spark2.sh         # Multi-node sync
│   └── verify-services.sh     # Service verification
├── files/                      # Input PDFs for processing
├── output/                     # Results and generated files
├── .env.example               # Environment template
├── README.md                  # This file
├── CHANGELOG.md               # Version history
└── TODO.md                    # Roadmap
```

## Troubleshooting

### GPT-OSS Harmony Tokenizer Error

If you see `openai_harmony.HarmonyError: error downloading or loading vocab file`:

```bash
# Use NVIDIA vLLM 25.12+ which has the fix:
export VLLM_IMAGE=nvcr.io/nvidia/vllm:25.12-py3
make restart
```

### Services Not Starting

```bash
# Check container logs
make logs

# Verify GPU availability
nvidia-smi

# Check port conflicts
lsof -i :8001 :8002 :8003 :8004 :8000 :8088
```

### Load Balancer Not Working

```bash
# Check both replicas are healthy
curl http://localhost:8001/health
curl http://localhost:8002/health

# Check Traefik dashboard
open http://localhost:8088

# Check Traefik logs
docker logs embeddings-lb
```

## Related

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Traefik Documentation](https://doc.traefik.io/traefik/)
- [FreeLaw ModernBERT](https://huggingface.co/freelawproject/modernbert-embed-base_finetune_512)
- [Tencent HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
- [OpenAI GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b)
