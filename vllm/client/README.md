# vLLM Hydra Client

TypeScript CLI client for the vLLM Hydra multi-model inference cluster. Provides health checks, demos, and comprehensive benchmarking with load balancer support.

## QUICKSTART

```bash
# 1. Install dependencies
bun install

# 2. Check service health
bun run src/index.ts health

# 3. Run embeddings demo
bun run src/index.ts demo --query "contract dispute"

# 4. Run bandwidth benchmark
bun run src/index.ts benchmark --skip-ocr
```

## Services

| Service | Port | Model | Purpose |
|---------|------|-------|---------|
| Embeddings | 8001 | FreeLaw ModernBERT | Text embeddings (768-dim) |
| Embeddings LB | 8000 | Traefik | Load balanced embeddings |
| OCR | 8003 | Tencent HunyuanOCR | Vision-based OCR |
| Inference | 8004 | GPT-OSS 20B / Llama 3.1 | Text generation |

## Commands

### Health Check

```bash
bun run src/index.ts health
```

Shows status of all services with response times and model info.

### Service Tests

```bash
bun run src/index.ts test
```

Runs functional tests for embeddings, OCR, and inference services.

### Embeddings Demo

```bash
bun run src/index.ts demo
bun run src/index.ts demo --query "intellectual property lawsuit"
```

Generates embeddings for sample documents and performs similarity search.

### OCR Processing

```bash
# Process a PDF
bun run src/index.ts ocr --pdf ../files/document.pdf

# Process an image
bun run src/index.ts ocr --image ../files/page.png

# Save output
bun run src/index.ts ocr --pdf ../files/document.pdf --output extracted.txt
```

### Full Pipeline

```bash
bun run src/index.ts pipeline
bun run src/index.ts pipeline --pdf-count 5 --query "damages"
bun run src/index.ts pipeline --force-ocr
```

Complete workflow: PDF -> OCR -> Embed -> Search

### Performance Benchmark

```bash
# Full benchmark
bun run src/index.ts benchmark

# Quick benchmark (skip OCR)
bun run src/index.ts benchmark --skip-ocr

# Custom settings
bun run src/index.ts benchmark \
  --iterations 20 \
  --requests 100 \
  --concurrency "1,5,10,20,50" \
  --skip-ocr

# Skip load balancer tests
bun run src/index.ts benchmark --skip-lb
```

## Benchmark Features

The benchmark command provides comprehensive performance testing:

### Latency Benchmark (Sequential)

Tests single-request latency with different text lengths:
- Short (23 chars)
- Medium (93 chars)
- Long (243 chars)

Reports min/max/p95 statistics.

### Throughput Benchmark (Concurrent)

Tests concurrent request handling:
- Multiple concurrency levels (1, 5, 10, 20)
- Single replica vs load-balanced endpoint comparison
- Head-to-head comparison with improvement percentage

### Batch Embedding Test

Tests batch API efficiency:
- Batch sizes: 1, 5, 10, 20 documents
- Per-document latency and throughput

### Output

Results saved to `../output/benchmark-{timestamp}.json`:

```json
{
  "timestamp": "2025-12-27T...",
  "config": {
    "iterations": 10,
    "totalRequests": 50,
    "concurrencyLevels": [1, 5, 10, 20],
    "singleEndpoint": "http://localhost:8001",
    "lbEndpoint": "http://localhost:8000"
  },
  "services": {
    "singleReplica": { "url": "...", "available": true, "model": "..." },
    "loadBalanced": { "url": "...", "available": true }
  },
  "results": [...],
  "summary": {
    "totalTests": 15,
    "avgLatencyMs": 9.2,
    "peakThroughput": 645.3,
    "lbImprovement": 87.5
  }
}
```

## Environment Variables

```bash
# Service endpoints
EMBEDDINGS_URL=http://localhost:8001
LB_URL=http://localhost:8000
OCR_URL=http://localhost:8003
INFERENCE_URL=http://localhost:8004

# Directories
FILES_DIR=../files
OUTPUT_DIR=../output
```

## Example Output

### Health Check

```
--- vLLM Hydra Health Check ---

OK Embeddings (vllm-freelaw-modernbert)
   URL: http://localhost:8001
   Response: 24ms
   Model: freelawproject/modernbert-embed-base_finetune_512

OK OCR (vllm-hunyuanOCR)
   URL: http://localhost:8003
   Response: 3ms
   Model: tencent/HunyuanOCR

--- All services healthy ---
```

### Benchmark

```
======================================================================
              vLLM Hydra Bandwidth Benchmark
======================================================================

Service Discovery
--------------------------------------------------
  [OK] Single Replica (http://localhost:8001) - freelawproject/modernbert-...
  [OK] Load Balanced (http://localhost:8000) - Traefik LB
  [OK] OCR - tencent/HunyuanOCR

Latency Benchmark (Sequential)
--------------------------------------------------
  Iterations: 10

  Short (23 chars): 9.4ms (min: 9.2, max: 9.5, p95: 9.5)
  Medium (93 chars): 9.1ms (min: 8.8, max: 9.4, p95: 9.4)
  Long (243 chars): 8.4ms (min: 8.4, max: 8.5, p95: 8.5)

Throughput Benchmark (Concurrent)
--------------------------------------------------

  Single Replica Performance:
    50 reqs @ 10 concurrent: 279.3 req/s (179ms total, 50/50 success)
    50 reqs @ 20 concurrent: 324.1 req/s (154ms total, 50/50 success)

  Load Balanced Performance:
    50 reqs @ 10 concurrent: 412.5 req/s (121ms total, 50/50 success)
    50 reqs @ 20 concurrent: 645.2 req/s (78ms total, 50/50 success)

  Head-to-Head Comparison:
    Single Replica: 324.1 req/s
    Load Balanced:  645.2 req/s
    Improvement:    +99.1%

Batch Embedding Test
--------------------------------------------------
  Batch of 1: 10ms total (10.1ms/doc, 99.4 docs/s)
  Batch of 5: 14ms total (2.8ms/doc, 359.5 docs/s)
  Batch of 10: 18ms total (1.8ms/doc, 549.4 docs/s)
  Batch of 20: 24ms total (1.2ms/doc, 849.5 docs/s)

======================================================================
                      Benchmark Summary
======================================================================

Peak Performance:
  Load Balanced - Throughput @ 20 concurrent: 645.2 req/s

Report saved: ../output/benchmark-1735315048064.json

======================================================================
                    Benchmark Complete!
======================================================================
```

## File Structure

```
client/
├── src/
│   ├── index.ts        # Main CLI with all commands
│   └── demo.ts         # Full demo workflow with OCR
├── package.json
├── tsconfig.json
├── Dockerfile          # Container with PDF tools
├── README.md           # This file
├── CHANGELOG.md
└── TODO.md
```

## Docker Usage

Run the client in a container with all PDF processing tools:

```bash
# From vllm/ directory
docker compose run --rm hydra-client health
docker compose run --rm hydra-client demo
docker compose run --rm hydra-client benchmark --skip-ocr
```

## Requirements

- Bun runtime (or Node.js 18+)
- vLLM Hydra services running
- For OCR/PDF: graphicsmagick, ghostscript (included in Docker image)

## Tips

1. **Start with health check**: Always verify services are running first
2. **Use --skip-ocr for quick benchmarks**: OCR is slow and not needed for embedding tests
3. **Enable load balancing**: Use `--profile embeddings-scaled` for 2x throughput
4. **Batch for throughput**: Use batch API (array of inputs) for 8x improvement
5. **Review JSON reports**: Full benchmark data saved to output directory
