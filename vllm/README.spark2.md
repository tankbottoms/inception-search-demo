# vLLM Hydra - Spark-2 Workers

Worker node for the vLLM Hydra cluster, providing additional inference capacity.

## Node Info

- **Hostname**: spark-2
- **IP**: 192.168.1.63
- **Tailscale**: 100.87.229.92
- **GPU**: NVIDIA GB10

## Services

| Service | Container | Port | GPU | Image | Profile |
|---------|-----------|------|-----|-------|---------|
| Embeddings | vllm-freelaw-modernbert-worker | 8001 | 10% | vllm-openai:nightly | (default) |
| GPT-OSS | vllm-gpt-oss-20b-worker | 8004 | 50% | nvidia/vllm:25.12 | vllm-gpt-oss-20b |
| OCR | vllm-hunyuanOCR-worker | 8003 | 40% | vllm-openai:nightly | vllm-hunyuanOCR |

## Quick Start

```bash
# Start embeddings worker (default)
docker compose up -d

# Add GPT-OSS inference (with embeddings)
docker compose --profile vllm-gpt-oss-20b up -d

# Add OCR worker (with embeddings)
docker compose --profile vllm-hunyuanOCR up -d

# Stop all
docker compose --profile vllm-gpt-oss-20b --profile vllm-hunyuanOCR down
```

## GPU Memory Constraints

| Service | Allocation | Actual Usage |
|---------|-----------|--------------|
| Embeddings | 10% | ~1 GiB |
| GPT-OSS | 45-50% | ~54 GiB |
| OCR | 40% | ~48 GiB |

**IMPORTANT:** GPT-OSS and OCR cannot run simultaneously due to CUDA graph
compilation conflicts (PTX toolchain issue). Choose one configuration:

```bash
# Option 1: Embeddings + GPT-OSS (for inference)
docker compose --profile vllm-gpt-oss-20b up -d

# Option 2: Embeddings + OCR (for document processing)
docker compose --profile vllm-hunyuanOCR up -d
```

Note: Spark-1 can run all services because they were started sequentially
and the CUDA graphs were compiled without interference.

## Load Balancing

The embeddings worker on spark-2 is included in the Traefik load balancer pool on spark-1:

```
http://spark-1:8000/v1/embeddings →
  - spark-1:8001 (vllm-freelaw-modernbert)
  - spark-1:8002 (vllm-freelaw-modernbert-2)
  - spark-2:8001 (vllm-freelaw-modernbert-worker)
```

## Health Checks

```bash
# Check embeddings
curl http://localhost:8001/health

# Check GPT-OSS (if running)
curl http://localhost:8004/health

# Check OCR (if running)
curl http://localhost:8003/health
```

## Logs

```bash
docker logs -f vllm-freelaw-modernbert-worker
docker logs -f vllm-gpt-oss-20b-worker
docker logs -f vllm-hunyuanOCR-worker
```

## Files

```
/home/rooot/Docker/vllm-hydra/
├── docker-compose.yml      # Main compose file (copied from spark2.yml)
├── .env                    # Environment variables
└── README.md               # This file
```

## Syncing from Spark-1

Files are synced from spark-1 using:
```bash
# On spark-1
./scripts/sync-spark2.sh
```

This syncs:
- docker-compose.spark2.yml → docker-compose.yml
- .env
- README.spark2.md → README.md
