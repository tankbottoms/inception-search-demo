# vLLM Distributed Inference

Distributed inference configuration for dual Nvidia DGX Spark nodes using vLLM and Ray.

## Overview

This setup enables distributed model inference across two DGX Spark nodes (spark-1 and spark-2) using:

- **vLLM**: High-throughput LLM serving
- **Ray**: Distributed computing framework
- **Tensor Parallelism**: Split model across multiple GPUs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ray Cluster                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐        ┌─────────────────────┐         │
│  │      spark-1        │        │      spark-2        │         │
│  │   (Head Node)       │◄──────►│   (Worker Node)     │         │
│  │                     │        │                     │         │
│  │  ┌───────────────┐  │        │  ┌───────────────┐  │         │
│  │  │ Ray Head      │  │        │  │ Ray Worker    │  │         │
│  │  │ Port: 6379    │  │        │  │               │  │         │
│  │  └───────────────┘  │        │  └───────────────┘  │         │
│  │                     │        │                     │         │
│  │  ┌───────────────┐  │        │  ┌───────────────┐  │         │
│  │  │ vLLM Server   │  │        │  │ vLLM Worker   │  │         │
│  │  │ Port: 8000    │  │        │  │               │  │         │
│  │  └───────────────┘  │        │  └───────────────┘  │         │
│  │                     │        │                     │         │
│  │  GPU: Blackwell     │        │  GPU: Blackwell     │         │
│  │  Memory: 128GB      │        │  Memory: 128GB      │         │
│  └─────────────────────┘        └─────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Network Configuration

| Node | IP Address | Role |
|------|------------|------|
| spark-1 | 192.168.100.10 | Ray head, vLLM server |
| spark-2 | 192.168.100.11 | Ray worker |

**Network Interface**: `enP2p1s0f1np1` (high-speed interconnect)

## Quick Start

### Start Cluster

```bash
# From spark-1
./scripts/start-cluster.sh

# With specific model
./scripts/start-cluster.sh --model tencent/HunyuanOCR
```

### Stop Cluster

```bash
./scripts/stop-cluster.sh
```

### Monitor Cluster

```bash
# One-time check
./scripts/monitor.sh

# Continuous monitoring (for crontab)
./scripts/monitor.sh --watch
```

## Supported Models

| Model | Type | Tensor Parallel | Memory |
|-------|------|-----------------|--------|
| `tencent/HunyuanOCR` | OCR | 2 | ~40GB |
| `deepseek-ai/DeepSeek-OCR` | OCR | 2 | ~30GB |
| `openai/gpt-oss-120b` | LLM | 2 | ~100GB |
| `Qwen/Qwen3-Next-80B-A3B-Thinking` | LLM | 2 | ~80GB |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MASTER_ADDR` | `192.168.100.10` | Ray head node IP |
| `VLLM_HOST_IP` | Node-specific | This node's IP |
| `UCX_NET_DEVICES` | `enP2p1s0f1np1` | Network interface |
| `RAY_memory_monitor_refresh_ms` | `0` | Disable memory monitor |

### CUDA Configuration

| Variable | Value | Description |
|----------|-------|-------------|
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | CUDA architecture (Spark 12.0, 12.1f, 12.1a) |
| `TRITON_PTXAS_PATH` | `/usr/local/cuda/bin/ptxas` | Triton compiler path |

### vLLM Server Options

```bash
vllm serve <model> \
  --tensor-parallel-size 2 \        # Distribute across 2 GPUs
  --gpu-memory-utilization 0.85 \   # Use 85% of GPU memory
  --max-model-len 131072 \          # Max context length
  --host 0.0.0.0 \
  --port 8000
```

## Docker Images

- **NVIDIA vLLM**: `nvcr.io/nvidia/vllm:25.10-py3`
- **Alternative**: `vllm/vllm-openai:nightly-aarch64`

## Files

```
vllm/
├── README.md                     # This file
├── docker-compose.spark-1.yml    # Head node configuration
├── docker-compose.spark-2.yml    # Worker node configuration
└── scripts/
    ├── start-cluster.sh          # Start Ray cluster + vLLM
    ├── stop-cluster.sh           # Stop all services
    └── monitor.sh                # Health monitoring
```

## Troubleshooting

### Ray Connection Issues

```bash
# Check Ray status on spark-1
docker compose -f docker-compose.spark-1.yml exec vllm ray status

# Check network connectivity
ping -c 3 192.168.100.11
```

### GPU Memory Issues

```bash
# Check GPU memory
docker compose exec vllm nvidia-smi

# Reduce memory utilization
vllm serve <model> --gpu-memory-utilization 0.50
```

### Model Loading Failures

```bash
# Check HuggingFace cache
ls -la /home/rooot/.cache/huggingface

# Pre-download model
docker compose exec vllm huggingface-cli download <model>
```

## Crontab Monitoring

Add to crontab for automatic restart on failure:

```bash
# Check every 5 minutes
*/5 * * * * /path/to/vllm/scripts/monitor.sh --cron >> /var/log/vllm-monitor.log 2>&1
```

## API Usage

Once running, the vLLM server exposes an OpenAI-compatible API:

```bash
curl http://spark-1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tencent/HunyuanOCR",
    "messages": [{"role": "user", "content": "Extract text from this image"}]
  }'
```

## Performance Notes

- **Tensor Parallel = 2**: Model split across both nodes
- **Shared Memory**: 10.24GB per container for inter-process communication
- **Network Mode**: Host networking for lowest latency
- **GPU Utilization**: 85% default, adjust based on other services (e.g., Ollama)

## Related

- [vLLM Documentation](https://docs.vllm.ai/)
- [Ray Documentation](https://docs.ray.io/)
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/data-center/dgx-spark/)
