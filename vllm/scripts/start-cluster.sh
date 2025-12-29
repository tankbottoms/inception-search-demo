#!/bin/bash
# Start vLLM distributed cluster on spark-1 and spark-2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Default model
MODEL="${MODEL:-tencent/HunyuanOCR}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --max-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-util)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "[INFO] Starting vLLM distributed cluster"
echo "[INFO] Model: $MODEL"
echo "[INFO] Tensor Parallel: $TENSOR_PARALLEL"
echo "[INFO] GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo ""

# Spark-2 configuration
SPARK2_HOST="${SPARK2_HOST:-rooot@spark-2}"
SPARK2_PATH="${SPARK2_PATH:-/home/rooot/Docker/vllm-hydra}"

# Stop existing containers
echo "[INFO] Stopping existing containers..."
docker compose -f "$VLLM_DIR/docker-compose.yml" --profile ray-cluster down 2>/dev/null || true
ssh "$SPARK2_HOST" "cd $SPARK2_PATH && docker compose -f docker-compose.spark2.yml down" 2>/dev/null || true

sleep 2

# Start spark-1 (head node)
echo "[INFO] Starting spark-1 (head node)..."
docker compose -f "$VLLM_DIR/docker-compose.yml" --profile ray-cluster up --build -d

sleep 5

# Start spark-2 (worker node)
echo "[INFO] Starting spark-2 (worker node)..."
ssh "$SPARK2_HOST" "cd $SPARK2_PATH && docker compose -f docker-compose.spark2.yml up --build -d"

sleep 5

# Check Ray cluster status
echo "[INFO] Checking Ray cluster status..."
docker compose -f "$VLLM_DIR/docker-compose.yml" exec ray-head ray status || true

# Start vLLM server
echo "[INFO] Starting vLLM server with model: $MODEL"
docker compose -f "$VLLM_DIR/docker-compose.yml" exec ray-head -d bash -c \
  "vllm serve $MODEL \
    --max_model_len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --host 0.0.0.0 \
    --port 8000"

echo ""
echo "[OK] vLLM cluster started"
echo "[INFO] API endpoint: http://spark-1:8000/v1"
echo "[INFO] Health check: curl http://spark-1:8000/health"
