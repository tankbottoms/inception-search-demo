#!/bin/bash
# Stop vLLM distributed cluster on spark-1 and spark-2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Spark-2 configuration
SPARK2_HOST="${SPARK2_HOST:-rooot@spark-2}"
SPARK2_PATH="${SPARK2_PATH:-/home/rooot/Docker/vllm-hydra}"

echo "[INFO] Stopping vLLM cluster..."

# Stop spark-1
echo "[INFO] Stopping spark-1..."
docker compose -f "$VLLM_DIR/docker-compose.yml" --profile ray-cluster down 2>/dev/null || true

# Stop spark-2
echo "[INFO] Stopping spark-2..."
ssh "$SPARK2_HOST" "cd $SPARK2_PATH && docker compose -f docker-compose.spark2.yml down" 2>/dev/null || true

echo "[OK] vLLM cluster stopped"
