#!/bin/bash
# Stop vLLM distributed cluster on spark-1 and spark-2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

echo "[INFO] Stopping vLLM cluster..."

# Stop spark-1
echo "[INFO] Stopping spark-1..."
docker compose -f "$VLLM_DIR/docker-compose.spark-1.yml" down 2>/dev/null || true

# Stop spark-2
echo "[INFO] Stopping spark-2..."
ssh rooot@spark-2 "docker compose -f /home/rooot/Docker/vllm/docker-compose.yml down" 2>/dev/null || true

echo "[OK] vLLM cluster stopped"
