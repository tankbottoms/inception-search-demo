#!/bin/bash
# Monitor vLLM cluster health and restart if needed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment
cd "$VLLM_DIR"
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
SPARK1_IP="${SPARK1_IP:-192.168.100.10}"
EMBEDDINGS_PORT="${EMBEDDINGS_PORT:-8001}"

CRON_MODE=false
WATCH_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cron)
      CRON_MODE=true
      shift
      ;;
    --watch)
      WATCH_MODE=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

check_health() {
  local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  # Check if containers are running
  SPARK1_STATUS=$(docker compose ps --format json 2>/dev/null | jq -r '.[0].State' 2>/dev/null || echo "stopped")
  SPARK2_STATUS=$(ssh "${SPARK2_HOST:-rooot@spark-2}" "docker ps --filter name=ray-worker --format '{{.State}}'" 2>/dev/null || echo "stopped")

  # Check vLLM API health (embeddings service)
  VLLM_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${EMBEDDINGS_PORT}/health" 2>/dev/null || echo "000")

  if [[ "$CRON_MODE" == "true" ]]; then
    # Cron mode: only output on errors
    if [[ "$SPARK1_STATUS" != "running" ]] || [[ "$VLLM_HEALTH" != "200" ]]; then
      echo "[$timestamp] [ERROR] Cluster unhealthy - spark-1: $SPARK1_STATUS, spark-2: $SPARK2_STATUS, vLLM API: $VLLM_HEALTH"
      echo "[$timestamp] [INFO] Attempting restart..."
      "$SCRIPT_DIR/hydra-up.sh"
    fi
  else
    # Interactive mode: always output status
    echo "[$timestamp] Cluster Status:"
    echo "  spark-1 containers: $SPARK1_STATUS"
    echo "  spark-2 containers: $SPARK2_STATUS"
    echo "  vLLM API health:    $VLLM_HEALTH (port $EMBEDDINGS_PORT)"

    if [[ "$SPARK1_STATUS" == "running" ]] && [[ "$VLLM_HEALTH" == "200" ]]; then
      echo "  [OK] Cluster healthy"
    else
      echo "  [WARN] Cluster unhealthy"
    fi
  fi
}

if [[ "$WATCH_MODE" == "true" ]]; then
  echo "[INFO] Monitoring cluster (Ctrl+C to stop)..."
  while true; do
    check_health
    sleep 30
  done
else
  check_health
fi
