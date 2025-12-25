#!/bin/bash
# Monitor vLLM cluster health and restart if needed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

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
  SPARK1_STATUS=$(docker compose -f "$VLLM_DIR/docker-compose.spark-1.yml" ps --format json 2>/dev/null | jq -r '.[0].State' 2>/dev/null || echo "stopped")
  SPARK2_STATUS=$(ssh rooot@spark-2 "docker ps --filter name=vllm-worker --format '{{.State}}'" 2>/dev/null || echo "stopped")

  # Check vLLM API health
  VLLM_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://spark-1:8000/health 2>/dev/null || echo "000")

  if [[ "$CRON_MODE" == "true" ]]; then
    # Cron mode: only output on errors
    if [[ "$SPARK1_STATUS" != "running" ]] || [[ "$SPARK2_STATUS" != "running" ]] || [[ "$VLLM_HEALTH" != "200" ]]; then
      echo "[$timestamp] [ERROR] Cluster unhealthy - spark-1: $SPARK1_STATUS, spark-2: $SPARK2_STATUS, vLLM: $VLLM_HEALTH"
      echo "[$timestamp] [INFO] Attempting restart..."
      "$SCRIPT_DIR/start-cluster.sh"
    fi
  else
    # Interactive mode: always output status
    echo "[$timestamp] Cluster Status:"
    echo "  spark-1 container: $SPARK1_STATUS"
    echo "  spark-2 container: $SPARK2_STATUS"
    echo "  vLLM API health: $VLLM_HEALTH"

    if [[ "$SPARK1_STATUS" == "running" ]] && [[ "$SPARK2_STATUS" == "running" ]] && [[ "$VLLM_HEALTH" == "200" ]]; then
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
