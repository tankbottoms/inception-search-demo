#!/bin/bash
# vLLM Hydra Cluster - Shutdown Script
#
# Usage:
#   ./hydra-down.sh                  # Stop all services
#   ./hydra-down.sh --keep-monitor   # Stop services but keep monitor
#   ./hydra-down.sh --service <name> # Stop specific service only
#   ./hydra-down.sh --distributed    # Stop distributed cluster (spark-1 + spark-2)
#   ./hydra-down.sh --volumes        # Also remove volumes
#   ./hydra-down.sh --force          # Force remove (kill)
#
# This script sets a flag to prevent the monitor from auto-restarting services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"
MANUAL_SHUTDOWN_FILE="/tmp/vllm-hydra-manual-shutdown"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default options
KEEP_MONITOR=false
REMOVE_VOLUMES=false
FORCE=false
DISTRIBUTED=false
SPECIFIC_SERVICE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-monitor|-k)
            KEEP_MONITOR=true
            shift
            ;;
        --volumes|-v)
            REMOVE_VOLUMES=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --distributed|-d)
            DISTRIBUTED=true
            shift
            ;;
        --service|-s)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "vLLM Hydra Cluster - Shutdown Script"
            echo ""
            echo "Usage:"
            echo "  ./hydra-down.sh                  # Stop all services"
            echo "  ./hydra-down.sh --keep-monitor   # Stop services but keep monitor"
            echo "  ./hydra-down.sh --service <name> # Stop specific service only"
            echo "  ./hydra-down.sh --distributed    # Stop distributed cluster"
            echo "  ./hydra-down.sh --volumes        # Also remove volumes"
            echo "  ./hydra-down.sh --force          # Force remove"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}       vLLM Hydra Cluster - Shutdown         ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd "$VLLM_DIR"

# Set manual shutdown flag to prevent monitor from restarting
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Manual shutdown initiated" > "$MANUAL_SHUTDOWN_FILE"
echo -e "${GREEN}[OK]${NC} Manual shutdown flag set (monitor will not auto-restart)"
echo ""

# Build compose command
COMPOSE_CMD="docker compose"
if [ "$FORCE" = "true" ]; then
    COMPOSE_CMD="$COMPOSE_CMD kill"
else
    COMPOSE_CMD="$COMPOSE_CMD down"
fi

if [ "$REMOVE_VOLUMES" = "true" ]; then
    COMPOSE_CMD="$COMPOSE_CMD --volumes"
fi

# Spark-2 configuration
SPARK2_HOST="${SPARK2_HOST:-rooot@spark-2}"
SPARK2_PATH="${SPARK2_PATH:-/home/rooot/Docker/vllm-hydra}"

# Stop services
if [ "$DISTRIBUTED" = "true" ]; then
    echo -e "${BLUE}[INFO]${NC} Stopping distributed cluster..."
    echo ""

    # Stop on spark-2 first
    echo -e "${YELLOW}Stopping spark-2...${NC}"
    ssh "$SPARK2_HOST" "cd $SPARK2_PATH && docker compose -f docker-compose.spark2.yml down" 2>/dev/null || true
    echo -e "${GREEN}[OK]${NC} spark-2 stopped"

    # Stop on spark-1
    echo -e "${YELLOW}Stopping spark-1...${NC}"
    docker compose --profile ray-cluster down 2>/dev/null || true
    echo -e "${GREEN}[OK]${NC} spark-1 stopped"

elif [ -n "$SPECIFIC_SERVICE" ]; then
    echo -e "${BLUE}[INFO]${NC} Stopping service: $SPECIFIC_SERVICE"
    docker compose stop "$SPECIFIC_SERVICE"
    docker compose rm -f "$SPECIFIC_SERVICE"
    echo -e "${GREEN}[OK]${NC} $SPECIFIC_SERVICE stopped"

else
    # Stop all services
    echo -e "${BLUE}[INFO]${NC} Stopping all vLLM Hydra services..."

    if [ "$KEEP_MONITOR" = "true" ]; then
        # Stop main services but keep monitor
        docker compose stop vllm-freelaw vllm-deepseekOCR vllm-inference
        docker compose rm -f vllm-freelaw vllm-deepseekOCR vllm-inference
        echo -e "${GREEN}[OK]${NC} Main services stopped (monitor kept running)"
    else
        # Stop everything
        $COMPOSE_CMD --profile monitor 2>/dev/null || $COMPOSE_CMD
        echo -e "${GREEN}[OK]${NC} All services stopped"
    fi
fi

echo ""

# Show remaining containers
remaining=$(docker compose ps -q 2>/dev/null | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo -e "${YELLOW}Remaining containers:${NC}"
    docker compose ps
else
    echo -e "${GREEN}[OK]${NC} All Hydra containers stopped"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}       vLLM Hydra Cluster Stopped           ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "To restart: ./scripts/hydra-up.sh"
echo "To remove shutdown flag: rm $MANUAL_SHUTDOWN_FILE"
echo ""
