#!/bin/bash
# vLLM Hydra Cluster - Startup Script
#
# Usage:
#   ./hydra-up.sh                    # Start core services (embeddings, OCR)
#   ./hydra-up.sh --monitor          # Start with monitor service
#   ./hydra-up.sh --build            # Rebuild and start
#   ./hydra-up.sh --service <name>   # Start specific service only
#   ./hydra-up.sh --gpt-oss-20b      # Start with GPT-OSS 20B
#   ./hydra-up.sh --gpt-oss-120b     # Start with GPT-OSS 120B (distributed)
#   ./hydra-up.sh --ray-cluster      # Start ray cluster for distributed inference
#   ./hydra-up.sh --spark2           # Also start services on spark-2
#
# Services:
#   - vllm-freelaw-modernbert-embed-base-finetune-512:  Embeddings
#   - vllm-deepSeekOCR:                                  DeepSeek OCR
#   - vllm-hunyuanOCR:                                   HunyuanOCR
#   - vllm-gpt-oss-20b:                                  GPT-OSS 20B inference (profile)
#   - ray-head:                                          Ray head for distributed (profile)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"
MANUAL_SHUTDOWN_FILE="/tmp/vllm-hydra-manual-shutdown"

# Spark-2 configuration
SPARK2_HOST="${SPARK2_HOST:-rooot@spark-2}"
SPARK2_PATH="${SPARK2_PATH:-/home/rooot/Docker/vllm-hydra}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default options
BUILD=false
WITH_MONITOR=false
WITH_SPARK2=false
PROFILE=""
SPECIFIC_SERVICE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build|-b)
            BUILD=true
            shift
            ;;
        --monitor|-m)
            WITH_MONITOR=true
            shift
            ;;
        --spark2)
            WITH_SPARK2=true
            shift
            ;;
        --gpt-oss-20b)
            PROFILE="gpt-oss-20b"
            shift
            ;;
        --gpt-oss-120b)
            PROFILE="gpt-oss-120b"
            WITH_SPARK2=true
            shift
            ;;
        --ray-cluster)
            PROFILE="ray-cluster"
            WITH_SPARK2=true
            shift
            ;;
        --service|-s)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "vLLM Hydra Cluster - Startup Script"
            echo ""
            echo "Usage:"
            echo "  ./hydra-up.sh                    # Start core services"
            echo "  ./hydra-up.sh --monitor          # Start with monitor service"
            echo "  ./hydra-up.sh --build            # Rebuild and start"
            echo "  ./hydra-up.sh --service <name>   # Start specific service only"
            echo "  ./hydra-up.sh --gpt-oss-20b      # Start with GPT-OSS 20B"
            echo "  ./hydra-up.sh --gpt-oss-120b     # Start with GPT-OSS 120B (distributed)"
            echo "  ./hydra-up.sh --ray-cluster      # Start ray cluster"
            echo "  ./hydra-up.sh --spark2           # Also start services on spark-2"
            echo ""
            echo "Services:"
            echo "  - vllm-freelaw-modernbert-embed-base-finetune-512  (port 8001)"
            echo "  - vllm-deepSeekOCR                                  (port 8002)"
            echo "  - vllm-hunyuanOCR                                   (port 8003)"
            echo "  - vllm-gpt-oss-20b                                  (port 8004, profile)"
            echo "  - ray-head                                          (ray cluster, profile)"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Clear manual shutdown flag
rm -f "$MANUAL_SHUTDOWN_FILE"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                   vLLM Hydra Cluster - Startup                   ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd "$VLLM_DIR"

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}[OK]${NC} Loaded environment from .env"
else
    echo -e "${YELLOW}[WARN]${NC} No .env file found, using defaults"
fi

echo ""
echo -e "${YELLOW}Services Configuration:${NC}"
echo "  Embeddings (vllm-freelaw-modernbert-embed-base-finetune-512): port ${EMBEDDINGS_PORT:-8001}"
echo "  DeepSeek OCR (vllm-deepSeekOCR):                              port ${DEEPSEEK_OCR_PORT:-8002}"
echo "  HunyuanOCR (vllm-hunyuanOCR):                                  port ${HUNYUAN_OCR_PORT:-8003}"
echo "  GPT-OSS 20B (vllm-gpt-oss-20b):                                port ${GPT_OSS_20B_PORT:-8004} (profile)"
echo ""

# Build if requested
if [ "$BUILD" = "true" ]; then
    echo -e "${BLUE}[INFO]${NC} Building containers..."
    if [ -n "$SPECIFIC_SERVICE" ]; then
        docker compose build "$SPECIFIC_SERVICE"
    else
        docker compose build
    fi
    echo -e "${GREEN}[OK]${NC} Build complete"
    echo ""
fi

# Sync to spark-2 if needed
if [ "$WITH_SPARK2" = "true" ]; then
    echo -e "${BLUE}[INFO]${NC} Syncing configuration to spark-2..."
    "$SCRIPT_DIR/sync-spark2.sh"
    echo ""
fi

# Build compose command
COMPOSE_CMD="docker compose"
if [ -n "$PROFILE" ]; then
    COMPOSE_CMD="$COMPOSE_CMD --profile $PROFILE"
fi
if [ "$WITH_MONITOR" = "true" ]; then
    COMPOSE_CMD="$COMPOSE_CMD --profile monitor"
fi

# Start services
if [ -n "$SPECIFIC_SERVICE" ]; then
    echo -e "${BLUE}[INFO]${NC} Starting service: $SPECIFIC_SERVICE"
    $COMPOSE_CMD up -d "$SPECIFIC_SERVICE"
    echo -e "${GREEN}[OK]${NC} $SPECIFIC_SERVICE started"
else
    echo -e "${BLUE}[INFO]${NC} Starting spark-1 services..."
    $COMPOSE_CMD up -d
    echo -e "${GREEN}[OK]${NC} Spark-1 services started"
fi

# Start spark-2 if requested
if [ "$WITH_SPARK2" = "true" ]; then
    echo ""
    echo -e "${BLUE}[INFO]${NC} Starting spark-2 ray worker..."
    ssh "$SPARK2_HOST" -f "cd $SPARK2_PATH && docker compose up -d"
    echo -e "${GREEN}[OK]${NC} Spark-2 ray worker started"
fi

echo ""

# Wait for services to be ready
echo -e "${BLUE}[INFO]${NC} Waiting for services to be ready..."
echo ""

MAX_WAIT=180
WAIT_INTERVAL=5
elapsed=0

while [ $elapsed -lt $MAX_WAIT ]; do
    # Check embeddings (required)
    embeddings_health=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${EMBEDDINGS_PORT:-8001}/health" 2>/dev/null || echo "000")

    if [ "$embeddings_health" = "200" ]; then
        break
    fi

    echo -e "  Waiting... ($elapsed/${MAX_WAIT}s)"
    sleep $WAIT_INTERVAL
    elapsed=$((elapsed + WAIT_INTERVAL))
done

echo ""

# Show final status
echo -e "${BLUE}━━━ Service Status ━━━${NC}"
echo ""
docker compose ps
echo ""

# Health check summary
embeddings_health=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${EMBEDDINGS_PORT:-8001}/health" 2>/dev/null || echo "000")
deepseek_ocr_health=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${DEEPSEEK_OCR_PORT:-8002}/health" 2>/dev/null || echo "000")
hunyuan_ocr_health=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${HUNYUAN_OCR_PORT:-8003}/health" 2>/dev/null || echo "000")
gpt_oss_health=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${GPT_OSS_20B_PORT:-8004}/health" 2>/dev/null || echo "000")

echo -e "${BLUE}━━━ Health Checks ━━━${NC}"
echo ""
if [ "$embeddings_health" = "200" ]; then
    echo -e "  vllm-freelaw-modernbert-embed-base-finetune-512:  ${GREEN}[OK]${NC}"
else
    echo -e "  vllm-freelaw-modernbert-embed-base-finetune-512:  ${YELLOW}[STARTING]${NC}"
fi

if [ "$deepseek_ocr_health" = "200" ]; then
    echo -e "  vllm-deepSeekOCR:                                  ${GREEN}[OK]${NC}"
else
    echo -e "  vllm-deepSeekOCR:                                  ${YELLOW}[STARTING]${NC}"
fi

if [ "$hunyuan_ocr_health" = "200" ]; then
    echo -e "  vllm-hunyuanOCR:                                   ${GREEN}[OK]${NC}"
else
    echo -e "  vllm-hunyuanOCR:                                   ${YELLOW}[STARTING]${NC}"
fi

if [ -n "$PROFILE" ] && [[ "$PROFILE" == *"gpt-oss"* ]]; then
    if [ "$gpt_oss_health" = "200" ]; then
        echo -e "  vllm-gpt-oss-20b:                                  ${GREEN}[OK]${NC}"
    else
        echo -e "  vllm-gpt-oss-20b:                                  ${YELLOW}[STARTING]${NC}"
    fi
fi

# Check spark-2 if running
if [ "$WITH_SPARK2" = "true" ]; then
    echo ""
    echo -e "${BLUE}━━━ Spark-2 Status ━━━${NC}"
    ssh "$SPARK2_HOST" "docker ps --filter 'name=ray' --format 'table {{.Names}}\t{{.Status}}'" 2>/dev/null || echo "  Unable to check spark-2"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                   vLLM Hydra Cluster Started                     ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "API Endpoints:"
echo "  Embeddings:   http://localhost:${EMBEDDINGS_PORT:-8001}/v1"
echo "  DeepSeek OCR: http://localhost:${DEEPSEEK_OCR_PORT:-8002}/v1"
echo "  HunyuanOCR:   http://localhost:${HUNYUAN_OCR_PORT:-8003}/v1"
if [ -n "$PROFILE" ] && [[ "$PROFILE" == *"gpt-oss"* ]]; then
    echo "  GPT-OSS:      http://localhost:${GPT_OSS_20B_PORT:-8004}/v1"
fi
echo ""
echo "Commands:"
echo "  View logs:   docker compose logs -f"
echo "  Stop:        ./scripts/hydra-down.sh"
echo "  Test:        ./scripts/verify-services.sh"
echo ""
