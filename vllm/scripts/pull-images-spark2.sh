#!/bin/bash
# vLLM Hydra - Pull Docker images on spark-2
#
# This script pulls the same Docker images used on spark-1 to spark-2.
# Run this after updating images on spark-1 to keep spark-2 in sync.
#
# Usage:
#   ./pull-images-spark2.sh              # Pull all images
#   ./pull-images-spark2.sh --dry-run    # Show what would be pulled

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
SPARK2_HOST="${SPARK2_HOST:-rooot@192.168.1.63}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
DRY_RUN=false
if [ "$1" = "--dry-run" ] || [ "$1" = "-n" ]; then
    DRY_RUN=true
    echo -e "${YELLOW}[DRY RUN]${NC} Preview mode - no images will be pulled"
fi

# Load environment
if [ -f "$VLLM_DIR/.env" ]; then
    source "$VLLM_DIR/.env"
fi

# Default images
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.12-py3}"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}     vLLM Hydra - Pull Images on Spark-2     ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Target:     $SPARK2_HOST"
echo "  VLLM Image: $VLLM_IMAGE"
echo ""

# Images to pull
IMAGES=(
    "$VLLM_IMAGE"
)

echo -e "${BLUE}--- Pulling images on spark-2 ---${NC}"
echo ""

for image in "${IMAGES[@]}"; do
    echo -e "  ${YELLOW}Pulling:${NC} $image"
    if [ "$DRY_RUN" = true ]; then
        echo -e "    ${YELLOW}[DRY RUN]${NC} Would pull $image"
    else
        ssh "$SPARK2_HOST" "docker pull $image" 2>&1 | while read line; do
            echo "    $line"
        done
        echo -e "  ${GREEN}[OK]${NC} $image"
    fi
    echo ""
done

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}     Image Pull Complete                     ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Sync config:  ./scripts/sync-spark2.sh"
echo "  2. Deploy:       ssh $SPARK2_HOST 'cd /home/rooot/Developer/vllm-hydra && docker compose up -d'"
echo ""
