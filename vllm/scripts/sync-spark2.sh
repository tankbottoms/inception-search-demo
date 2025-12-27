#!/bin/bash
# vLLM Hydra - Sync spark-2 ray worker configuration
#
# This script rsyncs the necessary files to spark-2 for ray worker deployment
#
# Usage:
#   ./sync-spark2.sh              # Sync files to spark-2
#   ./sync-spark2.sh --dry-run    # Preview what would be synced

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
SPARK2_HOST="${SPARK2_HOST:-rooot@192.168.1.63}"
SPARK2_PATH="${SPARK2_PATH:-/home/rooot/Developer/vllm-hydra}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
DRY_RUN=""
if [ "$1" = "--dry-run" ] || [ "$1" = "-n" ]; then
    DRY_RUN="--dry-run"
    echo -e "${YELLOW}[DRY RUN]${NC} Preview mode - no files will be transferred"
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}     vLLM Hydra - Sync to Spark-2            ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Source: $VLLM_DIR"
echo "  Target: $SPARK2_HOST:$SPARK2_PATH"
echo ""

# Ensure target directory exists
echo -e "${BLUE}--- Creating target directory ---${NC}"
ssh "$SPARK2_HOST" "mkdir -p $SPARK2_PATH"
echo -e "${GREEN}[OK]${NC} Directory ready"
echo ""

# Files to sync
FILES_TO_SYNC=(
    "docker-compose.spark2.yml"
    ".env"
    ".env.example"
)

echo -e "${BLUE}--- Syncing configuration files ---${NC}"

for file in "${FILES_TO_SYNC[@]}"; do
    if [ -f "$VLLM_DIR/$file" ]; then
        echo -e "  Syncing: $file"
        rsync -avz $DRY_RUN "$VLLM_DIR/$file" "$SPARK2_HOST:$SPARK2_PATH/"
    else
        echo -e "${YELLOW}  [SKIP]${NC} $file (not found)"
    fi
done

echo ""

# Rename spark2 compose to docker-compose.yml on target
if [ -z "$DRY_RUN" ]; then
    echo -e "${BLUE}--- Setting up docker-compose.yml ---${NC}"
    ssh "$SPARK2_HOST" "cd $SPARK2_PATH && cp docker-compose.spark2.yml docker-compose.yml"
    echo -e "${GREEN}[OK]${NC} docker-compose.yml created from spark2 config"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}     Sync Complete                           ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}To start ray worker on spark-2:${NC}"
echo "  ssh $SPARK2_HOST 'cd $SPARK2_PATH && docker compose up -d'"
echo ""
