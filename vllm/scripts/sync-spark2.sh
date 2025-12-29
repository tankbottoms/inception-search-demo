#!/bin/bash
# vLLM Hydra - Sync configuration to spark-2
#
# This script syncs the necessary files to spark-2 for worker deployment
#
# Usage:
#   ./sync-spark2.sh              # Sync files to spark-2
#   ./sync-spark2.sh --dry-run    # Preview what would be synced

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
SPARK2_HOST="${SPARK2_HOST:-rooot@192.168.1.63}"
SPARK2_PATH="${SPARK2_PATH:-/home/rooot/Docker/vllm-hydra}"

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
echo -e "${BLUE}--- Syncing configuration files ---${NC}"

# Sync docker-compose.spark2.yml
if [ -f "$VLLM_DIR/docker-compose.spark2.yml" ]; then
    echo -e "  Syncing: docker-compose.spark2.yml"
    rsync -avz $DRY_RUN "$VLLM_DIR/docker-compose.spark2.yml" "$SPARK2_HOST:$SPARK2_PATH/"
else
    echo -e "${RED}  [ERROR]${NC} docker-compose.spark2.yml not found!"
    exit 1
fi

# Sync .env files
for file in ".env" ".env.example"; do
    if [ -f "$VLLM_DIR/$file" ]; then
        echo -e "  Syncing: $file"
        rsync -avz $DRY_RUN "$VLLM_DIR/$file" "$SPARK2_HOST:$SPARK2_PATH/"
    fi
done

# Sync README
if [ -f "$VLLM_DIR/README.spark2.md" ]; then
    echo -e "  Syncing: README.spark2.md -> README.md"
    rsync -avz $DRY_RUN "$VLLM_DIR/README.spark2.md" "$SPARK2_HOST:$SPARK2_PATH/README.md"
fi

echo ""

# Set up docker-compose.yml and clean up old files
if [ -z "$DRY_RUN" ]; then
    echo -e "${BLUE}--- Setting up spark-2 ---${NC}"

    # Copy spark2 compose to docker-compose.yml
    ssh "$SPARK2_HOST" "cd $SPARK2_PATH && cp docker-compose.spark2.yml docker-compose.yml"
    echo -e "${GREEN}[OK]${NC} docker-compose.yml created"

    # Remove old spark2.yml to avoid confusion
    ssh "$SPARK2_HOST" "cd $SPARK2_PATH && rm -f docker-compose.spark2.yml"
    echo -e "${GREEN}[OK]${NC} Cleaned up docker-compose.spark2.yml"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}     Sync Complete                           ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}To start services on spark-2:${NC}"
echo ""
echo "  # Core services (embeddings + OCR) - default"
echo "  ssh $SPARK2_HOST 'cd $SPARK2_PATH && docker compose up -d'"
echo ""
echo "  # With GPT-OSS (optional, conflicts with OCR)"
echo "  ssh $SPARK2_HOST 'cd $SPARK2_PATH && docker compose --profile vllm-gpt-oss-20b up -d'"
echo ""
echo "  # Or use make commands from spark-1:"
echo "  make spark2-up          # Core services"
echo "  make spark2-up-gpt      # Add GPT-OSS (warning: conflicts with OCR)"
echo ""
