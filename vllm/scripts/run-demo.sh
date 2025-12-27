#!/bin/bash
# vLLM Hydra - Run Demo Client
#
# This script runs the containerized demo client with all necessary tools
# (graphicsmagick, ghostscript) for PDF processing.
#
# Usage:
#   ./scripts/run-demo.sh                    # Run full demo
#   ./scripts/run-demo.sh health             # Check services
#   ./scripts/run-demo.sh demo:status        # GPU and service status
#   ./scripts/run-demo.sh benchmark          # Run benchmarks
#   ./scripts/run-demo.sh demo:full --pages 5 # Custom options

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "$VLLM_DIR"

# Check if files directory exists
if [ ! -d "files" ]; then
    mkdir -p files
    echo -e "${YELLOW}Created files/ directory. Add PDF files here for processing.${NC}"
fi

# Check if output directory exists
if [ ! -d "output" ]; then
    mkdir -p output
fi

# Build client image if needed
echo -e "${BLUE}Checking client image...${NC}"
if ! docker images | grep -q "vllm-hydra-hydra-client"; then
    echo -e "${YELLOW}Building client image (first time only)...${NC}"
    docker compose build hydra-client
fi

# Default command
CMD="${1:-demo:full}"
shift 2>/dev/null || true

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                   vLLM Hydra - Demo Client                       ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "Command: ${GREEN}$CMD $@${NC}"
echo ""

# Run the demo
docker compose run --rm hydra-client "$CMD" "$@"

echo ""
echo -e "${GREEN}Demo complete!${NC}"
echo -e "Results saved to: ${VLLM_DIR}/output/"
echo ""
