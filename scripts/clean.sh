#!/bin/bash
# Inception ONNX - Clean Script
# Resets the project to a fresh state (preserves source and PDFs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo -e "${BLUE}━━━ Inception ONNX Clean ━━━${NC}\n"

# Parse arguments
CLEAN_MODELS=false
CLEAN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            CLEAN_MODELS=true
            shift
            ;;
        --all)
            CLEAN_ALL=true
            CLEAN_MODELS=true
            shift
            ;;
        *)
            echo "Usage: $0 [--models] [--all]"
            echo "  --models  Also remove converted models"
            echo "  --all     Remove everything including node_modules"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Stop any running services
log_info "Stopping running services..."
pkill -f "bun.*src/index.ts" 2>/dev/null || true
pkill -f "bun.*src/cli.ts" 2>/dev/null || true
docker compose down 2>/dev/null || true
log_success "Services stopped"

# Clean demo output
log_info "Cleaning demo output..."
rm -rf demo/output/*.json 2>/dev/null || true
rm -rf demo/output/benchmark-*.json 2>/dev/null || true
log_success "Demo output cleaned"

# Clean Python virtual environment
log_info "Cleaning Python environment..."
rm -rf converter/.venv 2>/dev/null || true
log_success "Python environment removed"

# Clean build artifacts
log_info "Cleaning build artifacts..."
rm -rf dist/ 2>/dev/null || true
rm -rf .bun/ 2>/dev/null || true
log_success "Build artifacts cleaned"

# Clean models if requested
if [ "$CLEAN_MODELS" = true ]; then
    log_info "Cleaning converted models..."
    find models/ -type d -name "*--*" -exec rm -rf {} + 2>/dev/null || true
    log_success "Models cleaned (registry.json preserved)"
fi

# Clean node_modules if --all
if [ "$CLEAN_ALL" = true ]; then
    log_info "Cleaning node_modules..."
    rm -rf node_modules/ 2>/dev/null || true
    rm -rf demo/node_modules/ 2>/dev/null || true
    rm -f bun.lock demo/bun.lock 2>/dev/null || true
    log_success "Node modules cleaned"
fi

echo -e "\n${GREEN}━━━ Clean Complete ━━━${NC}"
echo ""
echo "Project reset. To restore:"
echo "  ./scripts/setup.sh    # Full setup with model conversion"
echo "  bun install           # Just dependencies"
echo ""
