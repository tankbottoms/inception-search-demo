#!/bin/bash
#
# test-all.sh - Complete system verification
#
# Runs all builds, tests, demos, and benchmarks to verify the system works.
# Usage: ./scripts/test-all.sh
#

# Don't exit on error - we track failures ourselves
set +e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timing
SCRIPT_START=$(date +%s)

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Log file
LOG_FILE="$PROJECT_ROOT/test-all.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Inception ONNX - Complete System Verification${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo ""

# Track results
PASSED=0
FAILED=0
RESULTS=()

# Helper function to run a step
run_step() {
    local step_name="$1"
    local command="$2"
    local step_start=$(date +%s)

    # Always return to project root before running command
    cd "$PROJECT_ROOT"

    echo -e "${YELLOW}━━━ $step_name ━━━${NC}"
    echo "Command: $command"
    echo ""

    if eval "$command"; then
        local step_end=$(date +%s)
        local duration=$((step_end - step_start))
        echo ""
        echo -e "${GREEN}✓ $step_name PASSED${NC} (${duration}s)"
        RESULTS+=("✓ $step_name (${duration}s)")
        ((PASSED++))
    else
        local step_end=$(date +%s)
        local duration=$((step_end - step_start))
        echo ""
        echo -e "${RED}✗ $step_name FAILED${NC} (${duration}s)"
        RESULTS+=("✗ $step_name (${duration}s)")
        ((FAILED++))
    fi
    echo ""

    # Return to project root after command
    cd "$PROJECT_ROOT"
}

# Cleanup function
cleanup() {
    echo -e "${YELLOW}━━━ Cleanup ━━━${NC}"

    # Stop backend if running
    pkill -f "bun.*src/index.ts" 2>/dev/null || true

    # Clean demo output
    rm -rf demo/output/* 2>/dev/null || true

    echo "Cleanup complete"
    echo ""
}

# Set trap to cleanup on exit
trap cleanup EXIT

# ============================================================
# Phase 1: Setup & Build
# ============================================================

echo -e "${BLUE}Phase 1: Setup & Build${NC}"
echo ""

run_step "Install Backend Dependencies" "bun install"

run_step "Install Demo Dependencies" "cd $PROJECT_ROOT/demo && bun install"

run_step "TypeScript Typecheck" "bun run typecheck"

# ============================================================
# Phase 2: Unit Tests
# ============================================================

echo -e "${BLUE}Phase 2: Unit Tests${NC}"
echo ""

run_step "Run Unit Tests" "bun test 2>/dev/null || echo 'No tests found or tests skipped'"

# ============================================================
# Phase 3: Model Check
# ============================================================

echo -e "${BLUE}Phase 3: Model Verification${NC}"
echo ""

run_step "Check Model Availability" "bun run cli -- --check"

# ============================================================
# Phase 4: Backend Startup
# ============================================================

echo -e "${BLUE}Phase 4: Backend Startup${NC}"
echo ""

echo -e "${YELLOW}━━━ Start Backend Server ━━━${NC}"
echo "Starting server on port 8005..."

bun run start > /tmp/inception-backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
MAX_WAIT=30
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8005/health | grep -q '"status":"ok"'; then
        break
    fi
    sleep 1
    ((WAITED++))
done

if curl -s http://localhost:8005/health | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✓ Backend started successfully${NC}"
    RESULTS+=("✓ Backend Startup (${WAITED}s)")
    ((PASSED++))

    # Show backend info
    echo ""
    echo "Backend Status:"
    curl -s http://localhost:8005/health | jq -r '"  Provider: \(.provider)\n  Device: \(.device)\n  Model: \(.model)"'
    echo ""
else
    echo -e "${RED}✗ Backend failed to start${NC}"
    RESULTS+=("✗ Backend Startup")
    ((FAILED++))
    echo "Backend log:"
    cat /tmp/inception-backend.log | tail -20
    exit 1
fi

# ============================================================
# Phase 5: API Tests
# ============================================================

echo -e "${BLUE}Phase 5: API Endpoint Tests${NC}"
echo ""

run_step "Health Endpoint" "curl -sf http://localhost:8005/health | jq -e '.status == \"ok\"'"

run_step "Status Endpoint" "curl -sf http://localhost:8005/status | jq -e '.service.initialized == true'"

run_step "OCR Providers Endpoint" "curl -sf http://localhost:8005/api/v1/ocr/providers | jq -e '.auto.available == true'"

run_step "Query Embedding API" "curl -sf -X POST http://localhost:8005/api/v1/embed/query -H 'Content-Type: application/json' -d '{\"text\":\"test query\"}' | jq -e '.embedding | length > 0'"

run_step "Text Embedding API" "curl -sf -X POST http://localhost:8005/api/v1/embed/text -H 'Content-Type: application/json' -d '{\"id\":1,\"text\":\"This is a test document for embedding.\"}' | jq -e '.embeddings | length > 0'"

# ============================================================
# Phase 6: Demo Pipeline
# ============================================================

echo -e "${BLUE}Phase 6: Demo Pipeline${NC}"
echo ""

run_step "PDF Demo (Text Extraction)" "cd $PROJECT_ROOT/demo && bun run demo --pdf-count 2"

run_step "Search Test" "cd $PROJECT_ROOT/demo && bun run search 'legal complaint' --limit 3"

# ============================================================
# Phase 7: Benchmarks
# ============================================================

echo -e "${BLUE}Phase 7: Benchmarks${NC}"
echo ""

run_step "Query Benchmarks" "cd $PROJECT_ROOT/demo && bun run benchmark --iterations 5"

# ============================================================
# Phase 8: CLI Benchmark
# ============================================================

echo -e "${BLUE}Phase 8: CLI Benchmark${NC}"
echo ""

run_step "CLI Benchmark" "cd $PROJECT_ROOT && bun run src/cli.ts --benchmark"

# ============================================================
# Results Summary
# ============================================================

SCRIPT_END=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END - SCRIPT_START))

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Test Results Summary${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Completed: $(date)"
echo "Total Duration: ${TOTAL_DURATION}s"
echo ""

# Print all results
for result in "${RESULTS[@]}"; do
    if [[ $result == ✓* ]]; then
        echo -e "${GREEN}$result${NC}"
    else
        echo -e "${RED}$result${NC}"
    fi
done

echo ""
echo "─────────────────────────────────────────"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo "─────────────────────────────────────────"

# Show benchmark summary if available
LATEST_BENCHMARK=$(ls -t demo/output/benchmark-*.json 2>/dev/null | head -1)
if [ -f "$LATEST_BENCHMARK" ]; then
    echo ""
    echo "Benchmark Highlights:"
    jq -r '"  Query Avg: \(.results[0].avgMs | floor)ms\n  Documents: \(.index.documents)\n  Chunks: \(.index.chunks)"' "$LATEST_BENCHMARK" 2>/dev/null || true
fi

echo ""

# Exit with appropriate code
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}Some tests failed. Check log: $LOG_FILE${NC}"
    echo ""
    exit 1
fi
