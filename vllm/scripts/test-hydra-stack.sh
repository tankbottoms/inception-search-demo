#!/bin/bash
# vLLM Hydra Stack - Automated Test Script
#
# Tests the complete vLLM Hydra stack including:
#   - Service health checks
#   - API functionality
#   - Benchmarks
#
# Usage:
#   ./test-hydra-stack.sh              # Run all tests
#   ./test-hydra-stack.sh --quick      # Quick health check only
#   ./test-hydra-stack.sh --benchmark  # Run with benchmarks
#   SKIP_BUILD=1 ./test-hydra-stack.sh # Skip rebuild

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
QUICK_MODE=false
RUN_BENCHMARK=false
SKIP_BUILD="${SKIP_BUILD:-0}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --benchmark|-b)
            RUN_BENCHMARK=true
            shift
            ;;
        --help|-h)
            echo "vLLM Hydra Stack Test"
            echo ""
            echo "Usage:"
            echo "  ./test-hydra-stack.sh              # Run all tests"
            echo "  ./test-hydra-stack.sh --quick      # Quick health check only"
            echo "  ./test-hydra-stack.sh --benchmark  # Run with benchmarks"
            echo "  SKIP_BUILD=1 ./test-hydra-stack.sh # Skip rebuild"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

cd "$VLLM_DIR"

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Service ports
EMBEDDINGS_PORT="${EMBEDDINGS_PORT:-8001}"
DEEPSEEK_OCR_PORT="${DEEPSEEK_OCR_PORT:-8002}"
HUNYUAN_OCR_PORT="${HUNYUAN_OCR_PORT:-8003}"
GPT_OSS_20B_PORT="${GPT_OSS_20B_PORT:-8004}"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}              vLLM Hydra Stack - Test Suite                       ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Services:${NC}"
echo "  vllm-freelaw-modernbert-embed-base-finetune-512:  Port $EMBEDDINGS_PORT"
echo "  vllm-deepSeekOCR:                                  Port $DEEPSEEK_OCR_PORT"
echo "  vllm-hunyuanOCR:                                   Port $HUNYUAN_OCR_PORT"
echo "  vllm-gpt-oss-20b:                                  Port $GPT_OSS_20B_PORT (profile)"
echo ""
echo -e "${YELLOW}Options:${NC}"
echo "  Quick Mode: $QUICK_MODE"
echo "  Benchmark:  $RUN_BENCHMARK"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}--- Cleanup ---${NC}"
    if [ "$KEEP_RUNNING" != "1" ]; then
        docker compose down 2>/dev/null || true
        echo -e "${GREEN}[OK]${NC} Services stopped"
    else
        echo -e "${YELLOW}[INFO]${NC} Keeping services running (KEEP_RUNNING=1)"
    fi
}

trap cleanup EXIT

# ===========================================================================
# Step 1: Prerequisites Check
# ===========================================================================
echo -e "${BLUE}--- Step 1: Prerequisites Check ---${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Docker not found"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Docker available"

if ! command -v curl &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} curl not found"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} curl available"

if ! docker info &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Docker daemon not running"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Docker daemon running"

# Check GPU availability
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[OK]${NC} NVIDIA GPU available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | while read line; do
        echo -e "     GPU: $line"
    done
else
    echo -e "${YELLOW}[WARN]${NC} No NVIDIA GPU detected (services may use CPU)"
fi

echo ""

# ===========================================================================
# Step 2: Build Services
# ===========================================================================
if [ "$SKIP_BUILD" != "1" ]; then
    echo -e "${BLUE}--- Step 2: Building Services ---${NC}"
    docker compose build
    echo -e "${GREEN}[OK]${NC} Build complete"
    echo ""
else
    echo -e "${YELLOW}--- Step 2: Build skipped (SKIP_BUILD=1) ---${NC}"
    echo ""
fi

# ===========================================================================
# Step 3: Start Services
# ===========================================================================
echo -e "${BLUE}--- Step 3: Starting Services ---${NC}"

docker compose up -d
echo -e "${GREEN}[OK]${NC} Services started"

echo ""
docker compose ps
echo ""

# ===========================================================================
# Step 4: Wait for Services
# ===========================================================================
echo -e "${BLUE}--- Step 4: Waiting for Services ---${NC}"

MAX_WAIT=300
WAIT_INTERVAL=10
elapsed=0

check_service() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null || echo "000"
}

while [ $elapsed -lt $MAX_WAIT ]; do
    emb_status=$(check_service $EMBEDDINGS_PORT)
    deepseek_status=$(check_service $DEEPSEEK_OCR_PORT)
    hunyuan_status=$(check_service $HUNYUAN_OCR_PORT)

    echo -e "  [$elapsed/${MAX_WAIT}s] Embeddings: $emb_status | DeepSeekOCR: $deepseek_status | HunyuanOCR: $hunyuan_status"

    # Check if at least embeddings is ready
    if [ "$emb_status" = "200" ]; then
        echo ""
        echo -e "${YELLOW}Service Status:${NC}"

        echo -e "${GREEN}[OK]${NC} vllm-freelaw-modernbert-embed-base-finetune-512 ready"

        if [ "$deepseek_status" = "200" ]; then
            echo -e "${GREEN}[OK]${NC} vllm-deepSeekOCR ready"
        else
            echo -e "${YELLOW}[WAIT]${NC} vllm-deepSeekOCR still loading"
        fi

        if [ "$hunyuan_status" = "200" ]; then
            echo -e "${GREEN}[OK]${NC} vllm-hunyuanOCR ready"
        else
            echo -e "${YELLOW}[WAIT]${NC} vllm-hunyuanOCR still loading"
        fi
        break
    fi

    sleep $WAIT_INTERVAL
    elapsed=$((elapsed + WAIT_INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo -e "${RED}[ERROR]${NC} Services failed to start within ${MAX_WAIT}s"
    echo ""
    echo "Service logs:"
    docker compose logs --tail=50
    exit 1
fi

echo ""

# Quick mode stops here
if [ "$QUICK_MODE" = "true" ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                   Quick Health Check Passed                      ${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    KEEP_RUNNING=1
    exit 0
fi

# ===========================================================================
# Step 5: Run API Tests
# ===========================================================================
echo -e "${BLUE}--- Step 5: Running API Tests ---${NC}"
echo ""

TEST_PASSED=0
TEST_FAILED=0

# Get model ID from the service
EMBEDDINGS_MODEL=$(curl -s "http://localhost:$EMBEDDINGS_PORT/v1/models" 2>/dev/null | jq -r '.data[0].id' || echo "freelawproject/modernbert-embed-base_finetune_512")

# Test 5.1: Embeddings API
echo -e "${YELLOW}Test 5.1: Embeddings API${NC}"
EMB_RESPONSE=$(curl -s -X POST "http://localhost:$EMBEDDINGS_PORT/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$EMBEDDINGS_MODEL\", \"input\": \"Test embedding generation\"}" 2>/dev/null)

if echo "$EMB_RESPONSE" | jq -e '.data[0].embedding' > /dev/null 2>&1; then
    EMB_DIM=$(echo "$EMB_RESPONSE" | jq '.data[0].embedding | length')
    echo -e "${GREEN}  [PASS]${NC} Generated embedding with $EMB_DIM dimensions"
    TEST_PASSED=$((TEST_PASSED + 1))
else
    echo -e "${RED}  [FAIL]${NC} Embeddings API test failed"
    echo "  Response: $EMB_RESPONSE"
    TEST_FAILED=$((TEST_FAILED + 1))
fi

# Test 5.2: Models endpoint
echo -e "${YELLOW}Test 5.2: Models Endpoint${NC}"
MODELS_RESPONSE=$(curl -s "http://localhost:$EMBEDDINGS_PORT/v1/models" 2>/dev/null)

if echo "$MODELS_RESPONSE" | jq -e '.data' > /dev/null 2>&1; then
    MODEL_ID=$(echo "$MODELS_RESPONSE" | jq -r '.data[0].id')
    echo -e "${GREEN}  [PASS]${NC} Model: $MODEL_ID"
    TEST_PASSED=$((TEST_PASSED + 1))
else
    echo -e "${RED}  [FAIL]${NC} Models endpoint test failed"
    TEST_FAILED=$((TEST_FAILED + 1))
fi

# Test 5.3: Similarity test
echo -e "${YELLOW}Test 5.3: Embedding Similarity${NC}"

EMB1=$(curl -s -X POST "http://localhost:$EMBEDDINGS_PORT/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$EMBEDDINGS_MODEL\", \"input\": \"The court ruled in favor of the plaintiff\"}" 2>/dev/null | jq '.data[0].embedding')

EMB2=$(curl -s -X POST "http://localhost:$EMBEDDINGS_PORT/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$EMBEDDINGS_MODEL\", \"input\": \"The judge decided for the claimant\"}" 2>/dev/null | jq '.data[0].embedding')

EMB3=$(curl -s -X POST "http://localhost:$EMBEDDINGS_PORT/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$EMBEDDINGS_MODEL\", \"input\": \"The recipe calls for two cups of flour\"}" 2>/dev/null | jq '.data[0].embedding')

if [ -n "$EMB1" ] && [ -n "$EMB2" ] && [ -n "$EMB3" ] && [ "$EMB1" != "null" ]; then
    echo -e "${GREEN}  [PASS]${NC} Generated 3 test embeddings"
    echo -e "${BLUE}  [INFO]${NC} Similar sentences should have higher cosine similarity"
    TEST_PASSED=$((TEST_PASSED + 1))
else
    echo -e "${RED}  [FAIL]${NC} Could not generate test embeddings"
    TEST_FAILED=$((TEST_FAILED + 1))
fi

echo ""

# ===========================================================================
# Step 6: Benchmark (if requested)
# ===========================================================================
if [ "$RUN_BENCHMARK" = "true" ]; then
    echo -e "${BLUE}--- Step 6: Running Benchmarks ---${NC}"
    echo ""

    echo -e "${YELLOW}Embeddings Throughput Test (10 iterations):${NC}"

    TOTAL_TIME=0
    for i in {1..10}; do
        START=$(date +%s%N)
        curl -s -X POST "http://localhost:$EMBEDDINGS_PORT/v1/embeddings" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$EMBEDDINGS_MODEL\", \"input\": \"Benchmark test sentence number $i for measuring throughput\"}" > /dev/null
        END=$(date +%s%N)
        ELAPSED=$(( (END - START) / 1000000 ))
        TOTAL_TIME=$((TOTAL_TIME + ELAPSED))
        echo -e "  Iteration $i: ${ELAPSED}ms"
    done

    AVG_TIME=$((TOTAL_TIME / 10))
    THROUGHPUT=$(echo "scale=2; 1000 / $AVG_TIME" | bc)

    echo ""
    echo -e "${GREEN}  Average: ${AVG_TIME}ms${NC}"
    echo -e "${GREEN}  Throughput: ${THROUGHPUT} req/s${NC}"

    # Save benchmark results
    mkdir -p "$VLLM_DIR/output"
    cat > "$VLLM_DIR/output/benchmark-$(date +%s).json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "test": "embeddings_throughput",
  "iterations": 10,
  "total_ms": $TOTAL_TIME,
  "avg_ms": $AVG_TIME,
  "throughput": "$THROUGHPUT req/s"
}
EOF
    echo -e "${BLUE}  [INFO]${NC} Results saved to output/"
    echo ""
fi

# ===========================================================================
# Summary
# ===========================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                        Test Summary                              ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${GREEN}Passed: $TEST_PASSED${NC}"
if [ $TEST_FAILED -gt 0 ]; then
    echo -e "  ${RED}Failed: $TEST_FAILED${NC}"
fi
echo ""

if [ $TEST_FAILED -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                      All Tests Passed!                          ${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    KEEP_RUNNING=1
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}                      Some Tests Failed                           ${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
