#!/bin/bash
# vLLM Hydra - Service Verification Script
#
# Tests all vLLM service endpoints to verify they are working correctly
#
# Usage:
#   ./verify-services.sh              # Test all services
#   ./verify-services.sh --quick      # Quick health check only
#   ./verify-services.sh --spark2     # Also test spark-2 services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
QUICK_MODE=false
TEST_SPARK2=false
SPARK2_IP="${SPARK2_IP:-192.168.100.11}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --spark2|-s)
            TEST_SPARK2=true
            shift
            ;;
        --help|-h)
            echo "vLLM Hydra - Service Verification"
            echo ""
            echo "Usage:"
            echo "  ./verify-services.sh              # Test all services"
            echo "  ./verify-services.sh --quick      # Quick health check only"
            echo "  ./verify-services.sh --spark2     # Also test spark-2 services"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Load environment
cd "$VLLM_DIR"
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Service endpoints
EMBEDDINGS_PORT="${EMBEDDINGS_PORT:-8001}"
DEEPSEEK_OCR_PORT="${DEEPSEEK_OCR_PORT:-8002}"
HUNYUAN_OCR_PORT="${HUNYUAN_OCR_PORT:-8003}"
GPT_OSS_20B_PORT="${GPT_OSS_20B_PORT:-8004}"

# Spark-2 ports
EMBEDDINGS_RAY_PORT="${EMBEDDINGS_RAY_PORT:-8011}"
DEEPSEEK_OCR_RAY_PORT="${DEEPSEEK_OCR_RAY_PORT:-8012}"
HUNYUAN_OCR_RAY_PORT="${HUNYUAN_OCR_RAY_PORT:-8013}"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}              vLLM Hydra - Service Verification                   ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ===========================================================================
# Health Check Function
# ===========================================================================
check_health() {
    local name=$1
    local port=$2
    local host=${3:-localhost}

    local status=$(curl -s -o /dev/null -w "%{http_code}" "http://$host:$port/health" 2>/dev/null || echo "000")

    if [ "$status" = "200" ]; then
        echo -e "${GREEN}[PASS]${NC} $name - Health OK (http://$host:$port)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}[FAIL]${NC} $name - Health check failed (status: $status)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ===========================================================================
# Models Endpoint Test
# ===========================================================================
check_models() {
    local name=$1
    local port=$2
    local host=${3:-localhost}

    local response=$(curl -s "http://$host:$port/v1/models" 2>/dev/null)

    if echo "$response" | jq -e '.data[0].id' > /dev/null 2>&1; then
        local model_id=$(echo "$response" | jq -r '.data[0].id')
        echo -e "${GREEN}[PASS]${NC} $name - Models endpoint OK (model: $model_id)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}[FAIL]${NC} $name - Models endpoint failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ===========================================================================
# Embeddings API Test
# ===========================================================================
test_embeddings() {
    local name=$1
    local port=$2
    local host=${3:-localhost}

    echo -e "${CYAN}Testing $name embeddings API...${NC}"

    # Get actual model ID from service
    local model_id=$(curl -s "http://$host:$port/v1/models" 2>/dev/null | jq -r '.data[0].id' || echo "")

    local response=$(curl -s -X POST "http://$host:$port/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model_id\", \"input\": \"Test embedding generation for vLLM Hydra verification\"}" 2>/dev/null)

    if echo "$response" | jq -e '.data[0].embedding' > /dev/null 2>&1; then
        local dim=$(echo "$response" | jq '.data[0].embedding | length')
        echo -e "${GREEN}[PASS]${NC} $name - Embeddings API OK (dimensions: $dim)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}[FAIL]${NC} $name - Embeddings API failed"
        echo "  Response: $(echo "$response" | head -c 200)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ===========================================================================
# Chat Completion Test (for inference models)
# ===========================================================================
test_chat() {
    local name=$1
    local port=$2
    local host=${3:-localhost}

    echo -e "${CYAN}Testing $name chat API...${NC}"

    # Get actual model ID from service
    local model_id=$(curl -s "http://$host:$port/v1/models" 2>/dev/null | jq -r '.data[0].id' || echo "")

    local response=$(curl -s -X POST "http://$host:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model_id\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word.\"}],
            \"max_tokens\": 10,
            \"temperature\": 0.1
        }" 2>/dev/null)

    if echo "$response" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        local content=$(echo "$response" | jq -r '.choices[0].message.content' | head -c 50)
        echo -e "${GREEN}[PASS]${NC} $name - Chat API OK (response: \"$content\")"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}[FAIL]${NC} $name - Chat API failed"
        echo "  Response: $(echo "$response" | head -c 200)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ===========================================================================
# Spark-1 Service Tests
# ===========================================================================
echo -e "${YELLOW}━━━ Spark-1 Services ━━━${NC}"
echo ""

# Test vllm-freelaw-modernbert-embed-base-finetune-512
echo -e "${BLUE}[1/4] vllm-freelaw-modernbert-embed-base-finetune-512 (Embeddings)${NC}"
if check_health "vllm-freelaw-modernbert-embed-base-finetune-512" "$EMBEDDINGS_PORT"; then
    check_models "vllm-freelaw-modernbert-embed-base-finetune-512" "$EMBEDDINGS_PORT"
    if [ "$QUICK_MODE" = "false" ]; then
        test_embeddings "vllm-freelaw-modernbert-embed-base-finetune-512" "$EMBEDDINGS_PORT"
    fi
else
    TESTS_SKIPPED=$((TESTS_SKIPPED + 2))
fi
echo ""

# Test vllm-deepSeekOCR
echo -e "${BLUE}[2/4] vllm-deepSeekOCR (DeepSeek OCR)${NC}"
if check_health "vllm-deepSeekOCR" "$DEEPSEEK_OCR_PORT"; then
    check_models "vllm-deepSeekOCR" "$DEEPSEEK_OCR_PORT"
else
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
fi
echo ""

# Test vllm-hunyuanOCR
echo -e "${BLUE}[3/4] vllm-hunyuanOCR (HunyuanOCR)${NC}"
if check_health "vllm-hunyuanOCR" "$HUNYUAN_OCR_PORT"; then
    check_models "vllm-hunyuanOCR" "$HUNYUAN_OCR_PORT"
else
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
fi
echo ""

# Test vllm-gpt-oss-20b (optional profile)
echo -e "${BLUE}[4/4] vllm-gpt-oss-20b (GPT-OSS 20B - optional)${NC}"
if check_health "vllm-gpt-oss-20b" "$GPT_OSS_20B_PORT" 2>/dev/null; then
    check_models "vllm-gpt-oss-20b" "$GPT_OSS_20B_PORT"
    if [ "$QUICK_MODE" = "false" ]; then
        test_chat "vllm-gpt-oss-20b" "$GPT_OSS_20B_PORT"
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} vllm-gpt-oss-20b - Not running (use --profile gpt-oss-20b)"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
fi
echo ""

# ===========================================================================
# Spark-2 Service Tests (if requested)
# ===========================================================================
if [ "$TEST_SPARK2" = "true" ]; then
    echo -e "${YELLOW}━━━ Spark-2 Services ━━━${NC}"
    echo ""

    # Test embeddings ray worker
    echo -e "${BLUE}[S2-1] vllm-freelaw-modernbert-embed-base-finetune-512-ray-worker${NC}"
    if check_health "embeddings-ray-worker" "$EMBEDDINGS_RAY_PORT" "$SPARK2_IP" 2>/dev/null; then
        check_models "embeddings-ray-worker" "$EMBEDDINGS_RAY_PORT" "$SPARK2_IP"
    else
        echo -e "${YELLOW}[SKIP]${NC} embeddings-ray-worker - Not running"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi
    echo ""

    # Test deepseek ocr ray worker
    echo -e "${BLUE}[S2-2] vllm-deepSeekOCR-ray-worker${NC}"
    if check_health "deepseek-ocr-ray-worker" "$DEEPSEEK_OCR_RAY_PORT" "$SPARK2_IP" 2>/dev/null; then
        check_models "deepseek-ocr-ray-worker" "$DEEPSEEK_OCR_RAY_PORT" "$SPARK2_IP"
    else
        echo -e "${YELLOW}[SKIP]${NC} deepseek-ocr-ray-worker - Not running"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi
    echo ""

    # Test hunyuan ocr ray worker
    echo -e "${BLUE}[S2-3] vllm-hunyuanOCR-ray-worker${NC}"
    if check_health "hunyuan-ocr-ray-worker" "$HUNYUAN_OCR_RAY_PORT" "$SPARK2_IP" 2>/dev/null; then
        check_models "hunyuan-ocr-ray-worker" "$HUNYUAN_OCR_RAY_PORT" "$SPARK2_IP"
    else
        echo -e "${YELLOW}[SKIP]${NC} hunyuan-ocr-ray-worker - Not running"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi
    echo ""
fi

# ===========================================================================
# Summary
# ===========================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                        Test Summary                              ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${GREEN}Passed:  $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "  ${RED}Failed:  $TESTS_FAILED${NC}"
fi
if [ $TESTS_SKIPPED -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped: $TESTS_SKIPPED${NC}"
fi
echo ""

TOTAL=$((TESTS_PASSED + TESTS_FAILED))
if [ $TESTS_FAILED -eq 0 ] && [ $TESTS_PASSED -gt 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                    All Tests Passed!                            ${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
elif [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}                    Some Tests Failed                             ${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
else
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}                    No Tests Run                                 ${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
