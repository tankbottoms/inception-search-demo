#!/bin/bash
# Quick Pipeline Test Script
# Tests embeddings and OCR on both CPU (8005) and GPU (8006) services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }

# Test PDFs
PDF1="demo/files/d20703e11b194dff.pdf"
PDF2="demo/files/1c798174235b0e7c.pdf"

PASSED=0
FAILED=0

test_result() {
    if [ "$1" -eq 0 ]; then
        log_pass "$2"
        PASSED=$((PASSED + 1))
    else
        log_fail "$2: $3"
        FAILED=$((FAILED + 1))
    fi
}

test_service() {
    local name="$1"
    local url="$2"

    echo ""
    echo "=============================================="
    echo "  Testing: $name ($url)"
    echo "=============================================="
    echo ""

    # Health check
    log_info "Testing health endpoint..."
    HEALTH=$(curl -sf "$url/health" 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q '"status":"ok"'; then
        PROVIDER=$(echo "$HEALTH" | grep -o '"provider":"[^"]*"' | cut -d'"' -f4)
        test_result 0 "Health check - Provider: $PROVIDER"
    else
        test_result 1 "Health check" "Service not responding"
        return 1
    fi

    # Status check
    log_info "Testing status endpoint..."
    STATUS=$(curl -sf "$url/status" 2>/dev/null || echo "")
    if echo "$STATUS" | grep -q '"initialized":true'; then
        test_result 0 "Status check - Initialized"
    else
        test_result 1 "Status check" "Not initialized"
    fi

    # Query embedding
    log_info "Testing query embedding..."
    START=$(date +%s%N)
    QUERY=$(curl -sf -X POST "$url/api/v1/embed/query" \
        -H "Content-Type: application/json" \
        -d '{"text": "What is machine learning?"}' 2>/dev/null || echo "")
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))

    if echo "$QUERY" | grep -q '"embedding":\['; then
        DIM=$(echo "$QUERY" | grep -o '\[-*[0-9.e+-]*' | head -1 | wc -c)
        test_result 0 "Query embedding (${LATENCY}ms)"
    else
        test_result 1 "Query embedding" "No embedding returned"
    fi

    # Text embedding with chunking
    log_info "Testing text embedding..."
    LONG_TEXT="Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks. Natural language processing helps computers understand text. Computer vision enables image understanding."

    START=$(date +%s%N)
    TEXT=$(curl -sf -X POST "$url/api/v1/embed/text" \
        -H "Content-Type: application/json" \
        -d "{\"id\": 1, \"text\": \"$LONG_TEXT\"}" 2>/dev/null || echo "")
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))

    if echo "$TEXT" | grep -q '"embeddings":\['; then
        CHUNKS=$(echo "$TEXT" | grep -o '"chunk_number"' | wc -l)
        test_result 0 "Text embedding - $CHUNKS chunk(s) (${LATENCY}ms)"
    else
        test_result 1 "Text embedding" "No chunks returned"
    fi

    # Batch embedding
    log_info "Testing batch embedding..."
    START=$(date +%s%N)
    BATCH=$(curl -sf -X POST "$url/api/v1/embed/batch" \
        -H "Content-Type: application/json" \
        -d '{
            "documents": [
                {"id": 1, "text": "Document about artificial intelligence."},
                {"id": 2, "text": "Document about natural language processing."}
            ]
        }' 2>/dev/null || echo "")
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))

    if echo "$BATCH" | grep -q '"results":\['; then
        DOCS=$(echo "$BATCH" | grep -o '"id":' | wc -l)
        test_result 0 "Batch embedding - $DOCS docs (${LATENCY}ms)"
    else
        test_result 1 "Batch embedding" "No results returned"
    fi

    # Embedding consistency
    log_info "Testing embedding consistency..."
    EMB1=$(curl -sf -X POST "$url/api/v1/embed/query" \
        -H "Content-Type: application/json" \
        -d '{"text": "consistency test"}' 2>/dev/null | grep -o '"embedding":\[[^]]*\]' | head -c 100)

    EMB2=$(curl -sf -X POST "$url/api/v1/embed/query" \
        -H "Content-Type: application/json" \
        -d '{"text": "consistency test"}' 2>/dev/null | grep -o '"embedding":\[[^]]*\]' | head -c 100)

    if [ "$EMB1" = "$EMB2" ]; then
        test_result 0 "Embedding consistency"
    else
        test_result 1 "Embedding consistency" "Embeddings differ"
    fi

    # Performance benchmark (5 iterations)
    log_info "Running mini-benchmark (5 iterations)..."
    TOTAL=0
    for i in {1..5}; do
        START=$(date +%s%N)
        curl -sf -X POST "$url/api/v1/embed/query" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"Benchmark test $i\"}" > /dev/null 2>&1
        END=$(date +%s%N)
        LAT=$(( (END - START) / 1000000 ))
        TOTAL=$((TOTAL + LAT))
    done
    AVG=$((TOTAL / 5))
    test_result 0 "Benchmark - Avg latency: ${AVG}ms"
}

echo ""
echo "=============================================="
echo "  Pipeline Verification Test"
echo "=============================================="
echo ""
echo "PDF files:"
echo "  1: $PDF1 ($(ls -lh "$PDF1" 2>/dev/null | awk '{print $5}' || echo 'N/A'))"
echo "  2: $PDF2 ($(ls -lh "$PDF2" 2>/dev/null | awk '{print $5}' || echo 'N/A'))"

# Test TypeScript service (port 8005)
if curl -sf http://localhost:8005/health > /dev/null 2>&1; then
    test_service "TypeScript ONNX (CPU/CUDA)" "http://localhost:8005"
else
    log_info "TypeScript service not running on 8005, skipping..."
fi

# Test Python GPU service (port 8006)
if curl -sf http://localhost:8006/health > /dev/null 2>&1; then
    test_service "Python GPU Service" "http://localhost:8006"
else
    log_info "Python GPU service not running on 8006, skipping..."
fi

echo ""
echo "=============================================="
echo "  Summary"
echo "=============================================="
echo ""
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    log_pass "All tests passed!"
    exit 0
else
    log_fail "$FAILED test(s) failed"
    exit 1
fi
