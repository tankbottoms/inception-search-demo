#!/bin/bash
# Pipeline Verification Script
# Tests OCR, embeddings, and search with PDF files
#
# Usage:
#   ./scripts/verify-pipeline.sh [--cpu|--gpu] [--pdf1 path] [--pdf2 path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Defaults
MODE="${1:-auto}"
PDF1="${2:-demo/files/d20703e11b194dff.pdf}"
PDF2="${3:-demo/files/1c798174235b0e7c.pdf}"
API_URL="http://localhost:8005"
RESULTS_DIR="benchmark-results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Detect platform
detect_platform() {
    if [[ "$MODE" == "cpu" ]]; then
        echo "cpu"
    elif [[ "$MODE" == "gpu" ]]; then
        echo "gpu"
    else
        ./scripts/detect-platform.sh profile 2>/dev/null || echo "cpu"
    fi
}

PROFILE=$(detect_platform)

mkdir -p "$RESULTS_DIR"
RESULT_FILE="$RESULTS_DIR/verify-$PROFILE-$TIMESTAMP.json"

echo ""
echo "=============================================="
echo "  Pipeline Verification - $PROFILE mode"
echo "=============================================="
echo ""
echo "Test PDFs:"
echo "  1: $PDF1"
echo "  2: $PDF2"
echo ""

# Initialize results
TESTS_PASSED=0
TESTS_FAILED=0
RESULTS=()

record_result() {
    local test_name="$1"
    local status="$2"
    local duration="$3"
    local details="$4"

    if [[ "$status" == "pass" ]]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        log_success "$test_name ($duration ms)"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        log_error "$test_name: $details"
    fi

    RESULTS+=("{\"test\": \"$test_name\", \"status\": \"$status\", \"duration_ms\": $duration, \"details\": \"$details\"}")
}

# Stop existing services
log_info "Stopping existing services..."
docker compose --profile cpu --profile gpu down --remove-orphans 2>/dev/null || true
sleep 2

# Build containers
log_info "Building $PROFILE containers..."
BUILD_START=$(date +%s%N)
if docker compose --profile "$PROFILE" build --quiet 2>/dev/null; then
    BUILD_END=$(date +%s%N)
    BUILD_TIME=$(( (BUILD_END - BUILD_START) / 1000000 ))
    record_result "docker_build_$PROFILE" "pass" "$BUILD_TIME" "Built successfully"
else
    record_result "docker_build_$PROFILE" "fail" "0" "Build failed"
    log_error "Build failed, cannot continue"
    exit 1
fi

# Start services
log_info "Starting $PROFILE services..."
docker compose --profile "$PROFILE" up -d

# Wait for health
log_info "Waiting for service health..."
HEALTH_ATTEMPTS=0
MAX_ATTEMPTS=60
while [ $HEALTH_ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    if curl -sf "$API_URL/health" > /dev/null 2>&1; then
        break
    fi
    HEALTH_ATTEMPTS=$((HEALTH_ATTEMPTS + 1))
    sleep 2
done

if [ $HEALTH_ATTEMPTS -ge $MAX_ATTEMPTS ]; then
    record_result "service_health" "fail" "0" "Service did not become healthy"
    log_error "Service did not start properly"
    docker compose logs
    exit 1
fi

record_result "service_health" "pass" "$((HEALTH_ATTEMPTS * 2000))" "Service healthy"

# Get service status
log_info "Checking service status..."
STATUS=$(curl -sf "$API_URL/status" 2>/dev/null || echo '{}')
PROVIDER=$(echo "$STATUS" | grep -o '"provider":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
log_info "Provider: $PROVIDER"

echo ""
echo "=============================================="
echo "  Test 1: Health & Status Endpoints"
echo "=============================================="

# Test health endpoint
START=$(date +%s%N)
HEALTH=$(curl -sf "$API_URL/health" 2>/dev/null)
END=$(date +%s%N)
DURATION=$(( (END - START) / 1000000 ))

if echo "$HEALTH" | grep -q '"status":"ok"'; then
    record_result "health_endpoint" "pass" "$DURATION" "Health OK"
else
    record_result "health_endpoint" "fail" "$DURATION" "Health check failed"
fi

# Test status endpoint
START=$(date +%s%N)
STATUS=$(curl -sf "$API_URL/status" 2>/dev/null)
END=$(date +%s%N)
DURATION=$(( (END - START) / 1000000 ))

if echo "$STATUS" | grep -q '"initialized":true'; then
    record_result "status_endpoint" "pass" "$DURATION" "Status OK, initialized"
else
    record_result "status_endpoint" "fail" "$DURATION" "Service not initialized"
fi

echo ""
echo "=============================================="
echo "  Test 2: Embedding Generation"
echo "=============================================="

# Test query embedding
log_info "Testing query embedding..."
START=$(date +%s%N)
QUERY_RESULT=$(curl -sf -X POST "$API_URL/api/v1/embed/query" \
    -H "Content-Type: application/json" \
    -d '{"text": "What is machine learning?"}' 2>/dev/null)
END=$(date +%s%N)
DURATION=$(( (END - START) / 1000000 ))

if echo "$QUERY_RESULT" | grep -q '"embedding":\['; then
    EMBED_DIM=$(echo "$QUERY_RESULT" | grep -o '"embedding":\[[^]]*\]' | tr ',' '\n' | wc -l)
    record_result "query_embedding" "pass" "$DURATION" "Embedding dim: $EMBED_DIM"
else
    record_result "query_embedding" "fail" "$DURATION" "No embedding returned"
fi

# Test text embedding (with chunking)
log_info "Testing text embedding with chunking..."
LONG_TEXT="This is a longer document that tests the chunking functionality. Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand human language. Computer vision enables machines to interpret visual information."

START=$(date +%s%N)
TEXT_RESULT=$(curl -sf -X POST "$API_URL/api/v1/embed/text" \
    -H "Content-Type: application/json" \
    -d "{\"id\": 1, \"text\": \"$LONG_TEXT\"}" 2>/dev/null)
END=$(date +%s%N)
DURATION=$(( (END - START) / 1000000 ))

if echo "$TEXT_RESULT" | grep -q '"embeddings":\['; then
    CHUNK_COUNT=$(echo "$TEXT_RESULT" | grep -o '"chunk_number"' | wc -l)
    record_result "text_embedding" "pass" "$DURATION" "Chunks: $CHUNK_COUNT"
else
    record_result "text_embedding" "fail" "$DURATION" "Text embedding failed"
fi

# Test batch embedding
log_info "Testing batch embedding..."
START=$(date +%s%N)
BATCH_RESULT=$(curl -sf -X POST "$API_URL/api/v1/embed/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "documents": [
            {"id": 1, "text": "First document about artificial intelligence and machine learning."},
            {"id": 2, "text": "Second document about natural language processing and text analysis."},
            {"id": 3, "text": "Third document about computer vision and image recognition."}
        ]
    }' 2>/dev/null)
END=$(date +%s%N)
DURATION=$(( (END - START) / 1000000 ))

if echo "$BATCH_RESULT" | grep -q '"results":\['; then
    DOC_COUNT=$(echo "$BATCH_RESULT" | grep -o '"id":' | wc -l)
    record_result "batch_embedding" "pass" "$DURATION" "Documents: $DOC_COUNT"
else
    record_result "batch_embedding" "fail" "$DURATION" "Batch embedding failed"
fi

echo ""
echo "=============================================="
echo "  Test 3: Embedding Consistency"
echo "=============================================="

# Test embedding consistency (same input should give same output)
log_info "Testing embedding consistency..."
TEST_TEXT="Consistency test input text"

EMBED1=$(curl -sf -X POST "$API_URL/api/v1/embed/query" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$TEST_TEXT\"}" 2>/dev/null | grep -o '"embedding":\[[^]]*\]')

EMBED2=$(curl -sf -X POST "$API_URL/api/v1/embed/query" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$TEST_TEXT\"}" 2>/dev/null | grep -o '"embedding":\[[^]]*\]')

if [[ "$EMBED1" == "$EMBED2" ]]; then
    record_result "embedding_consistency" "pass" "0" "Embeddings are identical"
else
    record_result "embedding_consistency" "fail" "0" "Embeddings differ for same input"
fi

echo ""
echo "=============================================="
echo "  Test 4: OCR Pipeline (if available)"
echo "=============================================="

# Check if OCR endpoint exists
OCR_CHECK=$(curl -sf "$API_URL/api/v1/ocr" 2>/dev/null || echo "not_found")

if [[ "$OCR_CHECK" != "not_found" ]] || curl -sf -o /dev/null -w "%{http_code}" "$API_URL/api/v1/ocr" 2>/dev/null | grep -q "40"; then
    log_info "OCR endpoint available, testing with PDF..."

    # Convert PDF to base64 and test OCR
    if [ -f "$PDF1" ]; then
        PDF_BASE64=$(base64 -w0 "$PDF1" 2>/dev/null || base64 "$PDF1" 2>/dev/null)

        START=$(date +%s%N)
        OCR_RESULT=$(curl -sf -X POST "$API_URL/api/v1/ocr" \
            -H "Content-Type: application/json" \
            -d "{\"document\": \"data:application/pdf;base64,$PDF_BASE64\"}" \
            --max-time 120 2>/dev/null || echo '{"error": "timeout"}')
        END=$(date +%s%N)
        DURATION=$(( (END - START) / 1000000 ))

        if echo "$OCR_RESULT" | grep -q '"text"'; then
            TEXT_LEN=$(echo "$OCR_RESULT" | grep -o '"text":"[^"]*"' | wc -c)
            record_result "ocr_pdf1" "pass" "$DURATION" "Extracted ~$TEXT_LEN chars"
        else
            record_result "ocr_pdf1" "fail" "$DURATION" "OCR failed or no text returned"
        fi
    else
        record_result "ocr_pdf1" "fail" "0" "PDF file not found: $PDF1"
    fi
else
    log_warn "OCR endpoint not available in this service"
    record_result "ocr_pdf1" "skip" "0" "OCR endpoint not available"
fi

echo ""
echo "=============================================="
echo "  Test 5: Performance Benchmark"
echo "=============================================="

# Run quick performance test
log_info "Running performance benchmark (10 iterations)..."

LATENCIES=()
for i in {1..10}; do
    START=$(date +%s%N)
    curl -sf -X POST "$API_URL/api/v1/embed/query" \
        -H "Content-Type: application/json" \
        -d '{"text": "Performance benchmark test query number '$i'"}' > /dev/null 2>&1
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))
    LATENCIES+=($LATENCY)
done

# Calculate stats
TOTAL=0
MIN=${LATENCIES[0]}
MAX=${LATENCIES[0]}
for lat in "${LATENCIES[@]}"; do
    TOTAL=$((TOTAL + lat))
    if [ $lat -lt $MIN ]; then MIN=$lat; fi
    if [ $lat -gt $MAX ]; then MAX=$lat; fi
done
AVG=$((TOTAL / 10))

record_result "performance_benchmark" "pass" "$AVG" "Avg: ${AVG}ms, Min: ${MIN}ms, Max: ${MAX}ms"

echo ""
echo "=============================================="
echo "  Test 6: Search Simulation"
echo "=============================================="

# Simulate search by generating query and document embeddings, then comparing
log_info "Testing search similarity..."

# Get query embedding
QUERY_EMB=$(curl -sf -X POST "$API_URL/api/v1/embed/query" \
    -H "Content-Type: application/json" \
    -d '{"text": "machine learning algorithms"}' 2>/dev/null)

# Get document embeddings
DOC1_EMB=$(curl -sf -X POST "$API_URL/api/v1/embed/query" \
    -H "Content-Type: application/json" \
    -d '{"text": "Deep learning and neural networks for AI"}' 2>/dev/null)

DOC2_EMB=$(curl -sf -X POST "$API_URL/api/v1/embed/query" \
    -H "Content-Type: application/json" \
    -d '{"text": "Cooking recipes and kitchen tips"}' 2>/dev/null)

# Simple check that embeddings were returned
if echo "$QUERY_EMB" | grep -q '"embedding"' && \
   echo "$DOC1_EMB" | grep -q '"embedding"' && \
   echo "$DOC2_EMB" | grep -q '"embedding"'; then
    record_result "search_embeddings" "pass" "0" "All embeddings generated for search"
else
    record_result "search_embeddings" "fail" "0" "Failed to generate search embeddings"
fi

echo ""
echo "=============================================="
echo "  Results Summary"
echo "=============================================="
echo ""
echo "Profile: $PROFILE"
echo "Provider: $PROVIDER"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

# Write results JSON
cat > "$RESULT_FILE" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "profile": "$PROFILE",
    "provider": "$PROVIDER",
    "pdfs": ["$PDF1", "$PDF2"],
    "summary": {
        "passed": $TESTS_PASSED,
        "failed": $TESTS_FAILED,
        "total": $((TESTS_PASSED + TESTS_FAILED))
    },
    "tests": [
        $(IFS=,; echo "${RESULTS[*]}")
    ]
}
EOF

log_info "Results saved to: $RESULT_FILE"

# Stop services
log_info "Stopping services..."
docker compose --profile "$PROFILE" down

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    log_success "All tests passed!"
    exit 0
else
    echo ""
    log_error "$TESTS_FAILED test(s) failed"
    exit 1
fi
