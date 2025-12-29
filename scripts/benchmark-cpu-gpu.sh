#!/bin/bash
# CPU vs GPU Benchmark Comparison Script
# Compares inference performance between CPU and GPU backends
#
# Usage:
#   ./scripts/benchmark-cpu-gpu.sh [options]
#
# Options:
#   --iterations N    Number of benchmark iterations (default: 10)
#   --warmup N        Number of warmup iterations (default: 3)
#   --cpu-only        Run CPU benchmark only
#   --gpu-only        Run GPU benchmark only
#   --no-build        Skip building containers
#   --output DIR      Output directory for results (default: benchmark-results)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Defaults
ITERATIONS=10
WARMUP=3
RUN_CPU=true
RUN_GPU=true
BUILD=true
OUTPUT_DIR="benchmark-results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --cpu-only)
            RUN_GPU=false
            shift
            ;;
        --gpu-only)
            RUN_CPU=false
            shift
            ;;
        --no-build)
            BUILD=false
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log_warn "jq is not installed. JSON parsing will be limited."
    fi

    # Check for GPU
    if $RUN_GPU; then
        if ! command -v nvidia-smi &> /dev/null; then
            log_warn "nvidia-smi not found. GPU benchmark will be skipped."
            RUN_GPU=false
        else
            log_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        fi
    fi
}

# Get system info
get_system_info() {
    local output_file="$1"

    cat > "$output_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "platform": "$(uname -s)",
    "arch": "$(uname -m)",
    "kernel": "$(uname -r)",
    "hostname": "$(hostname)",
    "cpuModel": "$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')",
    "cpuCores": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1),
    "memoryGB": $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0),
    "gpuAvailable": $(command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null && echo "true" || echo "false"),
    "gpuInfo": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
}
EOF
}

# Wait for service to be healthy
wait_for_health() {
    local url="$1"
    local max_attempts="${2:-60}"
    local attempt=0

    log_info "Waiting for service at $url..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            log_success "Service is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    log_error "Service did not become healthy after $max_attempts attempts"
    return 1
}

# Run single benchmark
run_benchmark() {
    local provider="$1"
    local api_url="$2"
    local output_file="$3"

    log_info "Running $provider benchmark ($ITERATIONS iterations, $WARMUP warmup)..."

    # Get service status
    local status=$(curl -sf "$api_url/status" 2>/dev/null || echo '{}')

    # Initialize results
    local results=()
    local total_time=0
    local min_time=999999
    local max_time=0

    # Sample texts of varying lengths
    local texts=(
        "Short query for testing"
        "This is a medium-length query that should test the embedding service with a bit more content to process"
        "This is a longer document that contains multiple sentences. It should test how the embedding service handles longer inputs. The service should be able to chunk this appropriately and generate embeddings for each chunk. We want to measure both throughput and latency for different input sizes."
    )

    # Warmup
    log_info "  Warmup phase ($WARMUP iterations)..."
    for ((i=1; i<=WARMUP; i++)); do
        for text in "${texts[@]}"; do
            curl -sf -X POST "$api_url/api/v1/embed/query" \
                -H "Content-Type: application/json" \
                -d "{\"text\": \"$text\"}" > /dev/null 2>&1 || true
        done
    done

    # Benchmark
    log_info "  Benchmark phase ($ITERATIONS iterations)..."
    for ((i=1; i<=ITERATIONS; i++)); do
        for idx in "${!texts[@]}"; do
            local text="${texts[$idx]}"
            local text_len=${#text}

            local start=$(date +%s%N)
            local response=$(curl -sf -X POST "$api_url/api/v1/embed/query" \
                -H "Content-Type: application/json" \
                -d "{\"text\": \"$text\"}" 2>/dev/null || echo '{}')
            local end=$(date +%s%N)

            local elapsed=$(( (end - start) / 1000000 )) # Convert to ms

            results+=("$elapsed")
            total_time=$((total_time + elapsed))

            if [ $elapsed -lt $min_time ]; then
                min_time=$elapsed
            fi
            if [ $elapsed -gt $max_time ]; then
                max_time=$elapsed
            fi
        done
        echo -ne "    Progress: $i/$ITERATIONS iterations\r"
    done
    echo ""

    # Calculate statistics
    local count=${#results[@]}
    local avg_time=$((total_time / count))

    # Sort for percentiles
    IFS=$'\n' sorted=($(sort -n <<<"${results[*]}"))
    unset IFS

    local p50_idx=$((count * 50 / 100))
    local p95_idx=$((count * 95 / 100))
    local p99_idx=$((count * 99 / 100))

    local p50=${sorted[$p50_idx]}
    local p95=${sorted[$p95_idx]}
    local p99=${sorted[$p99_idx]}

    # Calculate throughput (requests per second)
    local throughput=$(echo "scale=2; $count * 1000 / $total_time" | bc)

    # Write results
    cat > "$output_file" << EOF
{
    "provider": "$provider",
    "apiUrl": "$api_url",
    "timestamp": "$(date -Iseconds)",
    "config": {
        "iterations": $ITERATIONS,
        "warmup": $WARMUP,
        "textSamples": ${#texts[@]}
    },
    "results": {
        "totalRequests": $count,
        "totalTimeMs": $total_time,
        "avgLatencyMs": $avg_time,
        "minLatencyMs": $min_time,
        "maxLatencyMs": $max_time,
        "p50LatencyMs": $p50,
        "p95LatencyMs": $p95,
        "p99LatencyMs": $p99,
        "throughputRps": $throughput
    },
    "serviceStatus": $status
}
EOF

    log_success "$provider benchmark complete"
    log_info "  Avg latency: ${avg_time}ms | P95: ${p95}ms | Throughput: ${throughput} req/s"
}

# Run batch benchmark
run_batch_benchmark() {
    local provider="$1"
    local api_url="$2"
    local output_file="$3"
    local batch_size="${4:-10}"

    log_info "Running $provider batch benchmark (batch_size=$batch_size)..."

    # Create batch request
    local documents='['
    for ((i=0; i<batch_size; i++)); do
        if [ $i -gt 0 ]; then documents+=','; fi
        documents+="{\"id\": $i, \"text\": \"Document $i with sample text for embedding benchmark testing. This is batch document number $i.\"}"
    done
    documents+=']'

    local total_time=0
    local results=()

    for ((i=1; i<=ITERATIONS; i++)); do
        local start=$(date +%s%N)
        local response=$(curl -sf -X POST "$api_url/api/v1/embed/batch" \
            -H "Content-Type: application/json" \
            -d "{\"documents\": $documents}" 2>/dev/null || echo '{}')
        local end=$(date +%s%N)

        local elapsed=$(( (end - start) / 1000000 ))
        results+=("$elapsed")
        total_time=$((total_time + elapsed))
        echo -ne "    Progress: $i/$ITERATIONS iterations\r"
    done
    echo ""

    local count=${#results[@]}
    local avg_time=$((total_time / count))
    local docs_per_sec=$(echo "scale=2; $batch_size * $count * 1000 / $total_time" | bc)

    cat >> "$output_file" << EOF
,
    "batchBenchmark": {
        "batchSize": $batch_size,
        "iterations": $count,
        "totalTimeMs": $total_time,
        "avgBatchLatencyMs": $avg_time,
        "documentsPerSecond": $docs_per_sec
    }
}
EOF

    log_info "  Batch throughput: ${docs_per_sec} docs/s"
}

# Stop all services
stop_services() {
    log_info "Stopping any running services..."
    docker compose --profile cpu down --remove-orphans 2>/dev/null || true
    docker compose --profile gpu down --remove-orphans 2>/dev/null || true
    docker compose --profile llm-cpu down --remove-orphans 2>/dev/null || true
    docker compose --profile llm-gpu down --remove-orphans 2>/dev/null || true
    sleep 2
}

# Main benchmark flow
main() {
    echo ""
    echo "=========================================="
    echo "  CPU vs GPU Benchmark Suite"
    echo "  Timestamp: $TIMESTAMP"
    echo "=========================================="
    echo ""

    check_prerequisites

    # Get system info
    get_system_info "$OUTPUT_DIR/system-info-$TIMESTAMP.json"
    log_info "System info saved to $OUTPUT_DIR/system-info-$TIMESTAMP.json"

    # Stop any running services
    stop_services

    # CPU Benchmark
    if $RUN_CPU; then
        echo ""
        echo "=========================================="
        echo "  CPU Benchmark"
        echo "=========================================="

        if $BUILD; then
            log_info "Building CPU container..."
            docker compose --profile cpu build --quiet
        fi

        log_info "Starting CPU service..."
        docker compose --profile cpu up -d

        if wait_for_health "http://localhost:8005/health"; then
            run_benchmark "cpu" "http://localhost:8005" "$OUTPUT_DIR/cpu-benchmark-$TIMESTAMP.json"
            # Remove closing brace for batch append
            sed -i '$ d' "$OUTPUT_DIR/cpu-benchmark-$TIMESTAMP.json"
            run_batch_benchmark "cpu" "http://localhost:8005" "$OUTPUT_DIR/cpu-benchmark-$TIMESTAMP.json"
        else
            log_error "CPU service failed to start"
        fi

        docker compose --profile cpu down
    fi

    # GPU Benchmark
    if $RUN_GPU; then
        echo ""
        echo "=========================================="
        echo "  GPU Benchmark"
        echo "=========================================="

        if $BUILD; then
            log_info "Building GPU container..."
            docker compose --profile gpu build --quiet
        fi

        log_info "Starting GPU service..."
        docker compose --profile gpu up -d

        if wait_for_health "http://localhost:8005/health"; then
            run_benchmark "gpu" "http://localhost:8005" "$OUTPUT_DIR/gpu-benchmark-$TIMESTAMP.json"
            # Remove closing brace for batch append
            sed -i '$ d' "$OUTPUT_DIR/gpu-benchmark-$TIMESTAMP.json"
            run_batch_benchmark "gpu" "http://localhost:8005" "$OUTPUT_DIR/gpu-benchmark-$TIMESTAMP.json"
        else
            log_error "GPU service failed to start"
        fi

        docker compose --profile gpu down
    fi

    # Generate comparison report
    echo ""
    echo "=========================================="
    echo "  Benchmark Comparison"
    echo "=========================================="

    generate_comparison_report

    log_success "Benchmark complete!"
    log_info "Results saved to: $OUTPUT_DIR/"
}

# Generate comparison report
generate_comparison_report() {
    local cpu_file="$OUTPUT_DIR/cpu-benchmark-$TIMESTAMP.json"
    local gpu_file="$OUTPUT_DIR/gpu-benchmark-$TIMESTAMP.json"
    local report_file="$OUTPUT_DIR/comparison-$TIMESTAMP.txt"

    {
        echo "================================================"
        echo "  CPU vs GPU Benchmark Comparison Report"
        echo "  Generated: $(date)"
        echo "================================================"
        echo ""

        if [ -f "$cpu_file" ] && command -v jq &> /dev/null; then
            echo "CPU Results:"
            echo "  Avg Latency:  $(jq -r '.results.avgLatencyMs' "$cpu_file") ms"
            echo "  P95 Latency:  $(jq -r '.results.p95LatencyMs' "$cpu_file") ms"
            echo "  Throughput:   $(jq -r '.results.throughputRps' "$cpu_file") req/s"
            echo "  Batch Rate:   $(jq -r '.batchBenchmark.documentsPerSecond' "$cpu_file") docs/s"
            echo ""
        fi

        if [ -f "$gpu_file" ] && command -v jq &> /dev/null; then
            echo "GPU Results:"
            echo "  Avg Latency:  $(jq -r '.results.avgLatencyMs' "$gpu_file") ms"
            echo "  P95 Latency:  $(jq -r '.results.p95LatencyMs' "$gpu_file") ms"
            echo "  Throughput:   $(jq -r '.results.throughputRps' "$gpu_file") req/s"
            echo "  Batch Rate:   $(jq -r '.batchBenchmark.documentsPerSecond' "$gpu_file") docs/s"
            echo ""
        fi

        if [ -f "$cpu_file" ] && [ -f "$gpu_file" ] && command -v jq &> /dev/null; then
            local cpu_avg=$(jq -r '.results.avgLatencyMs' "$cpu_file")
            local gpu_avg=$(jq -r '.results.avgLatencyMs' "$gpu_file")
            local cpu_throughput=$(jq -r '.results.throughputRps' "$cpu_file")
            local gpu_throughput=$(jq -r '.results.throughputRps' "$gpu_file")

            echo "Comparison:"
            if [ "$gpu_avg" != "0" ] && [ "$gpu_avg" != "null" ]; then
                local speedup=$(echo "scale=2; $cpu_avg / $gpu_avg" | bc)
                echo "  Latency Speedup (GPU vs CPU): ${speedup}x"
            fi
            if [ "$cpu_throughput" != "0" ] && [ "$cpu_throughput" != "null" ]; then
                local throughput_ratio=$(echo "scale=2; $gpu_throughput / $cpu_throughput" | bc)
                echo "  Throughput Ratio (GPU/CPU):   ${throughput_ratio}x"
            fi
        fi

        echo ""
        echo "================================================"
    } | tee "$report_file"

    log_info "Comparison report saved to: $report_file"
}

# Run main
main "$@"
