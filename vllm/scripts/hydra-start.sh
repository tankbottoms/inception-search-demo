#!/bin/bash
# vLLM Hydra Cluster - Smart Startup Script with Loading Progress
#
# Features:
#   - Verbose model loading progress with ETA
#   - Extracts loading metrics from container logs
#   - Updates registry with measured loading times
#   - Real-time status updates during startup
#
# Usage:
#   ./hydra-start.sh                    # Start core services with progress
#   ./hydra-start.sh --quick            # Start without waiting for ready
#   ./hydra-start.sh --update-registry  # Update registry with loading times after startup
#   ./hydra-start.sh --service <name>   # Start specific service only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"
REGISTRY_FILE="$(dirname "$VLLM_DIR")/models/registry.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Options
QUICK_START=false
UPDATE_REGISTRY=false
SPECIFIC_SERVICE=""

# Service definitions with expected loading times (from registry)
declare -A SERVICE_CONTAINERS=(
    ["embeddings"]="vllm-freelaw-modernbert"
    ["ocr"]="vllm-hunyuanOCR"
    ["inference"]="vllm-gpt-oss-20b"
)

declare -A SERVICE_PORTS=(
    ["embeddings"]="${EMBEDDINGS_PORT:-8001}"
    ["ocr"]="${HUNYUAN_OCR_PORT:-8003}"
    ["inference"]="${GPT_OSS_20B_PORT:-8004}"
)

declare -A SERVICE_NAMES=(
    ["embeddings"]="ModernBERT Embeddings"
    ["ocr"]="HunyuanOCR"
    ["inference"]="GPT-OSS 20B"
)

# Expected loading times (seconds) - will be updated from registry
declare -A EXPECTED_LOAD_TIME=(
    ["embeddings"]=11
    ["ocr"]=22
    ["inference"]=195  # Model load + CUDA graph
)

declare -A EXPECTED_MEMORY=(
    ["embeddings"]=0.29
    ["ocr"]=1.90
    ["inference"]=13.7
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_START=true
            shift
            ;;
        --update-registry|-u)
            UPDATE_REGISTRY=true
            shift
            ;;
        --service|-s)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "vLLM Hydra - Smart Startup with Loading Progress"
            echo ""
            echo "Usage:"
            echo "  ./hydra-start.sh                    # Start with verbose progress"
            echo "  ./hydra-start.sh --quick            # Start without waiting"
            echo "  ./hydra-start.sh --update-registry  # Update registry after startup"
            echo "  ./hydra-start.sh --service <name>   # Start specific service"
            echo ""
            echo "Services: embeddings, ocr, inference"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Load registry data
load_registry() {
    if [ -f "$REGISTRY_FILE" ]; then
        # Parse loading times from registry using jq if available
        if command -v jq &> /dev/null; then
            local emb_time=$(jq -r '.models[] | select(.id=="modernbert-embed") | .loading.time_seconds // 11' "$REGISTRY_FILE" 2>/dev/null)
            local ocr_time=$(jq -r '.models[] | select(.id=="hunyuan-ocr") | .loading.time_seconds // 22' "$REGISTRY_FILE" 2>/dev/null)
            local inf_time=$(jq -r '.models[] | select(.id=="gpt-oss-20b") | .loading.time_seconds // 75' "$REGISTRY_FILE" 2>/dev/null)
            local inf_cuda=$(jq -r '.models[] | select(.id=="gpt-oss-20b") | .loading.cuda_graph_seconds // 120' "$REGISTRY_FILE" 2>/dev/null)

            EXPECTED_LOAD_TIME["embeddings"]=${emb_time:-11}
            EXPECTED_LOAD_TIME["ocr"]=${ocr_time:-22}
            EXPECTED_LOAD_TIME["inference"]=$((${inf_time:-75} + ${inf_cuda:-120}))

            EXPECTED_MEMORY["embeddings"]=$(jq -r '.models[] | select(.id=="modernbert-embed") | .loading.memory_gb // 0.29' "$REGISTRY_FILE" 2>/dev/null)
            EXPECTED_MEMORY["ocr"]=$(jq -r '.models[] | select(.id=="hunyuan-ocr") | .loading.memory_gb // 1.90' "$REGISTRY_FILE" 2>/dev/null)
            EXPECTED_MEMORY["inference"]=$(jq -r '.models[] | select(.id=="gpt-oss-20b") | .loading.memory_gb // 13.7' "$REGISTRY_FILE" 2>/dev/null)
        fi
    fi
}

# Format seconds to human readable
format_time() {
    local seconds=$1
    if [ "$seconds" -lt 60 ]; then
        echo "${seconds}s"
    else
        local mins=$((seconds / 60))
        local secs=$((seconds % 60))
        echo "${mins}m ${secs}s"
    fi
}

# Get loading progress from container logs
get_loading_stage() {
    local container=$1
    local log_output

    log_output=$(docker logs "$container" 2>&1 | tail -50)

    # Check for completion
    if echo "$log_output" | grep -q "Uvicorn running on"; then
        echo "ready"
        return
    fi

    # Check for CUDA graph compilation
    if echo "$log_output" | grep -q "cudagraph"; then
        echo "cuda_graph"
        return
    fi

    # Check for model loading progress
    local load_progress=$(echo "$log_output" | grep -oP "Loading safetensors checkpoint shards:\s+\K\d+" | tail -1)
    if [ -n "$load_progress" ]; then
        echo "loading:$load_progress"
        return
    fi

    # Check if loading started
    if echo "$log_output" | grep -q "Starting to load model"; then
        echo "starting"
        return
    fi

    # Check if initializing
    if echo "$log_output" | grep -q "Initializing"; then
        echo "initializing"
        return
    fi

    echo "pending"
}

# Extract actual loading time from logs
extract_loading_time() {
    local container=$1
    local time_info

    time_info=$(docker logs "$container" 2>&1 | grep -oP "Model loading took \K[\d.]+ GiB memory and [\d.]+ seconds" | tail -1)

    if [ -n "$time_info" ]; then
        local memory=$(echo "$time_info" | grep -oP "[\d.]+" | head -1)
        local seconds=$(echo "$time_info" | grep -oP "[\d.]+" | tail -1)
        echo "$memory:$seconds"
    fi
}

# Display progress bar
progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "["
    printf "%${filled}s" | tr ' ' '='
    printf ">"
    printf "%${empty}s" | tr ' ' ' '
    printf "] %3d%%" "$percent"
}

# Monitor service loading with progress
monitor_service() {
    local service=$1
    local container=${SERVICE_CONTAINERS[$service]}
    local port=${SERVICE_PORTS[$service]}
    local name=${SERVICE_NAMES[$service]}
    local expected_time=${EXPECTED_LOAD_TIME[$service]}
    local expected_mem=${EXPECTED_MEMORY[$service]}

    local start_time=$(date +%s)
    local last_stage=""

    echo -e "\n${CYAN}${BOLD}Loading: $name${NC}"
    echo -e "${DIM}  Container: $container | Port: $port${NC}"
    echo -e "${DIM}  Expected: ~$(format_time $expected_time) | Memory: ${expected_mem}GB${NC}"
    echo ""

    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local stage=$(get_loading_stage "$container")

        # Calculate progress
        local progress=0
        local status_msg=""

        case $stage in
            ready)
                progress=100
                status_msg="${GREEN}Ready${NC}"
                ;;
            cuda_graph)
                progress=85
                status_msg="${YELLOW}Compiling CUDA graphs...${NC}"
                ;;
            loading:*)
                local pct=${stage#loading:}
                progress=$((pct * 80 / 100))  # Loading is 0-80%
                status_msg="${YELLOW}Loading model: ${pct}%${NC}"
                ;;
            starting)
                progress=10
                status_msg="${YELLOW}Starting model load...${NC}"
                ;;
            initializing)
                progress=5
                status_msg="${YELLOW}Initializing engine...${NC}"
                ;;
            *)
                progress=$((elapsed * 5 / expected_time))  # Estimate based on time
                [ $progress -gt 5 ] && progress=5
                status_msg="${YELLOW}Starting...${NC}"
                ;;
        esac

        # Calculate ETA
        local eta=""
        if [ $progress -gt 0 ] && [ $progress -lt 100 ]; then
            local remaining=$((expected_time - elapsed))
            [ $remaining -lt 0 ] && remaining=0
            eta="ETA: ~$(format_time $remaining)"
        fi

        # Print status line (overwrite previous)
        printf "\r  "
        progress_bar $progress 100
        printf " %s" "$status_msg"
        [ -n "$eta" ] && printf " ${DIM}%s${NC}" "$eta"
        printf "     "  # Clear any trailing chars

        if [ "$stage" = "ready" ]; then
            echo ""

            # Extract actual loading time
            local actual=$(extract_loading_time "$container")
            if [ -n "$actual" ]; then
                local actual_mem=${actual%:*}
                local actual_time=${actual#*:}
                echo -e "  ${GREEN}Loaded in ${actual_time}s using ${actual_mem}GB memory${NC}"
            else
                echo -e "  ${GREEN}Loaded in ${elapsed}s${NC}"
            fi
            return 0
        fi

        # Timeout check (3x expected time)
        local timeout=$((expected_time * 3))
        if [ $elapsed -gt $timeout ]; then
            echo ""
            echo -e "  ${YELLOW}Still loading after $(format_time $elapsed)...${NC}"
            echo -e "  ${DIM}Check logs: docker logs $container${NC}"
            return 1
        fi

        sleep 2
    done
}

# Update registry with actual loading times
update_registry_times() {
    echo -e "\n${BLUE}Updating registry with actual loading times...${NC}"

    for service in embeddings ocr inference; do
        local container=${SERVICE_CONTAINERS[$service]}

        # Check if container exists
        if ! docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
            continue
        fi

        local actual=$(extract_loading_time "$container")
        if [ -n "$actual" ]; then
            local actual_mem=${actual%:*}
            local actual_time=${actual#*:}
            echo -e "  ${SERVICE_NAMES[$service]}: ${actual_time}s, ${actual_mem}GB"
        fi
    done

    echo -e "\n${DIM}To update registry.json manually, use the values above.${NC}"
}

# Main startup
main() {
    echo -e "${BLUE}${BOLD}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                  vLLM Hydra - Smart Startup                      "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"

    cd "$VLLM_DIR"

    # Load environment
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
        echo -e "${GREEN}[OK]${NC} Loaded environment"
    fi

    # Load registry data
    load_registry
    echo -e "${GREEN}[OK]${NC} Loaded model registry"

    # Calculate total expected time
    local total_time=0
    for service in embeddings ocr inference; do
        if docker compose config --services 2>/dev/null | grep -q "${SERVICE_CONTAINERS[$service]}"; then
            total_time=$((total_time + EXPECTED_LOAD_TIME[$service]))
        fi
    done

    echo ""
    echo -e "${CYAN}Estimated total startup time: ~$(format_time $total_time)${NC}"
    echo ""

    # Start containers
    echo -e "${BLUE}[INFO]${NC} Starting containers..."
    if [ -n "$SPECIFIC_SERVICE" ]; then
        docker compose up -d "${SERVICE_CONTAINERS[$SPECIFIC_SERVICE]}" 2>/dev/null || \
            docker compose up -d "$SPECIFIC_SERVICE" 2>/dev/null
    else
        docker compose up -d
    fi
    echo -e "${GREEN}[OK]${NC} Containers started"

    if [ "$QUICK_START" = "true" ]; then
        echo -e "\n${YELLOW}Quick start - not waiting for models to load${NC}"
        echo "Monitor with: docker compose logs -f"
        exit 0
    fi

    # Monitor each service
    echo -e "\n${BLUE}${BOLD}━━━ Model Loading Progress ━━━${NC}"

    local all_ready=true

    # Monitor services in order of expected load time
    for service in embeddings ocr inference; do
        local container=${SERVICE_CONTAINERS[$service]}

        # Check if container is running
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            if ! monitor_service "$service"; then
                all_ready=false
            fi
        fi
    done

    # Final status
    echo -e "\n${BLUE}${BOLD}━━━ Final Status ━━━${NC}\n"

    for service in embeddings ocr inference; do
        local container=${SERVICE_CONTAINERS[$service]}
        local port=${SERVICE_PORTS[$service]}
        local name=${SERVICE_NAMES[$service]}

        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            local health=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null || echo "000")
            if [ "$health" = "200" ]; then
                echo -e "  ${GREEN}[OK]${NC} $name (port $port)"
            else
                echo -e "  ${YELLOW}[LOADING]${NC} $name (port $port)"
            fi
        fi
    done

    echo -e "\n${BLUE}${BOLD}━━━ API Endpoints ━━━${NC}\n"
    echo "  Embeddings:  http://localhost:${SERVICE_PORTS[embeddings]}/v1"
    echo "  OCR:         http://localhost:${SERVICE_PORTS[ocr]}/v1"
    echo "  Inference:   http://localhost:${SERVICE_PORTS[inference]}/v1"
    echo ""

    if [ "$UPDATE_REGISTRY" = "true" ]; then
        update_registry_times
    fi

    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                     vLLM Hydra Started                          ${NC}"
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

main "$@"
