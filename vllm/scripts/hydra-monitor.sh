#!/bin/bash
# vLLM Hydra Monitor - Health monitoring and auto-restart for vLLM services
#
# Usage:
#   ./hydra-monitor.sh              # Interactive one-time check
#   ./hydra-monitor.sh --watch      # Continuous monitoring
#   ./hydra-monitor.sh --cron       # Cron mode (silent unless error)
#   ./hydra-monitor.sh --monitor    # Install crontab entry
#   ./hydra-monitor.sh --status     # Show detailed service status
#
# Environment variables:
#   MONITOR_INTERVAL      - Check interval in seconds (default: 60)
#   MONITOR_AUTO_RESTART  - Auto-restart failed services (default: true)
#   EMBEDDINGS_PORT       - Embeddings service port (default: 8001)
#   OCR_PORT              - OCR service port (default: 8002)
#   INFERENCE_PORT        - Inference service port (default: 8003)
#   HOST_IP               - Host IP for health checks (default: localhost)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")"
LOCK_FILE="/tmp/vllm-hydra-monitor.lock"
MANUAL_SHUTDOWN_FILE="/tmp/vllm-hydra-manual-shutdown"

# Load environment
if [ -f "$VLLM_DIR/.env" ]; then
    export $(grep -v '^#' "$VLLM_DIR/.env" | xargs)
fi

# Default configuration
EMBEDDINGS_PORT="${EMBEDDINGS_PORT:-8001}"
OCR_PORT="${OCR_PORT:-8002}"
INFERENCE_PORT="${INFERENCE_PORT:-8003}"
HOST_IP="${HOST_IP:-localhost}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
MONITOR_AUTO_RESTART="${MONITOR_AUTO_RESTART:-true}"

# Mode flags
CRON_MODE=false
WATCH_MODE=false
INSTALL_CRON=false
STATUS_MODE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cron)
            CRON_MODE=true
            shift
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --monitor)
            INSTALL_CRON=true
            shift
            ;;
        --status)
            STATUS_MODE=true
            shift
            ;;
        --help|-h)
            echo "vLLM Hydra Monitor"
            echo ""
            echo "Usage:"
            echo "  ./hydra-monitor.sh              # Interactive one-time check"
            echo "  ./hydra-monitor.sh --watch      # Continuous monitoring"
            echo "  ./hydra-monitor.sh --cron       # Cron mode (silent unless error)"
            echo "  ./hydra-monitor.sh --monitor    # Install crontab entry"
            echo "  ./hydra-monitor.sh --status     # Show detailed service status"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Check if manual shutdown was triggered
is_manual_shutdown() {
    [ -f "$MANUAL_SHUTDOWN_FILE" ]
}

# Check single service health
check_service_health() {
    local service_name=$1
    local port=$2
    local timeout=${3:-5}

    local health_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout "$timeout" \
        "http://${HOST_IP}:${port}/health" 2>/dev/null || echo "000")

    echo "$health_code"
}

# Get container status
get_container_status() {
    local container_name=$1
    docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not_found"
}

# Get container health
get_container_health() {
    local container_name=$1
    docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "unknown"
}

# Show detailed status
show_status() {
    echo -e "${BLUE}━━━ vLLM Hydra Service Status ━━━${NC}"
    echo ""

    local services=("vllm-freelaw:$EMBEDDINGS_PORT:Embeddings"
                   "vllm-deepseekOCR:$OCR_PORT:OCR"
                   "vllm-inference:$INFERENCE_PORT:Inference")

    for service_info in "${services[@]}"; do
        IFS=':' read -r container port name <<< "$service_info"

        local status=$(get_container_status "$container")
        local health=$(get_container_health "$container")
        local api_health=$(check_service_health "$name" "$port")

        echo -e "${YELLOW}$name ($container)${NC}"
        echo -e "  Container: $status"
        echo -e "  Health: $health"
        echo -e "  API (port $port): $api_health"

        if [ "$status" = "running" ] && [ "$api_health" = "200" ]; then
            echo -e "  ${GREEN}[OK]${NC}"
        else
            echo -e "  ${RED}[UNHEALTHY]${NC}"
        fi
        echo ""
    done

    # Check monitor status
    local monitor_status=$(get_container_status "hydra-monitor")
    echo -e "${YELLOW}Monitor (hydra-monitor)${NC}"
    echo -e "  Container: $monitor_status"
    if [ "$monitor_status" = "running" ]; then
        echo -e "  ${GREEN}[OK]${NC}"
    else
        echo -e "  ${BLUE}[NOT RUNNING]${NC} (optional)"
    fi
    echo ""
}

# Check all services
check_all_services() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local all_healthy=true
    local unhealthy_services=()

    # Check embeddings service
    local embeddings_health=$(check_service_health "embeddings" "$EMBEDDINGS_PORT")
    if [ "$embeddings_health" != "200" ]; then
        all_healthy=false
        unhealthy_services+=("vllm-freelaw (port $EMBEDDINGS_PORT): $embeddings_health")
    fi

    # Check OCR service
    local ocr_health=$(check_service_health "ocr" "$OCR_PORT")
    if [ "$ocr_health" != "200" ]; then
        all_healthy=false
        unhealthy_services+=("vllm-deepseekOCR (port $OCR_PORT): $ocr_health")
    fi

    # Check inference service
    local inference_health=$(check_service_health "inference" "$INFERENCE_PORT")
    if [ "$inference_health" != "200" ]; then
        all_healthy=false
        unhealthy_services+=("vllm-inference (port $INFERENCE_PORT): $inference_health")
    fi

    if [ "$CRON_MODE" = "true" ]; then
        # Cron mode: only output on errors
        if [ "$all_healthy" = "false" ]; then
            # Check if manual shutdown was triggered
            if is_manual_shutdown; then
                echo "[$timestamp] [INFO] Manual shutdown detected, skipping restart"
                return 0
            fi

            echo "[$timestamp] [ERROR] Unhealthy services detected:"
            for service in "${unhealthy_services[@]}"; do
                echo "  - $service"
            done

            if [ "$MONITOR_AUTO_RESTART" = "true" ]; then
                echo "[$timestamp] [INFO] Attempting restart..."
                cd "$VLLM_DIR" && docker compose up -d
            fi
        fi
    else
        # Interactive mode: always output status
        echo "[$timestamp] vLLM Hydra Health Check"
        echo ""
        echo "  vllm-freelaw (Embeddings, port $EMBEDDINGS_PORT): $embeddings_health"
        echo "  vllm-deepseekOCR (OCR, port $OCR_PORT): $ocr_health"
        echo "  vllm-inference (Inference, port $INFERENCE_PORT): $inference_health"
        echo ""

        if [ "$all_healthy" = "true" ]; then
            echo -e "  ${GREEN}[OK] All services healthy${NC}"
        else
            echo -e "  ${RED}[WARN] Some services unhealthy${NC}"
            for service in "${unhealthy_services[@]}"; do
                echo -e "    ${RED}- $service${NC}"
            done
        fi
    fi

    return $([ "$all_healthy" = "true" ] && echo 0 || echo 1)
}

# Install crontab entry
install_cron() {
    local cron_entry="* * * * * $SCRIPT_DIR/hydra-monitor.sh --cron >> /var/log/vllm-hydra/monitor.log 2>&1"

    echo -e "${BLUE}Installing crontab entry for vLLM Hydra monitor...${NC}"
    echo ""

    # Check if entry already exists
    if crontab -l 2>/dev/null | grep -q "hydra-monitor.sh"; then
        echo -e "${YELLOW}Crontab entry already exists. Updating...${NC}"
        crontab -l 2>/dev/null | grep -v "hydra-monitor.sh" | crontab -
    fi

    # Add new entry
    (crontab -l 2>/dev/null; echo "$cron_entry") | crontab -

    echo -e "${GREEN}Crontab entry installed:${NC}"
    echo "  $cron_entry"
    echo ""
    echo -e "${BLUE}To remove the entry, run:${NC}"
    echo "  crontab -l | grep -v hydra-monitor.sh | crontab -"
    echo ""
    echo -e "${BLUE}To view logs:${NC}"
    echo "  tail -f /var/log/vllm-hydra/monitor.log"

    # Create log directory if needed
    sudo mkdir -p /var/log/vllm-hydra
    sudo chown $(whoami) /var/log/vllm-hydra
}

# Main execution
if [ "$INSTALL_CRON" = "true" ]; then
    install_cron
    exit 0
fi

if [ "$STATUS_MODE" = "true" ]; then
    show_status
    exit 0
fi

if [ "$WATCH_MODE" = "true" ]; then
    echo -e "${BLUE}[INFO] Continuous monitoring started (interval: ${MONITOR_INTERVAL}s)${NC}"
    echo -e "${BLUE}[INFO] Press Ctrl+C to stop${NC}"
    echo ""
    while true; do
        check_all_services
        echo ""
        sleep "$MONITOR_INTERVAL"
    done
else
    check_all_services
fi
