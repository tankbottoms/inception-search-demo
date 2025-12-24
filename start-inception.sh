#!/bin/bash
# Start the appropriate Inception service based on detected platform
# Usage: ./start-inception.sh [--gpu|--cpu]

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check for manual override
FORCE_MODE=""
if [ "$1" == "--gpu" ] || [ "$1" == "--cuda" ]; then
    FORCE_MODE="gpu"
elif [ "$1" == "--cpu" ]; then
    FORCE_MODE="cpu"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Inception Service Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect platform
detect_platform() {
    local os_type=$(uname -s)
    local arch_type=$(uname -m)

    echo -e "${CYAN}[Platform Detection]${NC}"
    echo -e "  OS: $os_type"
    echo -e "  Architecture: $arch_type"

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo -e "  NVIDIA GPU: ${GREEN}Detected${NC}"
            PLATFORM="gpu"
            SERVICE="inception-gpu"
            PROFILE="gpu"
            return
        fi
    fi

    # macOS ARM64 (M1/M2/M3)
    if [[ "$os_type" == "Darwin" && "$arch_type" == "arm64" ]]; then
        echo -e "  Platform: ${GREEN}macOS ARM64 (Apple Silicon)${NC}"
        PLATFORM="cpu"
        SERVICE="inception-cpu"
        PROFILE="default"
        return
    fi

    # Linux ARM64
    if [[ "$os_type" == "Linux" && "$arch_type" == "aarch64" ]]; then
        echo -e "  Platform: ${GREEN}Linux ARM64${NC}"
        PLATFORM="cpu"
        SERVICE="inception-cpu"
        PROFILE="default"
        return
    fi

    # Default to CPU
    echo -e "  Platform: ${YELLOW}Default (CPU)${NC}"
    PLATFORM="cpu"
    SERVICE="inception-cpu"
    PROFILE="default"
}

# Check if already running
check_running() {
    if docker ps --format '{{.Names}}' | grep -q "^inception-cpu$"; then
        echo -e "\n${GREEN}inception-cpu is already running.${NC}"
        echo -e "  URL: http://localhost:8005"
        return 0
    fi

    if docker ps --format '{{.Names}}' | grep -q "^inception-gpu$"; then
        echo -e "\n${GREEN}inception-gpu is already running.${NC}"
        echo -e "  URL: http://localhost:8005"
        return 0
    fi

    return 1
}

# Detect platform
detect_platform

# Apply manual override
if [ -n "$FORCE_MODE" ]; then
    echo -e "\n${YELLOW}[Override] Forcing $FORCE_MODE mode${NC}"
    if [ "$FORCE_MODE" == "gpu" ]; then
        SERVICE="inception-gpu"
        PROFILE="gpu"
    else
        SERVICE="inception-cpu"
        PROFILE="default"
    fi
fi

echo -e "\n${YELLOW}Configuration:${NC}"
echo -e "  Service: $SERVICE"
echo -e "  Profile: $PROFILE"

# Check if already running
if check_running; then
    echo -e "\n${YELLOW}To stop: docker compose down${NC}"
    exit 0
fi

# Start service
echo -e "\n${CYAN}[Starting Service]${NC}"
docker compose --profile $PROFILE up -d $SERVICE

echo -e "\n${CYAN}[Waiting for Service]${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f http://localhost:8005/health > /dev/null 2>&1; then
        echo -e "\n${GREEN}[OK] Inception service is ready!${NC}"
        echo -e "\n${YELLOW}Service URL:${NC} http://localhost:8005"
        echo -e "${YELLOW}Health Check:${NC} http://localhost:8005/health"
        echo -e "${YELLOW}Metrics:${NC} http://localhost:8005/metrics"
        echo -e "\n${YELLOW}To stop: docker compose down${NC}"
        exit 0
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo -e "  Waiting for model to load... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

echo -e "\n${RED}[FAIL] Service failed to start${NC}"
docker compose logs $SERVICE
exit 1
