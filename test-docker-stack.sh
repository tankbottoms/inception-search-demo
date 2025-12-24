#!/bin/bash
# Automated test script for the Inception Demo Docker stack
# This script builds, runs, and tests the complete demo application
# Supports automatic platform detection for M1/ARM64 (CPU) or NVIDIA CUDA (GPU)

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TEST_MODE="${TEST_MODE:-demo}"  # demo, single, or custom

# ============================================================================
# Platform Detection
# ============================================================================
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
            INCEPTION_SERVICE="inception-gpu"
            PROFILE="gpu"
            return
        fi
    fi

    # macOS ARM64 (M1/M2/M3)
    if [[ "$os_type" == "Darwin" && "$arch_type" == "arm64" ]]; then
        echo -e "  Platform: ${GREEN}macOS ARM64 (Apple Silicon)${NC}"
        PLATFORM="cpu"
        INCEPTION_SERVICE="inception-cpu"
        PROFILE="default"
        return
    fi

    # Linux ARM64
    if [[ "$os_type" == "Linux" && "$arch_type" == "aarch64" ]]; then
        echo -e "  Platform: ${GREEN}Linux ARM64${NC}"
        PLATFORM="cpu"
        INCEPTION_SERVICE="inception-cpu"
        PROFILE="default"
        return
    fi

    # Default to CPU
    echo -e "  Platform: ${YELLOW}Default (CPU)${NC}"
    PLATFORM="cpu"
    INCEPTION_SERVICE="inception-cpu"
    PROFILE="default"
}

# ============================================================================
# Check if Inception service is already running
# ============================================================================
check_existing_service() {
    echo -e "\n${CYAN}[Service Check]${NC}"

    # Check if inception-cpu container is running
    if docker ps --format '{{.Names}}' | grep -q "^inception-cpu$"; then
        echo -e "  inception-cpu: ${GREEN}Running${NC}"
        INCEPTION_RUNNING="cpu"
        return 0
    fi

    # Check if inception-gpu container is running
    if docker ps --format '{{.Names}}' | grep -q "^inception-gpu$"; then
        echo -e "  inception-gpu: ${GREEN}Running${NC}"
        INCEPTION_RUNNING="gpu"
        return 0
    fi

    # Check if port 8005 is responding
    if curl -s -f http://localhost:8005/health > /dev/null 2>&1; then
        echo -e "  Port 8005: ${GREEN}Active (external service)${NC}"
        INCEPTION_RUNNING="external"
        return 0
    fi

    echo -e "  Inception service: ${YELLOW}Not running${NC}"
    INCEPTION_RUNNING=""
    return 1
}

# ============================================================================
# Main Script
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Inception Demo - Docker Stack Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""

# Detect platform
detect_platform

# Allow manual override via PROFILE env var
if [ -n "${PROFILE_OVERRIDE:-}" ]; then
    PROFILE="$PROFILE_OVERRIDE"
    if [ "$PROFILE" == "gpu" ] || [ "$PROFILE" == "cuda" ]; then
        INCEPTION_SERVICE="inception-gpu"
    else
        INCEPTION_SERVICE="inception-cpu"
    fi
    echo -e "\n${YELLOW}[Override] Using profile: $PROFILE${NC}"
fi

echo -e "\n${YELLOW}Configuration:${NC}"
echo -e "  Profile: $PROFILE"
echo -e "  Inception Service: $INCEPTION_SERVICE"
echo -e "  Test Mode: $TEST_MODE"
echo -e ""

# Check for existing service
SKIP_SERVICE_START=false
if check_existing_service; then
    echo -e "\n${GREEN}Using existing Inception service.${NC}"
    SKIP_SERVICE_START=true
fi

# Function to cleanup on exit (only if we started services)
cleanup() {
    if [ "$SKIP_SERVICE_START" = false ]; then
        echo -e "\n${YELLOW}--- Shutting down services ---${NC}"
        docker compose --profile $PROFILE down 2>/dev/null || true
        echo -e "${GREEN}[OK] Services stopped${NC}"
    fi
}

# Trap script exit to run cleanup
trap cleanup EXIT

# Check for required files
echo -e "\n${CYAN}[Prerequisites]${NC}"

if [ ! -f ".env" ]; then
    echo -e "  ${RED}Error: .env file not found${NC}"
    echo -e "  ${YELLOW}Please create .env with:${NC}"
    echo "    MISTRAL_OCR_API_KEY=your-key-here"
    echo -e "  ${YELLOW}Or copy from .env.example:${NC}"
    echo "    cp .env.example .env"
    exit 1
fi
echo -e "  .env file: ${GREEN}Found${NC}"

if [ ! -d "client/files" ]; then
    echo -e "  Creating client/files directory..."
    mkdir -p client/files
fi
echo -e "  client/files: ${GREEN}Ready${NC}"

# Create ocr output directory
if [ ! -d "client/ocr" ]; then
    mkdir -p client/ocr
fi
echo -e "  client/ocr: ${GREEN}Ready${NC}"

echo -e "  ${GREEN}[OK] Prerequisites check passed${NC}"

# Build and start services if not already running
if [ "$SKIP_SERVICE_START" = false ]; then
    # Build services
    echo -e "\n${CYAN}[Building Services]${NC}"
    docker compose --profile $PROFILE build

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}[OK] Build successful${NC}"
    else
        echo -e "  ${RED}[FAIL] Build failed${NC}"
        exit 1
    fi

    # Start services
    echo -e "\n${CYAN}[Starting Services]${NC}"
    docker compose --profile $PROFILE up -d $INCEPTION_SERVICE

    echo -e "  Waiting for services to initialize..."
    sleep 5

    # Show service status
    docker compose --profile $PROFILE ps

    echo -e "  ${GREEN}[OK] Services started${NC}"

    # Wait for Inception service to be ready
    echo -e "\n${CYAN}[Waiting for Inception Service]${NC}"
    MAX_RETRIES=30
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s -f http://localhost:8005/health > /dev/null 2>&1; then
            echo -e "  ${GREEN}[OK] Inception service is ready${NC}"
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo -e "  Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "  ${RED}[FAIL] Inception service failed to start${NC}"
        docker compose logs $INCEPTION_SERVICE
        exit 1
    fi
fi

# Run tests based on mode
echo -e "\n${CYAN}[Running Tests]${NC}"
echo -e "  Mode: $TEST_MODE"
echo -e ""

case $TEST_MODE in
    demo)
        echo -e "${GREEN}Running full demo with built-in files and query...${NC}\n"
        docker compose run --rm client demo
        ;;

    single)
        if [ -z "$TEST_FILE" ]; then
            echo -e "${RED}Error: TEST_FILE environment variable not set${NC}"
            echo -e "${YELLOW}Usage: TEST_FILE='path/to/file' TEST_QUERY='search term' ./test-docker-stack.sh${NC}"
            exit 1
        fi

        echo -e "${GREEN}Testing with file: $TEST_FILE${NC}"
        echo -e "${GREEN}Query: ${TEST_QUERY:-'securities fraud'}${NC}\n"

        # Copy file to client/files if it exists
        if [ -f "$TEST_FILE" ]; then
            cp "$TEST_FILE" client/files/
            echo -e "${GREEN}[OK] File copied to client/files/${NC}\n"
        fi

        docker compose run --rm client demo "$(basename "$TEST_FILE")" "${TEST_QUERY:-securities fraud}"
        ;;

    custom)
        if [ -z "$CUSTOM_COMMAND" ]; then
            echo -e "${RED}Error: CUSTOM_COMMAND environment variable not set${NC}"
            echo -e "${YELLOW}Usage: CUSTOM_COMMAND='index files/' ./test-docker-stack.sh${NC}"
            exit 1
        fi

        echo -e "${GREEN}Running custom command: $CUSTOM_COMMAND${NC}\n"
        docker compose run --rm client $CUSTOM_COMMAND
        ;;

    *)
        echo -e "${RED}Unknown test mode: $TEST_MODE${NC}"
        echo -e "${YELLOW}Valid modes: demo, single, custom${NC}"
        exit 1
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  [OK] Test Complete - SUCCESS${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}  [FAIL] Test Failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

# Show logs summary
echo -e "\n${CYAN}[Service Logs Summary]${NC}"
echo -e "${YELLOW}$INCEPTION_SERVICE logs (last 10 lines):${NC}"
docker compose logs --tail=10 $INCEPTION_SERVICE 2>/dev/null || echo "  (no logs available)"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Test completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
