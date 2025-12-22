#!/bin/bash
# Automated test script for the Inception Demo Docker stack
# This script builds, runs, and tests the complete demo application
#
# Usage:
#   ./tests/test-docker-stack.sh           # Default CPU mode
#   PROFILE=gpu ./tests/test-docker-stack.sh  # GPU mode
#   SKIP_MISTRAL_TEST=1 ./tests/test-docker-stack.sh  # Skip Mistral API test

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROFILE="${PROFILE:-default}"  # default (cpu), gpu, or demo
TEST_MODE="${TEST_MODE:-demo}"  # demo, single, or custom
SKIP_MISTRAL_TEST="${SKIP_MISTRAL_TEST:-0}"

# Determine the inception service name based on profile
if [ "$PROFILE" = "gpu" ] || [ "$PROFILE" = "cuda" ]; then
    INCEPTION_SERVICE="inception-gpu"
else
    INCEPTION_SERVICE="inception-cpu"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Inception Demo - Docker Stack Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""
echo -e "${YELLOW}Profile:${NC} $PROFILE"
echo -e "${YELLOW}Inception Service:${NC} $INCEPTION_SERVICE"
echo -e "${YELLOW}Test Mode:${NC} $TEST_MODE"
echo -e ""

# Change to project root directory (one level up from tests/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "${YELLOW}Working directory:${NC} $(pwd)"
echo -e ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}--- Shutting down services ---${NC}"
    docker compose --profile $PROFILE down 2>/dev/null || true
    echo -e "${GREEN}OK Services stopped${NC}"
}

# Trap script exit to run cleanup
trap cleanup EXIT

# Check for required files
echo -e "${BLUE}--- Checking prerequisites ---${NC}"

if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found in project root${NC}"
    echo -e "${YELLOW}Please create .env with:${NC}"
    echo "  MISTRAL_OCR_API_KEY=your-key-here"
    echo -e "${YELLOW}Or copy from .env.example:${NC}"
    echo "  cp .env.example .env"
    exit 1
fi

# Source .env to get MISTRAL_OCR_API_KEY
export $(grep -v '^#' .env | xargs)

if [ -z "$MISTRAL_OCR_API_KEY" ]; then
    echo -e "${RED}Error: MISTRAL_OCR_API_KEY not set in .env${NC}"
    exit 1
fi

if [ ! -d "client/files" ]; then
    echo -e "${YELLOW}Creating client/files directory...${NC}"
    mkdir -p client/files
fi

echo -e "${GREEN}OK Prerequisites OK${NC}\n"

# --- Step 1: Test Mistral OCR API ---
if [ "$SKIP_MISTRAL_TEST" != "1" ]; then
    echo -e "${BLUE}--- Step 1: Testing Mistral OCR API ---${NC}"
    echo -e "${YELLOW}This validates your API key before starting Docker services.${NC}\n"

    # Check if bun is installed
    if ! command -v bun &> /dev/null; then
        echo -e "${RED}Error: bun is not installed${NC}"
        echo -e "${YELLOW}Install bun: curl -fsSL https://bun.sh/install | bash${NC}"
        exit 1
    fi

    # Install test dependencies if needed
    if [ ! -d "client/node_modules" ]; then
        echo -e "${YELLOW}Installing client dependencies...${NC}"
        cd client && bun install && cd ..
    fi

    # Run Mistral OCR test
    if bun run tests/test-mistral-ocr.ts; then
        echo -e "${GREEN}OK Mistral OCR API test passed${NC}\n"
    else
        echo -e "${RED}FAIL Mistral OCR API test failed${NC}"
        echo -e "${YELLOW}Please check your MISTRAL_OCR_API_KEY in .env${NC}"
        echo -e "${YELLOW}You may need to regenerate your API key at: https://console.mistral.ai/${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}--- Skipping Mistral OCR API test ---${NC}\n"
fi

# --- Step 2: Build services ---
echo -e "${BLUE}--- Step 2: Building Docker services ---${NC}"
docker compose --profile $PROFILE build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}OK Build successful${NC}\n"
else
    echo -e "${RED}FAIL Build failed${NC}"
    exit 1
fi

# --- Step 3: Start services ---
echo -e "${BLUE}--- Step 3: Starting services ---${NC}"
docker compose --profile $PROFILE up -d

echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 5

# Check service status
docker compose --profile $PROFILE ps

echo -e "${GREEN}OK Services started${NC}\n"

# --- Step 4: Wait for Inception service to be ready ---
echo -e "${BLUE}--- Step 4: Waiting for $INCEPTION_SERVICE to be ready ---${NC}"
MAX_RETRIES=60  # Increased for GPU model loading
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker compose exec -T $INCEPTION_SERVICE curl -f http://localhost:8005/health > /dev/null 2>&1; then
        echo -e "${GREEN}OK $INCEPTION_SERVICE is ready${NC}\n"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo -e "${YELLOW}Waiting... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}FAIL $INCEPTION_SERVICE failed to start${NC}"
    echo -e "${YELLOW}Service logs:${NC}"
    docker compose logs $INCEPTION_SERVICE
    exit 1
fi

# --- Step 5: Run tests ---
echo -e "${BLUE}--- Step 5: Running demo tests ---${NC}\n"

case $TEST_MODE in
    demo)
        echo -e "${GREEN}Running full demo with built-in files and query...${NC}\n"
        docker compose run --rm client demo
        ;;

    single)
        if [ -z "$TEST_FILE" ]; then
            echo -e "${RED}Error: TEST_FILE environment variable not set${NC}"
            echo -e "${YELLOW}Usage: TEST_FILE='path/to/file' TEST_QUERY='search term' ./tests/test-docker-stack.sh${NC}"
            exit 1
        fi

        echo -e "${GREEN}Testing with file: $TEST_FILE${NC}"
        echo -e "${GREEN}Query: ${TEST_QUERY:-'securities fraud'}${NC}\n"

        # Copy file to client/files if it exists
        if [ -f "$TEST_FILE" ]; then
            cp "$TEST_FILE" client/files/
            echo -e "${GREEN}OK File copied to client/files/${NC}\n"
        fi

        docker compose run --rm client demo "$(basename "$TEST_FILE")" "${TEST_QUERY:-securities fraud}"
        ;;

    custom)
        if [ -z "$CUSTOM_COMMAND" ]; then
            echo -e "${RED}Error: CUSTOM_COMMAND environment variable not set${NC}"
            echo -e "${YELLOW}Usage: CUSTOM_COMMAND='index files/' ./tests/test-docker-stack.sh${NC}"
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
    echo -e "${GREEN}  OK Test Complete - SUCCESS${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}  FAIL Test Failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

# Show logs summary
echo -e "\n${BLUE}--- Service Logs Summary ---${NC}"
echo -e "${YELLOW}$INCEPTION_SERVICE logs (last 10 lines):${NC}"
docker compose logs --tail=10 $INCEPTION_SERVICE

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Test completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
