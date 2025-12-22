#!/bin/bash
# Automated test script for the Inception Demo Docker stack
# This script builds, runs, and tests the complete demo application

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROFILE="${PROFILE:-default}"  # default, gpu, or demo
TEST_MODE="${TEST_MODE:-demo}"  # demo, single, or custom

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Inception Demo - Docker Stack Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""
echo -e "${YELLOW}Profile:${NC} $PROFILE"
echo -e "${YELLOW}Test Mode:${NC} $TEST_MODE"
echo -e ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}--- Shutting down services ---${NC}"
    docker compose --profile $PROFILE down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

# Trap script exit to run cleanup
trap cleanup EXIT

# Check for required files
echo -e "${BLUE}--- Checking prerequisites ---${NC}"

if [ ! -f "client/.env" ]; then
    echo -e "${RED}Error: client/.env file not found${NC}"
    echo -e "${YELLOW}Please create client/.env with:${NC}"
    echo "  MISTRAL_OCR_API_KEY=your-key-here"
    exit 1
fi

if [ ! -d "client/files" ]; then
    echo -e "${YELLOW}Creating client/files directory...${NC}"
    mkdir -p client/files
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}\n"

# Build services
echo -e "${BLUE}--- Building services ---${NC}"
docker compose --profile $PROFILE build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}\n"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Start services
echo -e "${BLUE}--- Starting services ---${NC}"
docker compose --profile $PROFILE up -d

echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 5

# Check service status
docker compose --profile $PROFILE ps

echo -e "${GREEN}✓ Services started${NC}\n"

# Wait for Inception service to be ready
echo -e "${BLUE}--- Waiting for Inception service ---${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker compose exec -T inception-cpu curl -f http://localhost:8005/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Inception service is ready${NC}\n"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo -e "${YELLOW}Waiting... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}✗ Inception service failed to start${NC}"
    docker compose logs inception-cpu
    exit 1
fi

# Run tests based on mode
echo -e "${BLUE}--- Running tests ---${NC}\n"

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
            echo -e "${GREEN}✓ File copied to client/files/${NC}\n"
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
    echo -e "${GREEN}  ✓ Test Complete - SUCCESS${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}  ✗ Test Failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

# Show logs summary
echo -e "\n${BLUE}--- Service Logs Summary ---${NC}"
echo -e "${YELLOW}Inception CPU logs (last 10 lines):${NC}"
docker compose logs --tail=10 inception-cpu

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Test completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
