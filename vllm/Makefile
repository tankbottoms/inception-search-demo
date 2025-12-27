# vLLM Hydra Cluster - Makefile
#
# Comprehensive management commands for the vLLM Hydra multi-model inference stack.
#
# Quick Start:
#   make up           - Start core services (embeddings + OCR)
#   make up-all       - Start all services including GPT-OSS inference
#   make test         - Run all verification tests
#   make demo         - Run interactive demos
#
# Multi-Node (spark-1 + spark-2):
#   make deploy       - Deploy full stack to spark-1
#   make workers      - Start Ray workers on spark-2
#   make verify       - Verify all services across nodes

.PHONY: help up up-all up-scaled down restart logs status health \
        test test-quick verify demo demo-cot demo-pipeline benchmark stress \
        build clean install sync-spark2 deploy workers

# ============================================================================
# Configuration
# ============================================================================

COMPOSE := docker compose
CLIENT_DIR := client
SCRIPTS_DIR := scripts

# Node Configuration (override with environment variables)
SPARK1_IP ?= 192.168.1.76
SPARK1_TAILSCALE ?= 100.70.220.58
SPARK2_IP ?= 192.168.1.63
SPARK2_TAILSCALE ?= 100.87.229.92

# Colors
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
BOLD := \033[1m
NC := \033[0m

# ============================================================================
# Help
# ============================================================================

help:
	@echo ""
	@echo "$(CYAN)$(BOLD)vLLM Hydra Cluster$(NC)"
	@echo "$(CYAN)==================$(NC)"
	@echo ""
	@echo "$(BOLD)Service Management:$(NC)"
	@echo "  $(GREEN)make up$(NC)            Start core services (embeddings + OCR)"
	@echo "  $(GREEN)make up-all$(NC)        Start all services (core + GPT-OSS + LB)"
	@echo "  $(GREEN)make up-scaled$(NC)     Start with 2x embeddings + load balancer"
	@echo "  $(GREEN)make up-gpt$(NC)        Start with GPT-OSS inference"
	@echo "  $(GREEN)make down$(NC)          Stop all services"
	@echo "  $(GREEN)make restart$(NC)       Restart all services"
	@echo "  $(GREEN)make logs$(NC)          Follow service logs"
	@echo "  $(GREEN)make status$(NC)        Show container status"
	@echo ""
	@echo "$(BOLD)Testing & Verification:$(NC)"
	@echo "  $(GREEN)make health$(NC)        Quick health check of all services"
	@echo "  $(GREEN)make test$(NC)          Run full test suite (14 tests)"
	@echo "  $(GREEN)make test-quick$(NC)    Run quick verification"
	@echo "  $(GREEN)make verify$(NC)        Verify OCR, embeddings, and search"
	@echo ""
	@echo "$(BOLD)Demos:$(NC)"
	@echo "  $(GREEN)make demo$(NC)          Run embeddings similarity demo"
	@echo "  $(GREEN)make demo-cot$(NC)      Run chain-of-thought demo (4 examples)"
	@echo "  $(GREEN)make demo-cot-quick$(NC) Quick CoT demo (2 examples)"
	@echo "  $(GREEN)make demo-pipeline$(NC) Run full OCR pipeline with citations"
	@echo ""
	@echo "$(BOLD)Benchmarking:$(NC)"
	@echo "  $(GREEN)make benchmark$(NC)     Run performance benchmark"
	@echo "  $(GREEN)make stress$(NC)        Stress test embedding load balancer"
	@echo "  $(GREEN)make stress-100$(NC)    Stress test with 100 concurrent requests"
	@echo ""
	@echo "$(BOLD)Multi-Node (spark-1 + spark-2):$(NC)"
	@echo "  $(GREEN)make deploy$(NC)        Deploy full stack to spark-1"
	@echo "  $(GREEN)make workers$(NC)       Start Ray workers on spark-2"
	@echo "  $(GREEN)make sync-spark2$(NC)   Sync files to spark-2"
	@echo "  $(GREEN)make verify-cluster$(NC) Verify all nodes"
	@echo ""
	@echo "$(BOLD)Development:$(NC)"
	@echo "  $(GREEN)make install$(NC)       Install client dependencies"
	@echo "  $(GREEN)make build$(NC)         Build Docker images"
	@echo "  $(GREEN)make clean$(NC)         Clean output files and caches"
	@echo ""
	@echo "$(BOLD)Configuration:$(NC)"
	@echo "  SPARK1_IP=$(SPARK1_IP)  SPARK2_IP=$(SPARK2_IP)"
	@echo ""

# ============================================================================
# Service Management
# ============================================================================

up:
	@echo "$(CYAN)Starting core services (embeddings + OCR)...$(NC)"
	@$(COMPOSE) up -d vllm-freelaw-modernbert vllm-hunyuanOCR
	@echo "$(GREEN)Core services started. Use 'make health' to check status.$(NC)"

up-gpt:
	@echo "$(CYAN)Starting services with GPT-OSS inference...$(NC)"
	@$(COMPOSE) --profile gpt-oss up -d
	@echo "$(GREEN)Services started. GPT-OSS takes ~2-3 min to load.$(NC)"

up-scaled:
	@echo "$(CYAN)Starting with scaled embeddings (2x replicas + load balancer)...$(NC)"
	@$(COMPOSE) --profile embeddings-scaled up -d
	@echo "$(GREEN)Scaled services started. LB endpoint: http://localhost:8000$(NC)"

up-all:
	@echo "$(CYAN)Starting all services (core + GPT-OSS + scaled embeddings)...$(NC)"
	@$(COMPOSE) --profile embeddings-scaled --profile gpt-oss up -d
	@echo "$(GREEN)All services started. Use 'make health' to check status.$(NC)"

down:
	@echo "$(CYAN)Stopping all services...$(NC)"
	@$(COMPOSE) --profile embeddings-scaled --profile gpt-oss --profile llama --profile monitor down
	@echo "$(GREEN)All services stopped.$(NC)"

restart: down up-all

logs:
	@$(COMPOSE) logs -f

logs-gpt:
	@docker logs -f vllm-gpt-oss-20b

logs-ocr:
	@docker logs -f vllm-hunyuanOCR

logs-embed:
	@docker logs -f vllm-freelaw-modernbert

status:
	@echo "$(CYAN)$(BOLD)Container Status$(NC)"
	@echo "$(CYAN)=================$(NC)"
	@docker ps --filter "name=vllm-" --filter "name=embeddings-lb" --filter "name=traefik" \
		--format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20

# ============================================================================
# Health & Verification
# ============================================================================

health:
	@echo ""
	@echo "$(CYAN)$(BOLD)Service Health Check$(NC)"
	@echo "$(CYAN)====================$(NC)"
	@echo ""
	@printf "  Embeddings (8001):  " && \
		(curl -sf http://localhost:8001/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(RED)DOWN$(NC)")
	@printf "  Embeddings-2 (8002): " && \
		(curl -sf http://localhost:8002/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  Load Balancer (8000): " && \
		(curl -sf http://localhost:8000/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  OCR (8003):         " && \
		(curl -sf http://localhost:8003/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(RED)DOWN$(NC)")
	@printf "  Inference (8004):   " && \
		(curl -sf http://localhost:8004/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@echo ""

test: install
	@echo ""
	@cd $(CLIENT_DIR) && bun run verify

test-quick: install
	@echo ""
	@cd $(CLIENT_DIR) && bun run verify:embedding

verify: install
	@echo ""
	@cd $(CLIENT_DIR) && bun run verify

verify-ocr: install
	@cd $(CLIENT_DIR) && bun run verify:ocr

verify-search: install
	@cd $(CLIENT_DIR) && bun run verify:search

# ============================================================================
# Demos
# ============================================================================

demo: install
	@cd $(CLIENT_DIR) && bun run demo

demo-cot: install
	@cd $(CLIENT_DIR) && bun run cot

demo-cot-quick: install
	@cd $(CLIENT_DIR) && bun run cot:quick

demo-pipeline: install
	@cd $(CLIENT_DIR) && bun run ocr-pipeline

# ============================================================================
# Benchmarking
# ============================================================================

benchmark: install
	@cd $(CLIENT_DIR) && bun run benchmark

benchmark-full: install
	@cd $(CLIENT_DIR) && bun run benchmark --iterations 20 --requests 100 --concurrency "1,5,10,20,50"

stress: install
	@cd $(CLIENT_DIR) && bun run stress

stress-100: install
	@cd $(CLIENT_DIR) && bun run stress --concurrency 100 --duration 30

stress-sustained: install
	@cd $(CLIENT_DIR) && bun run stress --concurrency 50 --duration 60 --ramp-up 10

# ============================================================================
# Multi-Node Deployment
# ============================================================================

deploy:
	@echo "$(CYAN)Deploying to spark-1 ($(SPARK1_IP))...$(NC)"
	@./$(SCRIPTS_DIR)/hydra-start.sh
	@echo "$(GREEN)Deployment complete.$(NC)"

workers:
	@echo "$(CYAN)Starting workers on spark-2 ($(SPARK2_IP))...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd ~/Developer/vllm-hydra && docker compose up -d"
	@echo "$(GREEN)Workers started.$(NC)"

workers-down:
	@echo "$(CYAN)Stopping workers on spark-2 ($(SPARK2_IP))...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd ~/Developer/vllm-hydra && docker compose down"
	@echo "$(GREEN)Workers stopped.$(NC)"

sync-spark2:
	@echo "$(CYAN)Syncing files to spark-2...$(NC)"
	@./$(SCRIPTS_DIR)/sync-spark2.sh

pull-images-spark2:
	@echo "$(CYAN)Pulling Docker images on spark-2...$(NC)"
	@./$(SCRIPTS_DIR)/pull-images-spark2.sh

verify-cluster: health
	@echo ""
	@echo "$(CYAN)$(BOLD)Verifying spark-2 ($(SPARK2_IP))...$(NC)"
	@printf "  spark-2 embeddings (8001): " && \
		(curl -sf http://$(SPARK2_IP):8001/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(RED)DOWN$(NC)")

# ============================================================================
# Development
# ============================================================================

install:
	@cd $(CLIENT_DIR) && bun install --silent 2>/dev/null || bun install

build:
	@echo "$(CYAN)Building Docker images...$(NC)"
	@$(COMPOSE) build
	@echo "$(GREEN)Build complete.$(NC)"

clean:
	@echo "$(CYAN)Cleaning output files...$(NC)"
	@rm -rf output/*.json output/*.md output/images/
	@echo "$(GREEN)Clean complete.$(NC)"

clean-all: clean
	@echo "$(CYAN)Cleaning Docker resources...$(NC)"
	@docker system prune -f
	@echo "$(GREEN)Full clean complete.$(NC)"

# ============================================================================
# Convenience Aliases
# ============================================================================

start: up
stop: down
check: health
tests: test
demos: demo demo-cot-quick
all: up-all health test demo-cot-quick
