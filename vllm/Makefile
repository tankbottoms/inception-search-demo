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

.PHONY: help up up-all up-scaled down restart logs status health health-spark2 health-all \
        test test-quick test-parallel test-parallel-8 test-sequential verify demo demo-cot demo-pipeline benchmark benchmark-full benchmark-extended stability stability-5 stress \
        build clean install sync-spark2 deploy workers workers-down \
        spark2-up spark2-up-gpt spark2-up-ocr spark2-up-embed spark2-up-embed-scaled spark2-down spark2-status spark2-logs \
        spark2-logs-embed spark2-logs-embed2 spark2-logs-gpt spark2-logs-ocr pull-images-spark2 verify-cluster

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

# Colors (Solarized Dark compatible)
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
WHITE := \033[1;37m
MAGENTA := \033[0;35m
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
	@echo "  $(GREEN)make health$(NC)        Health check spark-1 services"
	@echo "  $(GREEN)make health-spark2$(NC) Health check spark-2 services"
	@echo "  $(GREEN)make health-all$(NC)    Health check all nodes"
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
	@echo "  $(GREEN)make benchmark$(NC)          Run performance benchmark"
	@echo "  $(GREEN)make benchmark-full$(NC)     Extended benchmark (20 iters, 100 reqs)"
	@echo "  $(GREEN)make benchmark-extended$(NC) Full pipeline with demo/files PDFs"
	@echo "  $(GREEN)make stability$(NC)          OCR+embed+citation stability test (1x)"
	@echo "  $(GREEN)make stability-5$(NC)        OCR+embed+citation stability test (5x)"
	@echo "  $(GREEN)make stress$(NC)             Stress test embedding load balancer"
	@echo "  $(GREEN)make stress-100$(NC)         Stress test with 100 concurrent requests"
	@echo ""
	@echo "$(BOLD)Multi-Node (spark-1 + spark-2):$(NC)"
	@echo "  $(GREEN)make deploy$(NC)              Deploy full stack to spark-1"
	@echo "  $(GREEN)make spark2-up$(NC)           Start core services on spark-2 (embed+OCR)"
	@echo "  $(GREEN)make spark2-up-gpt$(NC)       Start GPT-OSS on spark-2 (optional)"
	@echo "  $(GREEN)make spark2-up-ocr$(NC)       Restart OCR on spark-2"
	@echo "  $(GREEN)make spark2-up-embed$(NC)     Restart embeddings on spark-2"
	@echo "  $(GREEN)make spark2-up-embed-scaled$(NC) Start 2x embeddings on spark-2"
	@echo "  $(GREEN)make spark2-down$(NC)         Stop all services on spark-2"
	@echo "  $(GREEN)make spark2-status$(NC)       Show spark-2 container status"
	@echo "  $(GREEN)make spark2-logs$(NC)         View spark-2 logs"
	@echo "  $(GREEN)make sync-spark2$(NC)         Sync config files to spark-2"
	@echo "  $(GREEN)make verify-cluster$(NC)      Health check all nodes"
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
	@echo ""
	@echo "$(CYAN)$(BOLD)Spark-1 (localhost) Services$(NC)"
	@echo "$(CYAN)=============================$(NC)"
	@docker ps --filter "name=vllm-freelaw-modernbert" --filter "name=vllm-hunyuanOCR" \
		--filter "name=vllm-gpt-oss" --filter "name=vllm-llama" --filter "name=inception-services-lb" \
		--format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep -v "worker" || echo "  No services running"
	@echo ""
	@echo "$(CYAN)$(BOLD)Spark-2 ($(SPARK2_IP)) Services$(NC)"
	@echo "$(CYAN)=============================$(NC)"
	@ssh rooot@$(SPARK2_IP) "docker ps --filter 'name=vllm-' --format 'table {{.Names}}\t{{.Status}}'" 2>/dev/null || echo "  $(YELLOW)Cannot connect to spark-2$(NC)"
	@echo ""

# ============================================================================
# Health & Verification
# ============================================================================

health:
	@echo ""
	@echo "$(CYAN)$(BOLD)Spark-1 (localhost) Health Check$(NC)"
	@echo "$(CYAN)===================================$(NC)"
	@echo ""
	@printf "  Embeddings (8001):   " && \
		(curl -sf http://localhost:8001/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(RED)DOWN$(NC)")
	@printf "  Embeddings-2 (8002): " && \
		(curl -sf http://localhost:8002/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  OCR (8003):          " && \
		(curl -sf http://localhost:8003/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(RED)DOWN$(NC)")
	@printf "  Inference (8004):    " && \
		(curl -sf http://localhost:8004/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@echo ""
	@echo "$(CYAN)$(BOLD)Load Balancers (Traefik)$(NC)"
	@echo "$(CYAN)========================$(NC)"
	@echo ""
	@printf "  Embeddings LB (8000):" && \
		(curl -sf http://localhost:8000/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  OCR LB (8010):       " && \
		(curl -sf http://localhost:8010/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  Inference LB (8020): " && \
		(curl -sf http://localhost:8020/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@echo ""

health-spark2:
	@echo ""
	@echo "$(CYAN)$(BOLD)Spark-2 ($(SPARK2_IP)) Health Check$(NC)"
	@echo "$(CYAN)===================================$(NC)"
	@echo ""
	@printf "  Embeddings (8001):   " && \
		(curl -sf --connect-timeout 2 http://$(SPARK2_IP):8001/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  Embeddings-2 (8002): " && \
		(curl -sf --connect-timeout 2 http://$(SPARK2_IP):8002/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  OCR (8003):          " && \
		(curl -sf --connect-timeout 2 http://$(SPARK2_IP):8003/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  Inference (8004):    " && \
		(curl -sf --connect-timeout 2 http://$(SPARK2_IP):8004/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@echo ""

health-all: health health-spark2
	@echo "$(CYAN)$(BOLD)Cluster Summary$(NC)"
	@echo "$(CYAN)===============$(NC)"
	@echo "  Spark-1: $(SPARK1_IP) (localhost)"
	@echo "  Spark-2: $(SPARK2_IP)"
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

test-parallel: install
	@echo "$(CYAN)Running parallel OCR test (4 images)...$(NC)"
	@cd $(CLIENT_DIR) && bun run parallel-ocr

test-parallel-8: install
	@echo "$(CYAN)Running parallel OCR test (8 images)...$(NC)"
	@cd $(CLIENT_DIR) && bun run parallel-ocr:8

test-sequential: install
	@echo "$(CYAN)Running sequential OCR test (comparison)...$(NC)"
	@cd $(CLIENT_DIR) && bun run parallel-ocr:seq

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

benchmark-extended: install
	@echo "$(CYAN)Running extended benchmark with demo/files PDFs...$(NC)"
	@cd $(CLIENT_DIR) && bun run benchmark:extended

stability: install
	@echo "$(CYAN)Running stability test (1 iteration)...$(NC)"
	@cd $(CLIENT_DIR) && bun run stability

stability-5: install
	@echo "$(CYAN)Running stability test (5 iterations)...$(NC)"
	@cd $(CLIENT_DIR) && bun run stability:5

benchmark-comprehensive: install
	@echo "$(CYAN)Running comprehensive benchmark (2x forward + 2x backward)...$(NC)"
	@cd $(CLIENT_DIR) && bun run benchmark:comprehensive

stress: install
	@cd $(CLIENT_DIR) && bun run stress

stress-100: install
	@cd $(CLIENT_DIR) && bun run stress --concurrency 100 --duration 30

stress-sustained: install
	@cd $(CLIENT_DIR) && bun run stress --concurrency 50 --duration 60 --ramp-up 10

# ============================================================================
# Multi-Node Deployment
# ============================================================================

# Spark-2 remote directory (synced via sync-spark2.sh)
SPARK2_DIR := /home/rooot/Docker/vllm-hydra

deploy:
	@echo "$(CYAN)Deploying to spark-1 ($(SPARK1_IP))...$(NC)"
	@./$(SCRIPTS_DIR)/hydra-start.sh
	@echo "$(GREEN)Deployment complete.$(NC)"

# --- Spark-2 Service Management ---
# Default: embeddings + OCR start with `docker compose up -d`
# Optional: GPT-OSS via --profile vllm-gpt-oss-20b (disabled by default, conflicts with OCR)

spark2-up:
	@echo "$(CYAN)Starting core services on spark-2 ($(SPARK2_IP))...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd $(SPARK2_DIR) && docker compose up -d"
	@echo "$(GREEN)Spark-2 services starting (embeddings + OCR).$(NC)"

spark2-up-gpt:
	@echo "$(CYAN)Starting GPT-OSS on spark-2 ($(SPARK2_IP))...$(NC)"
	@echo "$(YELLOW)WARNING: GPT-OSS conflicts with OCR on same GPU. Run spark-2-down first.$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd $(SPARK2_DIR) && docker compose --profile vllm-gpt-oss-20b up -d"
	@echo "$(GREEN)GPT-OSS starting (takes ~3-5 min to load).$(NC)"

spark2-up-ocr:
	@echo "$(CYAN)Restarting OCR on spark-2 ($(SPARK2_IP))...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd $(SPARK2_DIR) && docker compose up -d vllm-hunyuanOCR"
	@echo "$(GREEN)OCR starting (takes ~2-3 min to load).$(NC)"

spark2-up-embed:
	@echo "$(CYAN)Starting embeddings on spark-2 ($(SPARK2_IP))...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd $(SPARK2_DIR) && docker compose up -d vllm-freelaw-modernbert"
	@echo "$(GREEN)Embeddings started.$(NC)"

spark2-up-embed-scaled:
	@echo "$(CYAN)Starting scaled embeddings on spark-2 ($(SPARK2_IP))...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd $(SPARK2_DIR) && docker compose --profile embeddings-scaled up -d"
	@echo "$(GREEN)Scaled embeddings (2x replicas) started.$(NC)"

spark2-down:
	@echo "$(CYAN)Stopping all services on spark-2 ($(SPARK2_IP))...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd $(SPARK2_DIR) && docker compose --profile vllm-gpt-oss-20b --profile embeddings-scaled down"
	@echo "$(GREEN)Spark-2 services stopped.$(NC)"

spark2-status:
	@echo ""
	@echo "$(CYAN)$(BOLD)Spark-2 Container Status$(NC)"
	@echo "$(CYAN)=========================$(NC)"
	@ssh rooot@$(SPARK2_IP) "docker ps --filter 'name=vllm-' --format 'table {{.Names}}\t{{.Status}}'" 2>/dev/null || echo "$(RED)Cannot connect to spark-2$(NC)"
	@echo ""

spark2-logs:
	@echo "$(CYAN)Fetching logs from spark-2...$(NC)"
	@ssh rooot@$(SPARK2_IP) "cd $(SPARK2_DIR) && docker compose logs --tail 50"

spark2-logs-embed:
	@ssh rooot@$(SPARK2_IP) "docker logs --tail 100 vllm-freelaw-modernbert" 2>/dev/null || echo "$(YELLOW)Container not running$(NC)"

spark2-logs-embed2:
	@ssh rooot@$(SPARK2_IP) "docker logs --tail 100 vllm-freelaw-modernbert-2" 2>/dev/null || echo "$(YELLOW)Container not running$(NC)"

spark2-logs-gpt:
	@ssh rooot@$(SPARK2_IP) "docker logs --tail 100 vllm-gpt-oss-20b" 2>/dev/null || echo "$(YELLOW)Container not running$(NC)"

spark2-logs-ocr:
	@ssh rooot@$(SPARK2_IP) "docker logs --tail 100 vllm-hunyuanOCR" 2>/dev/null || echo "$(YELLOW)Container not running$(NC)"

# Aliases for backwards compatibility
workers: spark2-up
workers-down: spark2-down

sync-spark2:
	@echo "$(CYAN)Syncing files to spark-2...$(NC)"
	@./$(SCRIPTS_DIR)/sync-spark2.sh

pull-images-spark2:
	@echo "$(CYAN)Pulling Docker images on spark-2...$(NC)"
	@./$(SCRIPTS_DIR)/pull-images-spark2.sh

verify-cluster: health-all
	@echo "$(GREEN)Cluster verification complete.$(NC)"

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
