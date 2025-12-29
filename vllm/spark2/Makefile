# vLLM Hydra Cluster - Spark-2 Worker Makefile
#
# This Makefile is deployed to spark-2 (192.168.1.63) and provides
# commands for managing vLLM worker services locally on that node.
#
# Primary control should be from spark-1 using:
#   make spark2-up         (from spark-1)
#   make spark2-status     (from spark-1)
#   make health-spark2     (from spark-1)
#
# Local Commands (run on spark-2 directly):
#   make up           - Start embeddings worker (default)
#   make up-gpt       - Start GPT-OSS worker
#   make up-ocr       - Start OCR worker
#   make down         - Stop all workers
#   make status       - Show container status
#   make health       - Health check services
#   make logs         - View logs

.PHONY: help up up-gpt up-ocr down status health logs logs-embed logs-gpt logs-ocr clean

# ============================================================================
# Configuration
# ============================================================================

COMPOSE := docker compose
COMPOSE_FILE := docker-compose.spark2.yml

# Spark-1 (primary) configuration
SPARK1_IP ?= 192.168.1.76

# Colors (Solarized Dark compatible)
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
WHITE := \033[1;37m
BOLD := \033[1m
NC := \033[0m

# ============================================================================
# Help
# ============================================================================

help:
	@echo ""
	@echo "$(CYAN)$(BOLD)vLLM Hydra - Spark-2 Worker$(NC)"
	@echo "$(CYAN)===========================$(NC)"
	@echo ""
	@echo "$(BOLD)Service Management:$(NC)"
	@echo "  $(GREEN)make up$(NC)        Start embeddings worker (default)"
	@echo "  $(GREEN)make up-gpt$(NC)    Start GPT-OSS inference worker"
	@echo "  $(GREEN)make up-ocr$(NC)    Start OCR worker"
	@echo "  $(GREEN)make down$(NC)      Stop all workers"
	@echo "  $(GREEN)make status$(NC)    Show container status"
	@echo "  $(GREEN)make health$(NC)    Health check all services"
	@echo ""
	@echo "$(BOLD)Logs:$(NC)"
	@echo "  $(GREEN)make logs$(NC)       View all logs"
	@echo "  $(GREEN)make logs-embed$(NC) View embeddings logs"
	@echo "  $(GREEN)make logs-gpt$(NC)   View GPT-OSS logs"
	@echo "  $(GREEN)make logs-ocr$(NC)   View OCR logs"
	@echo ""
	@echo "$(BOLD)Notes:$(NC)"
	@echo "  - GPT-OSS and OCR cannot run simultaneously (CUDA conflict)"
	@echo "  - Embeddings can run with either GPT-OSS or OCR"
	@echo "  - Primary control from spark-1: make spark2-up, etc."
	@echo ""
	@echo "$(BOLD)Spark-1 Primary:$(NC) $(SPARK1_IP)"
	@echo ""

# ============================================================================
# Service Management
# ============================================================================

up:
	@echo "$(CYAN)Starting embeddings worker...$(NC)"
	@$(COMPOSE) -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)Embeddings worker started.$(NC)"

up-gpt:
	@echo "$(CYAN)Starting GPT-OSS worker (takes ~3-5 min to load)...$(NC)"
	@$(COMPOSE) -f $(COMPOSE_FILE) --profile vllm-gpt-oss-20b up -d
	@echo "$(GREEN)GPT-OSS worker starting. Run 'make health' to check status.$(NC)"

up-ocr:
	@echo "$(CYAN)Starting OCR worker (takes ~2-3 min to load)...$(NC)"
	@$(COMPOSE) -f $(COMPOSE_FILE) --profile vllm-hunyuanOCR up -d
	@echo "$(GREEN)OCR worker starting. Run 'make health' to check status.$(NC)"

down:
	@echo "$(CYAN)Stopping all workers...$(NC)"
	@$(COMPOSE) -f $(COMPOSE_FILE) --profile vllm-gpt-oss-20b --profile vllm-hunyuanOCR down
	@echo "$(GREEN)All workers stopped.$(NC)"

restart: down up

# ============================================================================
# Status & Health
# ============================================================================

status:
	@echo ""
	@echo "$(CYAN)$(BOLD)Spark-2 Container Status$(NC)"
	@echo "$(CYAN)=========================$(NC)"
	@docker ps --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}"
	@echo ""

health:
	@echo ""
	@echo "$(CYAN)$(BOLD)Spark-2 Health Check$(NC)"
	@echo "$(CYAN)====================$(NC)"
	@echo ""
	@printf "  Embeddings (8001): " && \
		(curl -sf http://localhost:8001/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  OCR (8003):        " && \
		(curl -sf http://localhost:8003/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@printf "  Inference (8004):  " && \
		(curl -sf http://localhost:8004/health > /dev/null && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)--$(NC)")
	@echo ""

# ============================================================================
# Logs
# ============================================================================

logs:
	@$(COMPOSE) -f $(COMPOSE_FILE) logs --tail 50

logs-embed:
	@docker logs --tail 100 vllm-freelaw-modernbert-worker 2>/dev/null || echo "$(YELLOW)Container not running$(NC)"

logs-gpt:
	@docker logs --tail 100 vllm-gpt-oss-20b-worker 2>/dev/null || echo "$(YELLOW)Container not running$(NC)"

logs-ocr:
	@docker logs --tail 100 vllm-hunyuanOCR-worker 2>/dev/null || echo "$(YELLOW)Container not running$(NC)"

logs-follow:
	@$(COMPOSE) -f $(COMPOSE_FILE) logs -f

# ============================================================================
# Utilities
# ============================================================================

clean:
	@echo "$(CYAN)Cleaning Docker resources...$(NC)"
	@docker system prune -f
	@echo "$(GREEN)Clean complete.$(NC)"

pull:
	@echo "$(CYAN)Pulling latest images...$(NC)"
	@docker pull vllm/vllm-openai:nightly-aarch64
	@docker pull nvcr.io/nvidia/vllm:25.12-py3
	@echo "$(GREEN)Images updated.$(NC)"
