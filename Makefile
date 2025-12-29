# Inception ONNX - Makefile
# Quick commands for development and deployment
# Auto-detects platform (Apple Silicon, x86, NVIDIA GPU)

.PHONY: help build up down services logs benchmark benchmark-extended test clean

# Detect platform
PROFILE := $(shell ./scripts/detect-platform.sh profile 2>/dev/null || echo "cpu")
PLATFORM := $(shell ./scripts/detect-platform.sh platform 2>/dev/null || echo "unknown")

# ============================================================
# Main Commands (Auto-Detecting)
# ============================================================

help:
	@echo "Inception ONNX - Simplified Commands"
	@echo ""
	@echo "Detected Platform: $(PLATFORM) (profile: $(PROFILE))"
	@echo ""
	@echo "Primary Commands:"
	@echo "  make build              - Build containers for detected platform"
	@echo "  make up / make services - Start services (auto-detects CPU/GPU)"
	@echo "  make down               - Stop all services"
	@echo "  make logs               - View service logs"
	@echo "  make benchmark          - Run performance benchmark"
	@echo "  make benchmark-extended - Run extended benchmark with test files"
	@echo ""
	@echo "Development:"
	@echo "  make dev                - Start dev server (no Docker)"
	@echo "  make test               - Run tests"
	@echo "  make setup              - Full setup (deps, models, validation)"
	@echo ""
	@echo "Force Specific Platform:"
	@echo "  make build-cpu          - Force CPU build"
	@echo "  make build-gpu          - Force GPU build"
	@echo "  make up-cpu             - Force CPU services"
	@echo "  make up-gpu             - Force GPU services"
	@echo ""

# ============================================================
# Auto-Detecting Commands
# ============================================================

# Build for detected platform
build:
	@echo "Building for platform: $(PLATFORM) (profile: $(PROFILE))"
	docker compose --profile $(PROFILE) build

# Start services for detected platform
up: services
services:
	@echo "Starting services for platform: $(PLATFORM) (profile: $(PROFILE))"
	docker compose --profile $(PROFILE) up -d
	@echo ""
	@echo "Waiting for health check..."
	@sleep 5
	@curl -sf http://localhost:8005/health && echo "Service ready at http://localhost:8005" || echo "Service starting..."

# Stop all services
down:
	docker compose --profile cpu --profile gpu --profile llm-cpu --profile llm-gpu down --remove-orphans

# View logs
logs:
	docker compose logs -f

# ============================================================
# Benchmarks
# ============================================================

benchmark:
	@echo "Running benchmark for platform: $(PLATFORM)"
	@if [ "$(PROFILE)" = "gpu" ]; then \
		./scripts/benchmark-cpu-gpu.sh --gpu-only --iterations 10; \
	else \
		./scripts/benchmark-cpu-gpu.sh --cpu-only --iterations 10; \
	fi

benchmark-extended:
	@echo "Running extended benchmark (CPU vs GPU comparison)"
	./scripts/benchmark-cpu-gpu.sh --iterations 50 --warmup 10

benchmark-quick:
	@echo "Running quick benchmark"
	@if [ "$(PROFILE)" = "gpu" ]; then \
		./scripts/benchmark-cpu-gpu.sh --gpu-only --iterations 5 --warmup 2; \
	else \
		./scripts/benchmark-cpu-gpu.sh --cpu-only --iterations 5 --warmup 2; \
	fi

# ============================================================
# Force Specific Platform
# ============================================================

build-cpu:
	docker compose --profile cpu build

build-gpu:
	docker compose --profile gpu build

up-cpu:
	docker compose --profile cpu up -d

up-gpu:
	docker compose --profile gpu up -d

# ============================================================
# Development (No Docker)
# ============================================================

dev:
	bun run dev

start:
	bun run start

test:
	bun test

typecheck:
	bun run typecheck

# ============================================================
# Setup & Installation
# ============================================================

setup:
	./scripts/setup.sh

install:
	@echo "Installing dependencies..."
	@bun install
	@if [ -d demo ]; then cd demo && bun install; fi
	@echo "Done!"

clean:
	./scripts/clean.sh

clean-all:
	./scripts/clean.sh --all

# ============================================================
# Demo & Pipeline
# ============================================================

demo:
	@echo "=== Running Demo ==="
	@cd demo && bun run demo

pipeline:
	@echo "=== Full OCR Pipeline ==="
	@cd demo && bun run pipeline

ocr:
	@cd demo && bun run ocr

search:
	@cd demo && bun run search "$(Q)"

# ============================================================
# Model Management
# ============================================================

convert-models:
	@echo "Converting models..."
	@cd converter && source .venv/bin/activate && python convert.py --from-registry ../models/registry.json

check-models:
	bun run cli -- --check

# ============================================================
# LLM Model Server (Python Backend)
# ============================================================

llm-up:
	@if [ "$(PROFILE)" = "gpu" ]; then \
		docker compose --profile llm-gpu up -d; \
	else \
		docker compose --profile llm-cpu up -d; \
	fi

llm-down:
	docker compose --profile llm-cpu --profile llm-gpu down

llm-logs:
	docker compose logs -f llm-model-server-cpu llm-model-server-gpu 2>/dev/null || true

# ============================================================
# Status & Info
# ============================================================

status:
	@echo "=== Platform Detection ==="
	@./scripts/detect-platform.sh all
	@echo ""
	@echo "=== Docker Services ==="
	@docker compose ps 2>/dev/null || echo "No services running"
	@echo ""
	@echo "=== Service Health ==="
	@curl -sf http://localhost:8005/health 2>/dev/null && echo "Inception API: OK (port 8005)" || echo "Inception API: Not running"
	@curl -sf http://localhost:8006/health 2>/dev/null && echo "LLM Server:    OK (port 8006)" || echo "LLM Server:    Not running"

info: status
