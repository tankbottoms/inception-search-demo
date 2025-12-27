# Inception ONNX - Makefile
# Quick commands for development and deployment

.PHONY: help setup clean dev start test demo docker-up docker-down benchmark pipeline ocr test-all

# Default target
help:
	@echo "Inception ONNX - Available Commands"
	@echo ""
	@echo "Setup & Development:"
	@echo "  make setup       - Full setup: install deps, convert models, validate"
	@echo "  make install     - Install all dependencies"
	@echo "  make clean       - Clean build artifacts and demo output"
	@echo "  make clean-all   - Clean everything including models"
	@echo "  make dev         - Start development server with hot reload"
	@echo "  make start       - Start production server"
	@echo ""
	@echo "Testing & Benchmarks:"
	@echo "  make test        - Run tests"
	@echo "  make check       - Check model availability"
	@echo "  make benchmark   - Run performance benchmarks"
	@echo "  make test-all    - Run complete system verification"
	@echo ""
	@echo "Demo Pipeline:"
	@echo "  make demo        - Run text extraction demo (PDF -> embed -> search)"
	@echo "  make pipeline    - Run OCR pipeline (PDF -> OCR -> embed -> search)"
	@echo "  make ocr         - Process single PDF/image with OCR"
	@echo "  make search Q=   - Search indexed documents (Q='your query')"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up   - Start Docker stack (auto-detects GPU)"
	@echo "  make docker-down - Stop Docker stack"
	@echo "  make docker-logs - View Docker logs"
	@echo ""

# Install dependencies
install:
	@echo "Installing backend dependencies..."
	@bun install
	@echo "Installing demo dependencies..."
	@cd demo && bun install
	@echo "Done!"

# Setup
setup:
	./scripts/setup.sh

# Clean
clean:
	./scripts/clean.sh

clean-all:
	./scripts/clean.sh --all

clean-models:
	./scripts/clean.sh --models

clean-demo:
	@rm -rf demo/output/*
	@echo "Demo output cleaned"

# Development
dev:
	bun run dev

start:
	bun run start

start-bg:
	@echo "Starting backend in background..."
	@bun run start > /tmp/inception.log 2>&1 &
	@sleep 3
	@curl -s http://localhost:8005/health | grep -q ok && echo "Backend started on port 8005" || echo "Failed to start backend"

stop:
	@pkill -f "bun.*src/index.ts" 2>/dev/null || true
	@echo "Backend stopped"

# Testing
test:
	bun test

check:
	bun run cli -- --check

# Benchmarks
benchmark:
	bun run cli -- --benchmark

benchmark-demo:
	@cd demo && bun run benchmark

# Demo - Text extraction pipeline
demo:
	@echo "=== PDF Text Extraction Demo ==="
	@cd demo && bun run demo

# OCR Pipeline
pipeline:
	@echo "=== Full OCR Pipeline ==="
	@cd demo && bun run pipeline

pipeline-force-ocr:
	@echo "=== Full OCR Pipeline (forced OCR) ==="
	@cd demo && bun run src/index.ts pipeline --force-ocr

# Single OCR
ocr:
	@cd demo && bun run ocr

ocr-pdf:
	@cd demo && bun run src/index.ts ocr --pdf $(PDF)

ocr-image:
	@cd demo && bun run src/index.ts ocr --image $(IMAGE)

# Search
search:
	@cd demo && bun run search "$(Q)"

# Index management
index:
	@cd demo && bun run index

index-force:
	@cd demo && bun run src/index.ts index --force

# Docker
docker-up:
	@if nvidia-smi &>/dev/null; then \
		echo "GPU detected, starting with CUDA support..."; \
		docker compose --profile gpu up -d; \
	else \
		echo "No GPU, starting CPU mode..."; \
		docker compose --profile demo up -d; \
	fi

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-demo:
	docker compose --profile demo up

# Model management
convert-models:
	cd converter && source .venv/bin/activate && python convert.py --from-registry ../models/registry.json

# Type checking
typecheck:
	bun run typecheck

# Format/lint
lint:
	bun run lint

# Complete system verification
test-all:
	@./scripts/test-all.sh
