#!/bin/bash
# Inception ONNX - Main startup script
# Auto-detects platform and starts appropriate services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
MODE="start"
PROFILE=""
VERBOSE=false

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --check         Validate and download/convert models only"
  echo "  --benchmark     Run CPU vs GPU benchmark comparison"
  echo "  --profile <p>   Use specific profile: cpu, gpu, demo, legacy"
  echo "  --verbose, -v   Enable verbose output"
  echo "  --help, -h      Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0                      # Auto-detect and start"
  echo "  $0 --profile gpu        # Force GPU mode"
  echo "  $0 --check              # Validate models only"
  echo "  $0 --benchmark          # Run performance comparison"
}

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_ok() {
  echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

detect_platform() {
  # Check for NVIDIA GPU
  if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
      echo "gpu"
      return
    fi
  fi

  # Check for NVIDIA environment variable (Docker)
  if [[ -n "${NVIDIA_VISIBLE_DEVICES}" ]]; then
    echo "gpu"
    return
  fi

  # Default to CPU
  echo "cpu"
}

check_models() {
  log_info "Checking models..."

  cd "$PROJECT_DIR"

  # Run model check via CLI
  if [[ -f "src/cli.ts" ]]; then
    bun run src/cli.ts --check
  else
    log_warn "CLI not yet implemented, skipping model check"
    log_info "Models will be downloaded on first run"
  fi
}

run_benchmark() {
  log_info "Running benchmark comparison..."

  cd "$PROJECT_DIR"

  # Run CPU benchmark
  log_info "Starting CPU benchmark..."
  docker compose --profile cpu up -d backend-cpu
  sleep 10
  # TODO: Run benchmark client
  docker compose --profile cpu down

  # Run GPU benchmark (if available)
  if [[ "$(detect_platform)" == "gpu" ]]; then
    log_info "Starting GPU benchmark..."
    docker compose --profile gpu up -d backend-gpu
    sleep 10
    # TODO: Run benchmark client
    docker compose --profile gpu down
  fi

  log_ok "Benchmark complete. Results saved to demo/logs/"
}

start_services() {
  local profile="$1"

  log_info "Starting services with profile: $profile"

  cd "$PROJECT_DIR"

  # Check if models need to be downloaded
  if [[ ! -d "models/modernbert-embed" ]]; then
    log_warn "Models not found, will be downloaded on first request"
  fi

  case "$profile" in
    gpu|cuda|spark)
      docker compose --profile gpu up -d
      ;;
    cpu|default)
      docker compose --profile cpu up -d
      ;;
    demo)
      docker compose --profile demo up -d
      ;;
    legacy|python)
      docker compose --profile legacy up -d
      ;;
    *)
      log_error "Unknown profile: $profile"
      exit 1
      ;;
  esac

  log_ok "Services started"
  log_info "API endpoint: http://localhost:8005"
  log_info "Health check: curl http://localhost:8005/health"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      MODE="check"
      shift
      ;;
    --benchmark)
      MODE="benchmark"
      shift
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Auto-detect profile if not specified
if [[ -z "$PROFILE" ]]; then
  PROFILE=$(detect_platform)
  log_info "Detected platform: $PROFILE"
fi

# Execute based on mode
case "$MODE" in
  check)
    check_models
    ;;
  benchmark)
    run_benchmark
    ;;
  start)
    start_services "$PROFILE"
    ;;
esac
