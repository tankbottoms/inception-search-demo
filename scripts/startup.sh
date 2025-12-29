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
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Defaults
MODE="start"
PROFILE=""
VERBOSE=false
SHOW_INFO=false

usage() {
  echo -e "${BOLD}Inception ONNX Startup Script${NC}"
  echo ""
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --info, -i      Show detailed platform information"
  echo "  --check         Validate and download/convert models only"
  echo "  --benchmark     Run CPU vs GPU benchmark comparison"
  echo "  --profile <p>   Use specific profile: cpu, gpu, demo, legacy"
  echo "  --verbose, -v   Enable verbose output"
  echo "  --help, -h      Show this help message"
  echo ""
  echo "Profiles:"
  echo "  cpu       TypeScript/ONNX CPU backend (port 8005)"
  echo "  gpu       Python/PyTorch GPU backend (port 8006)"
  echo "  demo      Run demo client with indexing"
  echo "  legacy    Python legacy backend"
  echo ""
  echo "Examples:"
  echo "  $0                      # Auto-detect and start"
  echo "  $0 --info               # Show platform info"
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

log_section() {
  echo ""
  echo -e "${BOLD}${CYAN}━━━ $1 ━━━${NC}"
  echo ""
}

# ============================================================
# Platform Detection and Information
# ============================================================

get_cpu_info() {
  local cpu_model=""
  local cpu_cores=""
  local cpu_threads=""
  local cpu_freq=""
  local cpu_arch=""

  # Get architecture
  cpu_arch=$(uname -m)

  # Get CPU model
  if [[ -f /proc/cpuinfo ]]; then
    cpu_model=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || echo "Unknown")
    if [[ -z "$cpu_model" || "$cpu_model" == "Unknown" ]]; then
      # ARM doesn't have "model name", try Hardware
      cpu_model=$(grep -m1 "Hardware" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || echo "Unknown")
    fi
    if [[ -z "$cpu_model" || "$cpu_model" == "Unknown" ]]; then
      # Try CPU implementer for ARM
      local implementer=$(grep -m1 "CPU implementer" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//')
      local part=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//')
      if [[ -n "$implementer" ]]; then
        case "$implementer" in
          "0x41") cpu_model="ARM" ;;
          "0x4e") cpu_model="NVIDIA" ;;
          *) cpu_model="Unknown ($implementer)" ;;
        esac
        [[ -n "$part" ]] && cpu_model="$cpu_model (part $part)"
      fi
    fi
  fi

  # Get core count
  cpu_cores=$(nproc 2>/dev/null || echo "Unknown")

  # Get thread count (different from cores on SMT systems)
  if [[ -f /proc/cpuinfo ]]; then
    cpu_threads=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo "$cpu_cores")
  else
    cpu_threads="$cpu_cores"
  fi

  # Get CPU frequency
  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq ]]; then
    local freq_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)
    if [[ -n "$freq_khz" ]]; then
      cpu_freq=$(echo "scale=2; $freq_khz / 1000000" | bc 2>/dev/null || echo "Unknown")
      cpu_freq="${cpu_freq} GHz"
    fi
  elif command -v lscpu &>/dev/null; then
    cpu_freq=$(lscpu 2>/dev/null | grep "CPU max MHz" | awk '{print $4}' | head -1)
    if [[ -n "$cpu_freq" ]]; then
      cpu_freq=$(echo "scale=2; $cpu_freq / 1000" | bc 2>/dev/null || echo "Unknown")
      cpu_freq="${cpu_freq} GHz"
    fi
  fi

  echo "CPU_MODEL='$cpu_model'"
  echo "CPU_ARCH='$cpu_arch'"
  echo "CPU_CORES='$cpu_cores'"
  echo "CPU_THREADS='$cpu_threads'"
  echo "CPU_FREQ='${cpu_freq:-Unknown}'"
}

get_memory_info() {
  local total_mem=""
  local free_mem=""
  local available_mem=""

  if [[ -f /proc/meminfo ]]; then
    total_mem=$(grep "MemTotal" /proc/meminfo | awk '{print $2}')
    free_mem=$(grep "MemFree" /proc/meminfo | awk '{print $2}')
    available_mem=$(grep "MemAvailable" /proc/meminfo | awk '{print $2}')

    # Convert to GB
    total_mem=$(echo "scale=1; $total_mem / 1048576" | bc 2>/dev/null || echo "Unknown")
    free_mem=$(echo "scale=1; $free_mem / 1048576" | bc 2>/dev/null || echo "Unknown")
    available_mem=$(echo "scale=1; $available_mem / 1048576" | bc 2>/dev/null || echo "Unknown")
  fi

  echo "MEM_TOTAL='${total_mem:-Unknown} GB'"
  echo "MEM_FREE='${free_mem:-Unknown} GB'"
  echo "MEM_AVAILABLE='${available_mem:-Unknown} GB'"
}

get_gpu_info() {
  local gpu_available=false
  local gpu_name=""
  local gpu_memory=""
  local gpu_memory_free=""
  local gpu_driver=""
  local cuda_version=""
  local compute_cap=""

  if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
      gpu_available=true

      # Get GPU info
      gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
      gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "Unknown")
      gpu_memory_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "Unknown")
      gpu_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")

      # Get CUDA version
      if command -v nvcc &>/dev/null; then
        cuda_version=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' || echo "Unknown")
      else
        # Try to get from nvidia-smi
        cuda_version=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' || echo "Unknown")
      fi

      # Get compute capability
      compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    fi
  fi

  echo "GPU_AVAILABLE='$gpu_available'"
  echo "GPU_NAME='$gpu_name'"
  echo "GPU_MEMORY='${gpu_memory:-0} MB'"
  echo "GPU_MEMORY_FREE='${gpu_memory_free:-0} MB'"
  echo "GPU_DRIVER='$gpu_driver'"
  echo "CUDA_VERSION='$cuda_version'"
  echo "COMPUTE_CAP='$compute_cap'"
}

get_os_info() {
  local os_name=""
  local os_version=""
  local kernel=""

  # Get kernel version
  kernel=$(uname -r)

  # Get OS info
  if [[ -f /etc/os-release ]]; then
    os_name=$(grep "^NAME=" /etc/os-release | cut -d= -f2 | tr -d '"')
    os_version=$(grep "^VERSION=" /etc/os-release | cut -d= -f2 | tr -d '"')
  elif [[ -f /etc/lsb-release ]]; then
    os_name=$(grep "DISTRIB_ID" /etc/lsb-release | cut -d= -f2)
    os_version=$(grep "DISTRIB_RELEASE" /etc/lsb-release | cut -d= -f2)
  else
    os_name=$(uname -s)
    os_version=$(uname -v)
  fi

  echo "OS_NAME='$os_name'"
  echo "OS_VERSION='$os_version'"
  echo "KERNEL='$kernel'"
}

get_docker_info() {
  local docker_available=false
  local docker_version=""
  local compose_version=""
  local nvidia_runtime=false

  if command -v docker &>/dev/null; then
    docker_available=true
    docker_version=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',' || echo "Unknown")

    # Check for compose
    if docker compose version &>/dev/null; then
      compose_version=$(docker compose version 2>/dev/null | awk '{print $4}' || echo "Unknown")
    elif command -v docker-compose &>/dev/null; then
      compose_version=$(docker-compose --version 2>/dev/null | awk '{print $4}' | tr -d ',' || echo "Unknown")
    fi

    # Check for NVIDIA runtime
    if docker info 2>/dev/null | grep -q "nvidia"; then
      nvidia_runtime=true
    fi
  fi

  echo "DOCKER_AVAILABLE='$docker_available'"
  echo "DOCKER_VERSION='$docker_version'"
  echo "COMPOSE_VERSION='$compose_version'"
  echo "NVIDIA_RUNTIME='$nvidia_runtime'"
}

get_runtime_info() {
  local bun_version=""
  local node_version=""
  local python_version=""

  if command -v bun &>/dev/null; then
    bun_version=$(bun --version 2>/dev/null || echo "Unknown")
  fi

  if command -v node &>/dev/null; then
    node_version=$(node --version 2>/dev/null || echo "Unknown")
  fi

  if command -v python3 &>/dev/null; then
    python_version=$(python3 --version 2>/dev/null | awk '{print $2}' || echo "Unknown")
  fi

  echo "BUN_VERSION='$bun_version'"
  echo "NODE_VERSION='$node_version'"
  echo "PYTHON_VERSION='$python_version'"
}

show_platform_info() {
  log_section "Platform Information"

  # Collect all info
  eval "$(get_os_info)"
  eval "$(get_cpu_info)"
  eval "$(get_memory_info)"
  eval "$(get_gpu_info)"
  eval "$(get_docker_info)"
  eval "$(get_runtime_info)"

  # Display OS info
  echo -e "${BOLD}Operating System:${NC}"
  echo -e "  Name:        ${CYAN}$OS_NAME${NC}"
  echo -e "  Version:     $OS_VERSION"
  echo -e "  Kernel:      $KERNEL"
  echo ""

  # Display CPU info
  echo -e "${BOLD}CPU:${NC}"
  echo -e "  Model:       ${CYAN}$CPU_MODEL${NC}"
  echo -e "  Architecture:${CYAN} $CPU_ARCH${NC}"
  echo -e "  Cores:       $CPU_CORES"
  echo -e "  Threads:     $CPU_THREADS"
  echo -e "  Max Freq:    $CPU_FREQ"
  echo ""

  # Display Memory info
  echo -e "${BOLD}Memory:${NC}"
  echo -e "  Total:       ${CYAN}$MEM_TOTAL${NC}"
  echo -e "  Available:   $MEM_AVAILABLE"
  echo ""

  # Display GPU info
  echo -e "${BOLD}GPU:${NC}"
  if [[ "$GPU_AVAILABLE" == "true" ]]; then
    echo -e "  Status:      ${GREEN}Available${NC}"
    echo -e "  Name:        ${CYAN}$GPU_NAME${NC}"
    echo -e "  Memory:      $GPU_MEMORY MB (Free: $GPU_MEMORY_FREE MB)"
    echo -e "  Driver:      $GPU_DRIVER"
    echo -e "  CUDA:        $CUDA_VERSION"
    echo -e "  Compute Cap: $COMPUTE_CAP"
  else
    echo -e "  Status:      ${YELLOW}Not Available${NC}"
  fi
  echo ""

  # Display Docker info
  echo -e "${BOLD}Docker:${NC}"
  if [[ "$DOCKER_AVAILABLE" == "true" ]]; then
    echo -e "  Status:      ${GREEN}Available${NC}"
    echo -e "  Version:     $DOCKER_VERSION"
    echo -e "  Compose:     $COMPOSE_VERSION"
    echo -e "  NVIDIA Runtime: $(if [[ "$NVIDIA_RUNTIME" == "true" ]]; then echo -e "${GREEN}Yes${NC}"; else echo -e "${YELLOW}No${NC}"; fi)"
  else
    echo -e "  Status:      ${RED}Not Available${NC}"
  fi
  echo ""

  # Display Runtime info
  echo -e "${BOLD}Runtimes:${NC}"
  echo -e "  Bun:         ${BUN_VERSION:-Not installed}"
  echo -e "  Node:        ${NODE_VERSION:-Not installed}"
  echo -e "  Python:      ${PYTHON_VERSION:-Not installed}"
  echo ""

  # Recommended profile
  log_section "Recommended Configuration"
  if [[ "$GPU_AVAILABLE" == "true" ]]; then
    echo -e "  Profile:     ${GREEN}gpu${NC} (CUDA acceleration available)"
    echo -e "  Backend:     Python/PyTorch on $GPU_NAME"
    echo -e "  Port:        8006"
  else
    echo -e "  Profile:     ${CYAN}cpu${NC}"
    echo -e "  Backend:     TypeScript/ONNX"
    echo -e "  Port:        8005"
  fi
  echo ""
}

detect_platform() {
  # Check for NVIDIA GPU with working nvidia-smi
  if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
      # Verify GPU is actually usable
      local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
      if [[ -n "$gpu_count" && "$gpu_count" -gt 0 ]]; then
        echo "gpu"
        return
      fi
    fi
  fi

  # Check for NVIDIA environment variable (Docker)
  if [[ -n "${NVIDIA_VISIBLE_DEVICES}" && "${NVIDIA_VISIBLE_DEVICES}" != "none" ]]; then
    echo "gpu"
    return
  fi

  # Check for CUDA libraries
  if [[ -d "/usr/local/cuda" ]] || ldconfig -p 2>/dev/null | grep -q libcuda; then
    # CUDA libraries exist, check if GPU is accessible
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
      echo "gpu"
      return
    fi
  fi

  # Default to CPU
  echo "cpu"
}

# ============================================================
# Model Management
# ============================================================

check_models() {
  log_section "Model Status"

  cd "$PROJECT_DIR"

  # Run model check via CLI
  if [[ -f "src/cli.ts" ]]; then
    bun run src/cli.ts --check
  else
    log_warn "CLI not available, checking models manually..."

    # Check models directory
    if [[ -d "models" ]]; then
      echo ""
      echo -e "${BOLD}Downloaded Models:${NC}"
      for model_dir in models/*/; do
        if [[ -d "$model_dir" ]]; then
          local model_name=$(basename "$model_dir")
          local has_onnx=$(find "$model_dir" -name "*.onnx" 2>/dev/null | head -1)
          local has_pytorch=$(find "$model_dir" -name "*.safetensors" -o -name "pytorch_model.bin" 2>/dev/null | head -1)

          echo -n "  • $model_name: "
          if [[ -n "$has_onnx" ]]; then
            echo -ne "${GREEN}ONNX${NC} "
          fi
          if [[ -n "$has_pytorch" ]]; then
            echo -ne "${CYAN}PyTorch${NC} "
          fi
          echo ""
        fi
      done
    else
      log_warn "Models directory not found"
    fi
  fi
}

# ============================================================
# Benchmark
# ============================================================

run_benchmark() {
  log_section "Performance Benchmark"

  cd "$PROJECT_DIR"

  local cpu_result=""
  local gpu_result=""

  # Run CPU benchmark
  log_info "Running CPU benchmark (TypeScript/ONNX)..."
  if bun run cli -- --benchmark --iterations 5 2>/dev/null; then
    log_ok "CPU benchmark complete"
  else
    log_warn "CPU benchmark failed or not available"
  fi

  # Run GPU benchmark if available
  if [[ "$(detect_platform)" == "gpu" ]]; then
    log_info "Running GPU benchmark (Python/PyTorch)..."
    if docker ps --filter "name=inception-gpu" --format "{{.Names}}" | grep -q "inception-gpu"; then
      API_URL=http://localhost:8006 bun run --cwd demo src/index.ts benchmark --iterations 5 2>/dev/null || log_warn "GPU benchmark failed"
    else
      log_warn "GPU container not running. Start with: $0 --profile gpu"
    fi
  fi

  log_ok "Benchmark complete"
}

# ============================================================
# Service Management
# ============================================================

start_services() {
  local profile="$1"

  log_section "Starting Services"

  cd "$PROJECT_DIR"

  # Show platform info first
  eval "$(get_gpu_info)"

  log_info "Profile: $profile"
  log_info "GPU Available: $GPU_AVAILABLE"
  [[ "$GPU_AVAILABLE" == "true" ]] && log_info "GPU: $GPU_NAME"

  case "$profile" in
    gpu|cuda|spark)
      if [[ "$GPU_AVAILABLE" != "true" ]]; then
        log_error "GPU profile requested but no GPU detected!"
        log_info "Falling back to CPU profile..."
        profile="cpu"
      else
        log_info "Starting GPU backend (Python/PyTorch)..."

        # Check if container exists
        if docker ps -a --filter "name=inception-gpu" --format "{{.Names}}" | grep -q "inception-gpu"; then
          docker rm -f inception-gpu 2>/dev/null || true
        fi

        # Start GPU container
        docker run -d --rm \
          --name inception-gpu \
          --gpus all \
          --ipc=host \
          --ulimit memlock=-1 \
          --ulimit stack=67108864 \
          -v "$PROJECT_DIR/models:/models:rw" \
          -v "$PROJECT_DIR/python-gpu-service/src:/app:ro" \
          -p 8006:8006 \
          -e MODEL_CACHE_DIR=/models \
          -e PORT=8006 \
          -w /app \
          nvcr.io/nvidia/vllm:25.12-py3 \
          bash -c "pip install --quiet fastapi uvicorn sentence-transformers pydantic python-dotenv 2>/dev/null && python main.py"

        log_ok "GPU service started on port 8006"
        log_info "Health check: curl http://localhost:8006/health"
        return
      fi
      ;;
  esac

  # CPU fallback or explicit CPU profile
  if [[ "$profile" == "cpu" || "$profile" == "default" ]]; then
    log_info "Starting CPU backend (TypeScript/ONNX)..."
    bun run start &
    log_ok "CPU service started on port 8005"
    log_info "Health check: curl http://localhost:8005/health"
  fi
}

# ============================================================
# Main
# ============================================================

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --info|-i)
      SHOW_INFO=true
      shift
      ;;
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

# Show platform info if requested
if [[ "$SHOW_INFO" == "true" ]]; then
  show_platform_info
  exit 0
fi

# Auto-detect profile if not specified
if [[ -z "$PROFILE" ]]; then
  PROFILE=$(detect_platform)
  log_info "Auto-detected platform: $PROFILE"
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
