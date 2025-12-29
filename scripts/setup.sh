#!/bin/bash
# Inception ONNX - Automated Setup Script
# Downloads, converts, and validates all required models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
CONVERTER_DIR="$PROJECT_ROOT/converter"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${BLUE}━━━ Inception ONNX Setup ━━━${NC}\n"

# ============================================================
# Step 1: Platform Detection
# ============================================================
log_info "Detecting platform..."

ARCH=$(uname -m)
OS=$(uname -s)
GPU_AVAILABLE=false
GPU_NAME=""
CUDA_VERSION=""

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        GPU_AVAILABLE=true
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)

        # Try to get CUDA version
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' || true)
        fi
    fi
fi

echo "  Platform: $OS ($ARCH)"
if [ "$GPU_AVAILABLE" = true ]; then
    log_success "GPU detected: $GPU_NAME"
    [ -n "$CUDA_VERSION" ] && echo "  CUDA: $CUDA_VERSION"
else
    log_warn "No GPU detected, will use CPU"
fi
echo ""

# ============================================================
# Step 2: Check Dependencies
# ============================================================
log_info "Checking dependencies..."

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 found: $(command -v $1)"
        return 0
    else
        log_error "$1 not found"
        return 1
    fi
}

DEPS_OK=true
check_command "bun" || DEPS_OK=false
check_command "python3" || DEPS_OK=false
check_command "docker" || log_warn "Docker not found (optional for containerized setup)"

if [ "$DEPS_OK" = false ]; then
    log_error "Missing required dependencies"
    exit 1
fi
echo ""

# ============================================================
# Step 3: Install Node Dependencies
# ============================================================
log_info "Installing Node.js dependencies..."
cd "$PROJECT_ROOT"
bun install --quiet
log_success "Dependencies installed"
echo ""

# ============================================================
# Step 4: Setup Python Converter Environment
# ============================================================
log_info "Setting up Python converter environment..."

cd "$CONVERTER_DIR"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    log_info "Created virtual environment"
fi

source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet optimum transformers torch onnx onnxruntime onnxscript sentence-transformers
log_success "Python environment ready"
echo ""

# ============================================================
# Step 5: Download and Convert Models
# ============================================================
log_info "Processing models from registry..."

cd "$PROJECT_ROOT"
REGISTRY="$MODELS_DIR/registry.json"

# Parse registry and convert embedding models
MODELS=$(python3 -c "
import json
with open('$REGISTRY') as f:
    registry = json.load(f)
for m in registry.get('models', []):
    if m.get('enabled') and m.get('type') == 'embedding':
        print(m['name'])
")

for MODEL_NAME in $MODELS; do
    SAFE_NAME=$(echo "$MODEL_NAME" | sed 's|/|--|g')
    MODEL_PATH="$MODELS_DIR/$SAFE_NAME"

    if [ -f "$MODEL_PATH/model.onnx" ] || [ -f "$MODEL_PATH/model.onnx.data" ]; then
        log_success "Model already exists: $SAFE_NAME"
        continue
    fi

    log_info "Converting model: $MODEL_NAME"

    cd "$CONVERTER_DIR"
    source .venv/bin/activate

    # Use PyTorch ONNX export
    python3 << EOF
import os
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "$MODEL_NAME"
output_dir = "$MODEL_PATH"
os.makedirs(output_dir, exist_ok=True)

print(f"  Loading model: {model_name}")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.eval()

print(f"  Saving tokenizer...")
tokenizer.save_pretrained(output_dir)

print(f"  Exporting to ONNX...")
dummy_input = tokenizer("Hello world", return_tensors="pt")

with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        os.path.join(output_dir, "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "last_hidden_state": {0: "batch_size", 1: "sequence"},
        },
        opset_version=14,
    )

print(f"  Model exported to {output_dir}")
EOF

    if [ -f "$MODEL_PATH/model.onnx" ]; then
        log_success "Converted: $SAFE_NAME"
    else
        log_error "Conversion failed: $MODEL_NAME"
    fi
done

cd "$PROJECT_ROOT"
echo ""

# ============================================================
# Step 6: Validate Setup
# ============================================================
log_info "Validating setup..."

# Run model check
bun run cli -- --check 2>/dev/null | grep -E "(✓|✗|Available|Not available)" || true
echo ""

# ============================================================
# Step 7: Quick Test
# ============================================================
log_info "Running quick inference test..."

# Start server in background
bun run start &
SERVER_PID=$!
sleep 5

# Test health
HEALTH=$(curl -s http://localhost:8005/health 2>/dev/null || echo '{}')
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    log_success "Server is healthy"

    # Test embedding
    RESULT=$(curl -s -X POST http://localhost:8005/api/v1/embed/query \
        -H "Content-Type: application/json" \
        -d '{"text": "test query"}' 2>/dev/null || echo '{}')

    if echo "$RESULT" | grep -q '"embedding"'; then
        DIMS=$(echo "$RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('embedding',[])))" 2>/dev/null || echo "0")
        log_success "Inference working: ${DIMS}-dim embeddings"
    else
        log_error "Inference test failed"
    fi
else
    log_error "Server health check failed"
fi

# Stop server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo ""

# ============================================================
# Summary
# ============================================================
echo -e "${BLUE}━━━ Setup Complete ━━━${NC}\n"
echo "Next steps:"
echo "  1. Start backend:  bun run dev"
echo "  2. Run demo:       cd demo && bun run demo"
echo "  3. Full stack:     docker compose up"
echo ""
