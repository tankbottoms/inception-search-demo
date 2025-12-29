#!/bin/bash
# Run CPU vs GPU benchmark comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "[INFO] Running benchmark comparison..."

# Run benchmark
bun run src/cli.ts --benchmark
