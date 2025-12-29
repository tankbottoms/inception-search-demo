#!/bin/bash
# Check and download/convert models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "[INFO] Checking models..."

# Check if bun is available
if ! command -v bun &> /dev/null; then
  echo "[ERROR] Bun is not installed. Please install Bun first."
  exit 1
fi

# Run model check
bun run src/cli.ts --check
