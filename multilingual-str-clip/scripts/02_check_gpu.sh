#!/bin/bash
# Step 2: Confirm GPU is visible and actually usable (not just "available"
# per the driver). Run once per machine; re-run after any driver/CUDA/
# PyTorch reinstall.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 2: GPU / environment check ==="
cd "$PROJECT_ROOT"
"$PYTHON" check_gpu.py
