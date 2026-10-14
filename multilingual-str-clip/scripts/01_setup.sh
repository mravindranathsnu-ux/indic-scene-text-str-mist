#!/bin/bash
# Step 1: Create a virtualenv and install dependencies. Run this once
# per machine (not per language).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Step 1: Environment setup ==="
cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip --quiet

# Install torch/torchvision from a SPECIFIC CUDA build, matching your
# driver — pip's default (unpinned) install grabs the newest CUDA build
# available, which can be newer than your driver supports (a driver can
# run OLDER CUDA builds fine, but not newer ones). Run 02_check_gpu.sh
# once to find your driver's max CUDA version (via `nvidia-smi`), set
# TORCH_CUDA_INDEX_URL in config.sh to match.
if [ -n "$TORCH_CUDA_INDEX_URL" ]; then
    DESIRED_CUDA_TAG=$(echo "$TORCH_CUDA_INDEX_URL" | grep -oE 'cu[0-9]+' || true)
    CURRENT_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
    CURRENT_CUDA_TAG="cu$(echo "$CURRENT_CUDA_VERSION" | tr -d '.')"

    if [ -n "$DESIRED_CUDA_TAG" ] && [ "$CURRENT_CUDA_TAG" = "$DESIRED_CUDA_TAG" ]; then
        echo "torch: OK (matching CUDA build $CURRENT_CUDA_TAG already installed)."
    else
        echo "Installing torch/torchvision from $TORCH_CUDA_INDEX_URL (currently: ${CURRENT_CUDA_TAG:-none installed})..."
        # --force-reinstall is REQUIRED here: if some torch build is already
        # installed (any version), plain `pip install torch` sees
        # "requirement already satisfied" and SKIPS it entirely even if
        # it's the wrong CUDA build. We only reach this branch when we've
        # already confirmed a mismatch above, so the (slow, large-download)
        # reinstall only happens when actually needed, not on every run.
        # Left verbose deliberately (unlike the routine installs below) --
        # this is a large, infrequent download, and seeing progress is
        # genuinely useful rather than noise.
        pip install --force-reinstall torch torchvision --index-url "$TORCH_CUDA_INDEX_URL"
    fi
else
    echo "TORCH_CUDA_INDEX_URL not set in config.sh — using pip's default torch build."
    echo "If 02_check_gpu.sh then reports a driver/CUDA mismatch, set TORCH_CUDA_INDEX_URL"
    echo "in config.sh (e.g. https://download.pytorch.org/whl/cu128) and re-run this script."
fi

PIP_LOG="$PROJECT_ROOT/pip_install.log"
if pip install -r requirements.txt > "$PIP_LOG" 2>&1; then
    echo "requirements.txt: OK (full pip output: $PIP_LOG)"
else
    echo "!! pip install -r requirements.txt FAILED. Last 20 lines of $PIP_LOG:"
    tail -20 "$PIP_LOG"
    exit 1
fi

echo "Setup complete."
echo "Every later step script calls venv's python directly (no need to"
echo "manually 'source venv/bin/activate' again), but you can if you prefer"
echo "to run project commands by hand instead of via these scripts."
