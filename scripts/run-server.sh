#!/usr/bin/env bash
# Runs on the RunPod pod.
#
# - Installs uv if missing
# - Clones / updates the repo
# - Installs Python dependencies (CUDA-enabled torch on Linux)
# - Pre-downloads model weights from HuggingFace
# - Starts the inference server in the background
#
# Required environment variables (set in RunPod pod config):
#   HF_TOKEN    HuggingFace token with access to the model
#   MODEL_NAME  Model to serve (default: meta-llama/Meta-Llama-3.1-8B-Instruct)

set -euo pipefail

REPO_URL="https://github.com/bhdrkn/self-inference.git"
BRANCH="01-naive"
WORK_DIR="/workspace/self-inference"
LOG_FILE="/workspace/server.log"
PID_FILE="/workspace/server.pid"
SERVER_READY_TIMEOUT=120  # seconds — model is pre-downloaded so startup should be fast

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Add it under Environment Variables in the RunPod pod configuration."
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

echo "==> Model : $MODEL_NAME"
echo "==> Workdir: $WORK_DIR"
echo ""

# ---------------------------------------------------------------------------
# Install uv
# ---------------------------------------------------------------------------

if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer adds uv to PATH in shell profile — source it for this session
    export PATH="$HOME/.local/bin:$PATH"
    echo "    uv $(uv --version) installed."
else
    echo "==> uv already installed ($(uv --version))"
fi

# ---------------------------------------------------------------------------
# Clone / update repo
# ---------------------------------------------------------------------------

if [ ! -d "$WORK_DIR/.git" ]; then
    echo "==> Cloning repo (branch: $BRANCH)..."
    git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
else
    echo "==> Repo exists, pulling latest..."
    git -C "$WORK_DIR" fetch origin
    git -C "$WORK_DIR" checkout "$BRANCH"
    git -C "$WORK_DIR" reset --hard "origin/$BRANCH"
fi

cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------

echo ""
echo "==> Installing Python dependencies..."
# torch on Linux is resolved from the pytorch-cu124 index (see pyproject.toml)
# so uv sync installs the CUDA 12.4-compatible wheel automatically.
uv sync --group post-01
echo "    Torch: $(uv run python -c 'import torch; print(torch.__version__, \"| CUDA:\", torch.cuda.is_available())')"

# ---------------------------------------------------------------------------
# Pre-download model weights
# ---------------------------------------------------------------------------

echo ""
echo "==> Downloading model weights: $MODEL_NAME"
echo "    This takes 15–25 minutes on first run (~16 GB). Subsequent runs are instant."
echo ""

uv run python - <<PYEOF
import os
from huggingface_hub import snapshot_download

model = "$MODEL_NAME"
token = os.environ["HF_TOKEN"]

print(f"Downloading {model} ...")
path = snapshot_download(repo_id=model, token=token)
print(f"Weights cached at: {path}")
PYEOF

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------

# Stop any existing server instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo ""
        echo "==> Stopping existing server (PID $OLD_PID)..."
        kill "$OLD_PID" && sleep 2
    fi
    rm -f "$PID_FILE"
fi

echo ""
echo "==> Starting inference server..."
echo "    MODEL_NAME=$MODEL_NAME"
nohup env MODEL_NAME="$MODEL_NAME" HF_TOKEN="$HF_TOKEN" \
    uv run python src/server.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "    PID: $(cat $PID_FILE) | Log: $LOG_FILE"

# ---------------------------------------------------------------------------
# Wait for ready
# ---------------------------------------------------------------------------

echo ""
ELAPSED=0
LAST_SHOWN=""

while [ $ELAPSED -lt $SERVER_READY_TIMEOUT ]; do
    if ! kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo ""
        echo "ERROR: Server process exited unexpectedly. Last log output:"
        echo "---"
        tail -20 "$LOG_FILE"
        echo "---"
        exit 1
    fi

    if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "==> Server ready on port 8000"
        echo ""
        echo "    To tail logs : tail -f $LOG_FILE"
        echo "    To stop      : kill \$(cat $PID_FILE)"
        exit 0
    fi

    CURRENT_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null || true)
    if [ "$CURRENT_LINE" != "$LAST_SHOWN" ] && [ -n "$CURRENT_LINE" ]; then
        echo "    [${ELAPSED}s] $CURRENT_LINE"
        LAST_SHOWN="$CURRENT_LINE"
    fi

    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

echo ""
echo "Timed out after ${SERVER_READY_TIMEOUT}s. Check $LOG_FILE for progress."
exit 1
