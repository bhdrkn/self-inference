#!/usr/bin/env bash
# Run a small benchmark against the local server to verify the pipeline
# end-to-end: server → benchmark → JSON output.
#
# This is NOT for generating post numbers — use this to confirm everything
# works before spending money on RunPod.
#
# Requires:
#   - Local server running (scripts/start-local-server.sh)
#   - ShareGPT dataset at benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json
#   - .env with MODEL_NAME set

set -euo pipefail

DATASET="benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT="benchmarks/results/post-01/smoke-test.json"
NUM_PROMPTS=5
CONCURRENCY="1"

# Load MODEL_NAME from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep MODEL_NAME | xargs)
fi
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"

# --- Checks ---
if [ ! -f "$DATASET" ]; then
    echo "ERROR: ShareGPT dataset not found at $DATASET"
    echo "Download it with:"
    echo "  curl -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \\"
    echo "       -o $DATASET"
    exit 1
fi

if ! curl -sf http://localhost:8000/v1/chat/completions -o /dev/null \
     -X POST -H "Content-Type: application/json" \
     -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' 2>/dev/null; then
    echo "ERROR: Server not responding at http://localhost:8000"
    echo "Start it with: scripts/start-local-server.sh"
    exit 1
fi

# --- Run ---
echo "Running smoke test ($NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT"
echo ""

.venv/bin/python benchmarks/benchmark.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --concurrency $CONCURRENCY \
    --output "$OUTPUT"

echo ""
echo "Smoke test complete. Results at $OUTPUT"
