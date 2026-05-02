# Scripts

Helper scripts that aren't part of the main `src/` implementations: GPU rental setup, model download helpers, environment bootstrap, etc.

Keep these small, single-purpose, and well-commented.

## Conventions

- One script per purpose. Don't combine setup and benchmarking into one script.
- Every script must be runnable from a fresh RunPod instance without prior state.
- Scripts that require environment variables (e.g. `HF_TOKEN`) should fail fast with a clear error if they're missing — don't silently continue.

## Scripts

| Script | Post | Purpose |
|--------|------|---------|
| `setup_runpod.sh` | 01+ | Bootstrap a fresh RunPod instance: install `uv`, Python deps, pull Llama 3.1 8B weights from Hugging Face. |
| `stop_runpod.sh` | 01+ | Stop the RunPod instance via the RunPod API. Run this when done benchmarking — leaving the instance running is money burning. |
| `plot_results.py` | 01+ | Read JSON benchmark output from `benchmarks/results/post-N/`, produce SVGs for embedding in posts. |
