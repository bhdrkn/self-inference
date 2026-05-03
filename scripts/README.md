# Scripts

Helper scripts that aren't part of the main `src/` implementations: GPU rental setup, model download helpers, environment bootstrap, etc.

Keep these small, single-purpose, and well-commented.

## Conventions

- One script per purpose. Don't combine setup and benchmarking into one script.
- Every script must be runnable from a fresh RunPod instance without prior state.
- Scripts that require environment variables (e.g. `HF_TOKEN`) should fail fast with a clear error if they're missing — don't silently continue.

## Scripts

| Script | Runs on | Post | Purpose |
|--------|---------|------|---------|
| `run-server.sh` | RunPod pod | 01+ | Install `uv`, clone repo, install CUDA deps, download model weights, start inference server. |
| `start-local-server.sh` | Laptop | 01+ | Start inference server locally, wait until ready, report status. |
| `stop-local-server.sh` | Laptop | 01+ | Stop the local server using the PID file. |
| `run-smoke-test-local.sh` | Laptop | 01+ | Run 5 prompts against the local server to verify the benchmark pipeline end-to-end. |
| `plot_results.py` | Laptop | 01+ | Read JSON benchmark output from `benchmarks/results/post-N/`, produce SVGs for embedding in posts. |

## Troubleshooting `run-server.sh`

### Verify the model loaded onto the GPU

After the server prints "Server ready on port 8000", check on the pod:

```bash
# Should show ~16 GB VRAM used (Llama 8B), not 2 MiB
nvidia-smi

# Should say "on cuda (torch.bfloat16)", not "on cpu"
grep -E "Loading|Device map|Model ready" /workspace/server.log
```

If VRAM shows ~2 MiB, the model is running on CPU — see the torch CUDA mismatch section below.

### torch CUDA version mismatch

Symptom: `nvidia-smi` shows 2 MiB VRAM; server logs say `on cpu`; `torch.cuda.is_available()` returns `False`.

Cause: `uv sync` resolved a torch wheel built for a newer CUDA than the pod's driver. PyTorch falls back to CPU silently — no crash, no error in responses.

Check the torch CUDA version:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

This repo pins torch on Linux to the CUDA 12.4 index via `[tool.uv.sources]` in `pyproject.toml`. If you're running on a pod with a different driver, change the index URL to match (e.g. `cu126` for CUDA 12.6+).

### MODEL_NAME not set

Symptom: server loads `Llama-3.2-1B-Instruct` (the local dev default) instead of the 8B model.

Fix: set `MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct` in RunPod's Environment Variables, or export it before running the script:

```bash
export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
./scripts/run-server.sh
```
