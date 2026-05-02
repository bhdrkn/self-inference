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
