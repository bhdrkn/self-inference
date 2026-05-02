# Post 1 — Working notes

> Running notes, draft fragments, benchmark results, and surprises encountered while building Post 1. This is a scratchpad, not a draft. The post itself lives in `posts/01-naive.md` once writing starts.

## Build checklist

### Scripts
- [ ] Write `scripts/setup_runpod.sh` — bootstrap RunPod instance: install `uv`, pull Llama 3.1 8B weights (requires `HF_TOKEN` env var)
- [ ] Write `scripts/stop_runpod.sh` — stop instance via RunPod API

### Server
- [ ] Add Post 1 dependencies to `pyproject.toml`: `transformers`, `torch`, `fastapi`, `uvicorn`, `altair`
- [ ] Write `src/server.py`: FastAPI + `transformers.generate()` + `ThreadPoolExecutor(max_workers=4)`
- [ ] Add mock mode (`MOCK_MODE=true` env var): skip model load, sleep for configurable delay, return dummy tokens

### Local validation (before touching RunPod)
- [ ] Start server in mock mode (`MOCK_MODE=true`), send 3–5 requests manually, verify response schema
- [ ] Run `benchmark_serving.py` against mock server, verify JSON output lands in `benchmarks/results/post-01/`
- [ ] Run `scripts/plot_results.py` against mock output, verify SVGs render correctly
- Note: real inference not possible locally — PyTorch 2.3+ (required by transformers 5.x) dropped Intel Mac x86_64 wheels

### Benchmarking on RunPod
- [ ] Rent RTX 4090 instance, run `scripts/setup_runpod.sh`, verify GPU access
- [ ] Start server with Llama 3.1 8B, verify single-request inference end-to-end
- [ ] Download ShareGPT dataset locally (`ShareGPT_V3_unfiltered_cleaned_split.json`, ~200 MB)
- [ ] Run `benchmark_serving.py` at concurrency 1, 5, 10, 20 — save raw JSON to `benchmarks/results/post-01/`
- [ ] Capture GPU utilization during load test (`nvidia-smi dmon` on RunPod)
- [ ] Run `scripts/stop_runpod.sh`

### Post-benchmark
- [ ] Run `scripts/plot_results.py` against real results, generate final SVGs
- [ ] Save SVGs to `benchmarks/results/post-01/`

## Surprises log

> Anything that didn't behave the way I expected. These are the post's most valuable content — write them down immediately, even half-formed.

### torch + CUDA version mismatch — the silent GPU bypass

**What happened.** `uv sync` installed a torch build compiled for CUDA 12.6+, but the RunPod driver was CUDA 12.4 (driver 550.127.05). At startup, PyTorch printed a one-line warning about `CUDA initialization: Unexpected error` and fell back to CPU — silently. The server returned 200 OK, the benchmark ran, and `nvidia-smi` showed 2 MiB VRAM. No crash, no error in the response. Everything _looked_ fine.

**Why it matters.** This is a category of bug that a backend engineer doesn't instinctively reach for. An incompatible shared library usually segfaults or throws an obvious import error. PyTorch's graceful CPU fallback is a feature, not a bug — but it means a misconfigured deployment looks identical to a working one from the outside.

**How to detect it.** Three places to check, in order:

1. `nvidia-smi` — a working server with Llama 8B loaded should show ~16 GB VRAM consumed. 2 MiB means the GPU is unused.
2. Server startup logs — look for `Loading … on cuda` vs `Loading … on cpu`. The lifespan handler logs this explicitly.
3. Python one-liner: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` — `False` or a version mismatch tells the story.

**Root cause.** PyTorch publishes separate wheel indexes per CUDA version:
- `https://download.pytorch.org/whl/cu124` — for CUDA 12.4 drivers (550.x)
- `https://download.pytorch.org/whl/cu126` — for CUDA 12.6+ drivers

PyPI's default torch wheels target the _latest_ CUDA release. `uv sync` pulled from PyPI and got a cu126 wheel. The fix is to tell uv to resolve torch from the cu124 index on Linux via `[tool.uv.sources]` in `pyproject.toml`.

**Secondary surprise.** The fix itself surfaced a uv API version mismatch: `[[tool.uv.indexes]]` (plural) was the old field name; `[[tool.uv.index]]` (singular) is what uv 0.11.x expects. The error message from uv listed the expected field names, which made it easy to spot.

**Worth writing about.** This is a good concrete example of why infrastructure debugging for ML systems requires checking one layer below the application: the CUDA driver/runtime ABI contract isn't surfaced by the application layer at all.

---

## Troubleshooting

### Is the model loaded on the GPU?

Run these on the RunPod pod:

```bash
# 1. Check VRAM — should show ~16 GB used for Llama 8B, not 2 MiB
nvidia-smi

# 2. Check server startup log — should say "on cuda", not "on cpu"
grep -E "Loading|Device map|Model ready" /workspace/server.log

# 3. One-liner inside the venv
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version torch was built for:', torch.version.cuda)"
```

Expected output when working correctly:
```
CUDA available: True
CUDA version torch was built for: 12.4
```

### uv resolved the wrong torch wheel

Symptom: `torch.cuda.is_available()` returns `False` on a CUDA-capable machine.

Cause: uv resolved torch from PyPI (which ships the latest CUDA variant), but the pod's NVIDIA driver supports an earlier CUDA version.

Fix: `pyproject.toml` uses `[tool.uv.sources]` to pin torch on Linux to the PyTorch CUDA 12.4 index:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

After any change to `pyproject.toml`, re-run `./scripts/run-server.sh` on the pod — it runs `uv sync` which will pull the correct wheel.

### uv index field name error

Symptom:
```
unknown field `indexes`, expected one of ... `index` ...
```

Cause: uv 0.11.x uses `[[tool.uv.index]]` (singular). The plural `indexes` form is not recognised.

Fix: use `[[tool.uv.index]]` in `pyproject.toml`. Already fixed in this repo.

### MODEL_NAME not picked up

Symptom: server logs show loading `Llama-3.2-1B-Instruct` instead of `Meta-Llama-3.1-8B-Instruct`.

Cause: `MODEL_NAME` env var not set in the RunPod pod environment, so the server uses the default (1B model for local dev).

Fix: in RunPod pod settings → Environment Variables, set `MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct`. Verify with `echo $MODEL_NAME` before running `run-server.sh`.

---

## Open questions

(none — all resolved. See planning notes in conversation history.)

## Draft fragments

> Half-written paragraphs that might end up in the final post.

(none yet)

## Things I want to remember when writing

- Lead with the failure, not the setup. Reader should feel the pain in the first 200 words.
- Cite numbers from `benchmarks/results/post-01/` — never invent them.
- Resist explaining the KV cache here. That's Post 2's job. This post's job is to make the reader want to know about Post 2.
