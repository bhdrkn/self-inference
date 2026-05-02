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

## Benchmark design

### What we're using and why

**Dataset: ShareGPT V3** (`ShareGPT_V3_unfiltered_cleaned_split.json`, ~469 MB, gitignored)

94,145 real conversations scraped from ChatGPT, totalling 702k turns (334k human, 368k assistant). We sample 200 conversations and use only the **first human turn** from each as the prompt. Why first-turn only:

- Multi-turn would inflate prompt token counts with conversation history, making throughput numbers hard to compare across posts
- We're measuring the serving layer under load, not conversation fidelity
- The prompt length distribution from first turns is already representative enough to stress the server

Why ShareGPT over synthetic prompts:
- Realistic prompt length variance — short one-liners and long pastes both appear
- Reproducible via fixed random seed (`--seed 42`)
- The same dataset is reused across all posts so benchmark results are directly comparable

**Script: `benchmarks/benchmark.py`**

Custom script rather than the vLLM benchmark tool (`vllm bench serve`), which was deprecated and removed from the CLI. Sends async HTTP requests using `aiohttp`, controls concurrency with a semaphore, collects per-request latency and token counts from the server's `usage` response field.

### How we run it

```bash
uv run python benchmarks/benchmark.py \
    --host <pod>.proxy.runpod.net --port 443 --https \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 200 \
    --concurrency 1 5 10 20 \
    --output benchmarks/results/post-01/results.json
```

- Runs from the laptop against the RunPod pod over HTTPS (RunPod proxy requires it)
- Each concurrency level sends all 200 prompts with that many in-flight at once
- Levels run sequentially (not parallel) — one run finishes before the next starts
- Results saved as a single JSON file per run

For the 1B preview (before 8B access is approved): run concurrency 5 only, save to `results-1b-preview.json`. Full 4-level benchmark runs once with 8B.

### Known limitations of this benchmark design

**Single-turn only.** We use the first human turn from each ShareGPT conversation and discard the rest. This keeps prompt token counts predictable and results comparable across posts, but it has two consequences that matter later:

1. **Post 2 (batching)**: head-of-line blocking is most visible with conversations of varying lengths. Post 2 introduces `--conversation-mode` in the benchmark, which sends accumulated multi-turn history and makes the HOL blocking problem concrete.

2. **Post 4 (prefix routing)**: prefix-aware routing only has something to exploit if requests share common prefixes (same conversation history). Independent first turns have unique prefixes — the routing optimization would be invisible. Post 4's benchmark runs in conversation mode to demonstrate the improvement.

**No response quality check.** The benchmark measures throughput and latency only — it ignores what the model actually outputs. For Posts 1 and 3 (vLLM drop-in) this is fine. **Post 2 is the exception**: static batching introduces attention masks and padding, and incorrect masking can silently corrupt outputs (the model still returns 200 OK with plausible-looking text). Post 2 adds a regression check: run the same prompt solo and batched, confirm outputs match.

### Metrics we collect

| Metric | What it measures |
|--------|-----------------|
| `throughput_tokens_per_s` | Completion tokens generated per wall-clock second across all requests — the primary throughput signal |
| `throughput_requests_per_s` | Requests completed per second — useful for SLA framing |
| `latency_mean_s` | Mean end-to-end request latency (time-to-last-token for non-streaming) |
| `latency_p50_s` | Median latency — typical experience |
| `latency_p90_s` | 90th percentile — what most users see under load |
| `latency_p99_s` | Tail latency — where the pain shows up as concurrency grows |

**What we're looking for in the results:**

At concurrency 1, the server is a simple wrapper — throughput is bounded by single-sequence generation speed. As concurrency increases, we expect throughput to flatten quickly (no batching) while latency rises sharply. The p99 diverging from p50 is the headline story: requests queue behind each other because `transformers.generate()` is synchronous and the thread pool serializes on the GPU. That's the setup for Post 2.

We're also capturing `nvidia-smi dmon` during the run to show GPU utilization — another way to see the wasted capacity.

---

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
