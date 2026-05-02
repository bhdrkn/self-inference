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

(none yet)

## Open questions

(none — all resolved. See planning notes in conversation history.)

## Draft fragments

> Half-written paragraphs that might end up in the final post.

(none yet)

## Things I want to remember when writing

- Lead with the failure, not the setup. Reader should feel the pain in the first 200 words.
- Cite numbers from `benchmarks/results/post-01/` — never invent them.
- Resist explaining the KV cache here. That's Post 2's job. This post's job is to make the reader want to know about Post 2.
