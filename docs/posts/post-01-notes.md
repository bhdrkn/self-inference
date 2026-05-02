# Post 1 — Working notes

> Running notes, draft fragments, benchmark results, and surprises encountered while building Post 1. This is a scratchpad, not a draft. The post itself lives in `posts/01-naive.md` once writing starts.

## Build checklist

- [ ] Rent RunPod RTX 4090 instance, verify GPU access
- [ ] Install dependencies (`uv` + `transformers`, `torch`, `fastapi`, `uvicorn`)
- [ ] Pull Llama 3.1 8B Instruct from Hugging Face
- [ ] Write minimal FastAPI server that wraps `transformers.generate()`
- [ ] Verify single-request inference works end-to-end
- [ ] Write load testing script (use `wrk`, `vegeta`, or a custom asyncio script)
- [ ] Benchmark: latency at concurrency 1, 5, 10, 20
- [ ] Benchmark: GPU utilization during load test (`nvidia-smi dmon` or equivalent)
- [ ] Benchmark: memory usage with varying prompt lengths
- [ ] Save raw benchmark outputs to `benchmarks/results/post-01/`

## Surprises log

> Anything that didn't behave the way I expected. These are the post's most valuable content — write them down immediately, even half-formed.

(none yet)

## Open questions

- Should I use Llama 3.1 8B Instruct or a smaller model (e.g., Qwen 2.5 3B) to keep costs down? 8B is more representative of "real" inference workloads, so probably worth the cost.
- What's the right load testing tool? `vegeta` is great for HTTP, but variable-length LLM responses might confuse its latency math.
- Should the benchmark prompt distribution be uniform-length or realistic (mix of short and long)? Probably uniform for Post 1 (simpler story), realistic distribution becomes the wedge for Post 2.

## Draft fragments

> Half-written paragraphs that might end up in the final post.

(none yet)

## Things I want to remember when writing

- Lead with the failure, not the setup. Reader should feel the pain in the first 200 words.
- Cite numbers from `benchmarks/results/post-01/` — never invent them.
- Resist explaining the KV cache here. That's Post 2's job. This post's job is to make the reader want to know about Post 2.
