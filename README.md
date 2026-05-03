# self-inference

> A distributed systems engineer's path into AI infrastructure, learned by building.

This repo accompanies a six-part blog series in which I build a progressively more sophisticated LLM inference stack from scratch — starting with the worst possible serving setup and ending with a small, observable, multi-instance system that routes intelligently between replicas.

Each post is paired with a working implementation in this repo. The goal is not to build production inference infrastructure (you should use vLLM or similar). The goal is to **understand why production inference infrastructure looks the way it does**, by feeling the problems firsthand.

## Who this is for

Engineers who have shipped backend services and want to understand what is genuinely different about serving language models — and are tired of "intro to LLMs" content that stops where the interesting problems start.

If you have ever built something that had to handle real load and wondered what makes GPU inference behave unlike anything else you have served — this series is for you.

## The series

| # | Post | Status | Branch | What breaks |
|---|------|--------|--------|-------------|
| 1 | Running an LLM at home is easy. Serving one is not. | ✅ Complete | `01-naive` | A FastAPI server wrapping `transformers.generate()` with a thread pool. Concurrent requests, standard backend setup. Measure what happens under load. |
| 2 | Why your GPU is bored: batching, KV cache, and the memory wall | ⚪ Planned | `02-batching` | Static batching on top of `transformers`. Better throughput, but a new failure mode appears: head-of-line blocking from variable-length requests. |
| 3 | Continuous batching and PagedAttention: how vLLM actually works | ⚪ Planned | `03-vllm` | Drop in vLLM. Read the scheduler and block manager source. Understand why it's different, not just that it's faster. |
| 4 | Routing inference: when one GPU isn't enough | ⚪ Planned | `04-routing` | Two vLLM instances behind a custom router. Round-robin as baseline. Prefix-aware routing to maximize KV cache reuse across replicas. |
| 5 | Observability for inference: what to measure and why | ⚪ Planned | `05-observability` | Instrument the Post 4 stack. TTFT, ITL, p99 under variable request cost. What the metrics reveal — and what they hide. |
| 6 | What I'd do differently: a retrospective | ⚪ Planned | `06-retrospective` | Honest reflection on what surprised me, what I got wrong, and what I deliberately left untouched. |

Each branch contains the code as it exists at the end of that post. `master` always reflects the latest completed post.

## Reproducing

You will need:

- A GPU. Cheapest option: rent one on RunPod (~$0.34–0.69/hour for an RTX 4090). For Posts 3+ you'll want an A100 40GB occasionally.
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv) for dependency management.
- A Hugging Face account and access token (for pulling open model weights).

Each post's branch has its own README with setup steps and benchmark commands.

## Running on RunPod

### Starting a new pod

1. Create a pod with an RTX 4090 (or A100 for Posts 3+), set disk to at least 50 GB.
2. Under **Environment Variables**, set:
   - `HF_TOKEN` — your Hugging Face token
   - `MODEL_NAME` — e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - Any post-specific vars (e.g. `BATCH_SIZE`, `BATCH_TIMEOUT_MS` for Post 2)
3. Once the pod is up, open a terminal and run:
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/bhdrkn/self-inference/<branch>/scripts/run-server.sh)
```
Replace `<branch>` with the post branch you're running (e.g. `02-batching`).

4. Verify the model loaded onto the GPU:
```bash
nvidia-smi                                              # should show ~16 GB VRAM used
grep -E "Loading|Model ready|Batching loop" /workspace/server.log
```

5. Sanity-check the endpoint:
```bash
curl -s https://<pod-id>.proxy.runpod.net/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"model":"llama","messages":[{"role":"user","content":"hi"}],"max_tokens":10}' \
  | python3 -m json.tool
```

### Updating a running pod

If you push new code and want to pick it up without recreating the pod:
```bash
cd /workspace/self-inference
git fetch origin && git reset --hard origin/<branch>
kill $(cat /workspace/server.pid)
bash scripts/run-server.sh
```

## Troubleshooting

### Model not using the GPU (VRAM shows ~2 MiB)

PyTorch silently falls back to CPU when the installed wheel's CUDA version doesn't match the driver. This repo pins torch on Linux to the PyTorch CUDA 12.4 index via `[tool.uv.sources]` in `pyproject.toml`. If you're on a different driver, update the index URL accordingly.

Diagnose with:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi  # should show model VRAM (e.g. ~16 GB for Llama 8B), not 2 MiB
```

See `docs/posts/post-01-notes.md` for the full write-up.

## Why I'm writing this

I have a decade of distributed systems experience and am currently working on agentic LLM systems. I want to move deeper into the infrastructure side of AI — and the most honest way to learn it is to build it badly first, then better, in public.

If you're on a similar path, follow along. If you spot something wrong, open an issue.

## License

MIT. Use anything here however you like.
