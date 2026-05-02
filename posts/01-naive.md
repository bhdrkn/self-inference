# Running an LLM at home is easy. Serving one is not.

> Post 1 of *Self-Hosting Intelligence*. Code: branch [`01-naive`](../).

---

## Picking a model

Before writing a single line of server code, there's a decision that shapes everything downstream: which model to run. The choice is not about which model produces the best output — it's about what constraints it places on hardware, and whether those constraints are representative of real serving problems.

### The candidates

The two serious options for a project like this are:

- **Llama 3.1 8B Instruct** (Meta) — the current reference-class open-weights model for single-GPU serving. Widely benchmarked, well-documented failure modes, and what most people actually reach for when self-hosting.
- **Qwen 2.5 3B** (Alibaba) — a capable 3B model that punches above its weight on benchmarks. Smaller footprint, lower cost.

Both are open-weights, commercially licensed, and available on Hugging Face. Either would work for the serving experiments in this series.

### What model size actually means

"8B parameters" means the model has 8 billion learned floating-point weights. These are the values adjusted during training to encode the patterns the model learned from its training corpus — syntax, facts, reasoning strategies, style. More parameters means more capacity to encode more complex patterns, but it does not mean strictly better outputs on every task. A 3B model fine-tuned on the right data can outperform an 8B base model on a narrow task.

For inference, the parameter count has a direct and unavoidable implication: **every parameter must live in memory before the first token can be generated**. The model weights are not streamed or paged like application data — the full weight matrix must be resident in GPU VRAM for the forward pass to run.

### The memory math

In production, models are typically loaded in **bf16** (bfloat16), a 16-bit floating-point format that halves memory use versus fp32 with negligible quality loss. At bf16, each parameter occupies 2 bytes.

| Model | Parameters | Weight size (bf16) | Overhead (KV cache, activations) | Minimum VRAM |
|---|---|---|---|---|
| Qwen 2.5 3B | 3 × 10⁹ | ~6 GB | ~2–3 GB | ~8–9 GB |
| Llama 3.1 8B | 8 × 10⁹ | ~16 GB | ~2–4 GB | ~18–20 GB |

The overhead column is not a rounding artifact — the KV cache (explained in depth in Post 2) grows with sequence length and can dominate VRAM at high concurrency. For now, treat the overhead as a ~20% buffer on top of weight size.

These numbers determine which GPU tier you need:

- 8–10 GB VRAM: RTX 3080. Fits Qwen 2.5 3B comfortably. Llama 3.1 8B will not fit.
- 24 GB VRAM: RTX 3090 or RTX 4090. Either model fits with headroom to spare.

### Cost implications

On RunPod's community cloud, the relevant tiers as of mid-2026:

| GPU | VRAM | Rate (community cloud) |
|---|---|---|
| RTX 3080 | 10 GB | ~$0.17/hr |
| RTX 3090 | 24 GB | ~$0.22/hr |
| RTX 4090 | 24 GB | ~$0.34/hr |

For this project — a few hours of benchmarking per post — the difference between running Qwen 2.5 3B on an RTX 3080 versus Llama 3.1 8B on an RTX 4090 is roughly **$0.17/hr**. Over a three-hour session, that's about $0.50. Negligible.

In production, the math looks different. A serving fleet running 24/7 at that $0.17/hr delta:

```
$0.17/hr × 24 hr × 365 days = ~$1,490/GPU/year
```

Across 100 GPUs, the model size choice alone accounts for **~$149,000/year** in hardware cost — before factoring in power, cooling, or the operational overhead of managing a larger fleet. At that scale, the decision between a 3B and an 8B model is a budget line item, not a benchmark footnote.

For us, the $0.50 is not the deciding factor. Something else is.

### Why Llama 3.1 8B

The serving problems this series is designed to expose — request queuing, GPU underutilization, KV cache pressure, head-of-line blocking — are architectural, not model-specific. A 3B model would demonstrate them just as clearly.

But this series is also about building credibility with an audience that serves models at scale. Llama 3.1 8B is what that audience actually runs. Using a 3B model risks the reaction: "interesting, but 8B behaves differently." It probably doesn't, but the doubt is distracting.

More practically: the failure modes we will instrument in Post 2 and Post 3 — KV cache eviction under memory pressure, continuous batching scheduler decisions — become visible at 8B in a way they might not at 3B, because the memory budget is tight enough that the scheduler actually has to make tradeoffs.

We are going with **Llama 3.1 8B Instruct** on an **RTX 4090** (24 GB VRAM, ~$0.34/hr community cloud).

---

## Designing the server

The model is chosen. Now the server.

The goal is to make something that a competent backend engineer would actually ship — not a straw man we already know is broken. The failure should come from the problem, not from the implementation.

### The approach we skipped

The simplest possible server would handle one request at a time: a single thread, no concurrency, every request blocks until `generate()` finishes before the next one starts.

```python
@app.post("/generate")
def generate(req: GenerateRequest):
    # blocks the server for the entire generation
    output = model.generate(**tokenize(req.prompt))
    return decode(output)
```

This is too naive to be instructive. We can already see why it fails before running a single benchmark: requests queue behind each other, the server can accept no new work during generation, and the GPU is idle between requests. There's nothing surprising there. Building it, measuring it, and writing about it would be wasted words.

### The approach we're taking

Any backend engineer who has shipped a high-throughput service would look at the above and reach immediately for concurrency. The service is slow under load? Add a thread pool. Let multiple requests run at the same time. This is standard practice — it's what you'd do for a slow database query, a blocking HTTP call, or any other I/O-bound operation.

`transformers.generate()` is a blocking call. FastAPI and uvicorn are built around async I/O. The correct bridge between them is `run_in_executor`: offload the blocking work to a thread pool, free the event loop to accept the next request.

```python
executor = ThreadPoolExecutor(max_workers=4)

@app.post("/generate")
async def generate(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run_inference, req)
    return result
```

The HTTP layer is fully async — uvicorn accepts new connections immediately. The thread pool allows multiple `generate()` calls to be in flight simultaneously. The server won't block on a single request. This is the setup a senior engineer would deploy.

Let's see what happens when we put it under load.

---

*[Remainder of post pending. See `docs/posts/post-01-notes.md` for working notes and benchmark results.]*
