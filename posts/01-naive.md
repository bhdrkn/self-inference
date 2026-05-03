# Running an LLM at home is easy. Serving one is not.

> Post 1 of *Self-Hosting Intelligence*. Code: branch [`01-naive`](../).

---

## Introduction

Running a language model locally takes one command. `ollama run llama3`, wait thirty seconds, and you have a model answering questions in your terminal. The hard part is already done — someone else packaged the weights, wrote the runtime, and figured out the memory layout. You just pulled it.

Serving that model to users is a different problem. Suddenly you care about latency under concurrent load, GPU utilization, memory pressure, request queuing, and the gap between what your infrastructure handles and what users actually experience.

This series is about building an inference stack from scratch — one post at a time, deliberately making it fail before making it better. The goal is not to build production infrastructure. The goal is to understand why production inference infrastructure looks the way it does, by feeling the problems firsthand.

The target audience is engineers who have shipped backend services and want to understand what is genuinely different about serving language models. Serving an LLM has enough new concepts — forward passes, KV caches, GPU memory constraints — that they are worth explaining in context, as they become relevant, rather than front-loading them. When we hit something GPU-specific that behaves unlike anything you have served before, we will stop and explain it.

The approach: build the most reasonable server we can. No deliberate naivety. We want to investigate the bottlenecks on the GPU and model side of the problem, not re-discover basic HTTP server problems. Then measure it, explain precisely why it fails under load, and fix it.

First question: which model are we serving?

---

## Picking a model

In a real production setting, model selection involves evaluating dozens of options: API-hosted models from OpenAI, Anthropic, or Google; large open-weight models in the 70B–400B range; mid-size models from 7B to 13B; and increasingly capable small models under 4B. Each sits at a different point on the tradeoff between quality, latency, hardware cost, and operational complexity.

For this series, one constraint collapses most of that space: **the model must run on a single GPU we can rent for a few hours.** That eliminates API-hosted models (we are building the serving layer, not calling someone else's) and the very large open-weight models that require multiple high-end GPUs. What remains is the open-weight landscape from roughly 1B to 13B parameters — models that fit comfortably on a single 24 GB consumer GPU.

Within that space, the two most useful reference points are:

- **Llama 3.1 8B Instruct** (Meta) — the current reference-class model for single-GPU self-hosting. Widely benchmarked, well-documented, and what most people actually reach for when self-hosting.
- **Qwen 2.5 3B** (Alibaba) — a capable 3B model that represents the smaller end of the useful range. Lower memory footprint, lower cost, competitive quality for its size.

These are not the only options. Mistral 7B, Gemma 2 9B, Phi-3, and others are all legitimate choices. The reason we compare these two specifically is that they sit on opposite sides of a GPU tier boundary — 3B fits on an 8–10 GB GPU, 8B requires 24 GB — which makes the size tradeoff concrete rather than abstract.

### What model size actually means

"8B parameters" means the model has 8 billion learned floating-point weights — the values adjusted during training to encode everything the model knows. More parameters means more capacity to represent complex patterns, but not necessarily better outputs on every task. A smaller model trained on the right data can outperform a larger one on a specific task.

For inference, parameter count has a direct hardware implication: **every parameter must live in GPU memory before the model can generate a single token.**

To understand why, it helps to know what happens when you send a prompt to the model. Generating a response is done one token at a time. To generate each token, the model runs a **forward pass**: the current sequence of tokens is multiplied through dozens of weight matrices — one per attention head, one per feed-forward layer — producing a probability distribution over what token should come next. The highest-probability token is selected and appended to the sequence, then the process repeats. Every weight matrix participates in every forward pass. They all need to be in memory before the first token can be produced.

This is fundamentally different from serving application data. A database does not need to load its entire dataset into RAM to answer a query. A model does need all its weights in VRAM to run inference.

### The memory math

In production, models are typically loaded in **bf16** — bfloat16, a 16-bit floating-point format. To understand why this matters, a brief note on floating-point precision: numbers in computers are stored with a fixed number of bits. More bits means higher precision but more memory. The traditional format, **fp32** (32-bit float), uses 4 bytes per number and offers about 7 decimal digits of precision. **bf16** uses 2 bytes per number — half the memory — but with a different bit layout than the older fp16 format. It sacrifices some decimal precision but preserves a wide range of representable values, which is what deep networks actually need. For inference, the reduced precision has negligible effect on output quality. The memory saving is real.

At bf16, each parameter occupies 2 bytes:

| Model | Parameters | Weight size (bf16) | KV cache + activations | Minimum VRAM |
|---|---|---|---|---|
| Qwen 2.5 3B | 3 × 10⁹ | ~6 GB | ~2–3 GB | ~8–9 GB |
| Llama 3.1 8B | 8 × 10⁹ | ~16 GB | ~2–4 GB | ~18–20 GB |

The KV cache column is not a rounding artifact — it grows with the length of the sequences being processed and can dominate VRAM at high concurrency. Post 2 covers this in depth. For now, treat it as a ~20% buffer on top of weight size.

These numbers map directly to GPU tiers:

- **8–10 GB VRAM** (RTX 3080): Qwen 2.5 3B fits. Llama 3.1 8B does not.
- **24 GB VRAM** (RTX 3090 or RTX 4090): either model fits with room to spare.

### Cost implications

On RunPod's community cloud, as of mid-2026:

| GPU | VRAM | Rate (community cloud) |
|---|---|---|
| RTX 3080 | 10 GB | ~$0.17/hr |
| RTX 3090 | 24 GB | ~$0.22/hr |
| RTX 4090 | 24 GB | ~$0.34/hr |

For a few hours of benchmarking, the difference between the cheapest and most expensive option here is a few dollars. Negligible for a project.

In production, the math looks different. A fleet running 24/7 at the $0.17/hr delta between the RTX 3080 and RTX 4090 tiers:

```
$0.17/hr × 24 hr × 365 days = ~$1,490/GPU/year
```

Across 100 GPUs, the model size choice accounts for **~$149,000/year** in hardware cost — before power, cooling, or operational overhead. At that scale, the choice between a 3B and an 8B model is a budget line item.

For us, the cost difference is not the deciding factor.

### Why Llama 3.1 8B

There are a few reasons to reach for Llama 3.1 8B specifically over the alternatives:

**License.** The Llama 3 Community License permits commercial use with reasonable restrictions (primarily: don't use it to train competing models, and attribute Meta). Most open-weight models in this size range have similar terms, but Llama's specific terms are well understood by the community and have been reviewed by legal teams at many companies.

**Ecosystem.** Llama 3.1 8B is one of the most widely deployed open-weight models. The benchmarks are well-documented, the failure modes are known, and support across inference frameworks (vLLM, TGI, llama.cpp) is mature. Choosing a less common model introduces model-specific quirks on top of the infrastructure problems we are trying to isolate.

**Size and generality.** At 8B parameters, Llama 3.1 8B handles a broad range of tasks adequately — code, reasoning, conversation, instruction following. A 3B model handles the same range but with lower quality on harder tasks. For demonstrating serving infrastructure, this matters less than it would in production, but it makes the model a more representative reference point.

**For this series specifically:** the failure modes in Posts 2 and 3 — KV cache eviction under memory pressure, continuous batching scheduler decisions — require enough memory pressure to actually manifest. At 8B on a 24 GB card, the memory budget is tight enough that the scheduler has to make real tradeoffs.

We are going with **Llama 3.1 8B Instruct** on an **RTX 4090** (24 GB VRAM, ~$0.34/hr community cloud).

---

## Designing the server

We want to investigate the bottlenecks on the GPU and model side of the problem, not revisit general HTTP server problems. So the HTTP layer should be solid enough not to be the bottleneck — and then we focus on what happens at the GPU.

### What we skipped

The most minimal server would handle one request at a time: a single thread, no concurrency, every request blocks until `generate()` finishes.

```python
@app.post("/v1/chat/completions")
def generate(req: ChatCompletionRequest):
    output = model.generate(**tokenize(req.messages))
    return format_response(output)
```

The problem with starting here is that we already know why it fails: requests queue behind each other, and the GPU sits idle between requests. There is nothing to learn from measuring it. We want failures that reveal something about GPU inference specifically, not failures that any backend engineer would anticipate.

### What we built

The natural fix for a slow synchronous operation behind an async server is a thread pool. `transformers.generate()` is a blocking call. FastAPI and uvicorn are built around async I/O. The correct bridge is `run_in_executor`: offload the blocking work to a thread pool and free the event loop to accept new requests immediately.

```python
executor = ThreadPoolExecutor(max_workers=4)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    loop = asyncio.get_event_loop()
    text, prompt_tokens, completion_tokens = await loop.run_in_executor(
        executor, _run_inference, request
    )
    return ChatCompletionResponse(...)
```

The HTTP layer is fully async — uvicorn accepts new connections without blocking. The thread pool allows multiple `generate()` calls to be in-flight simultaneously. Let's see what happens when we put this under load.

**How the pieces wire together.** The stack has four layers:

- **uvicorn** runs the ASGI server, handling raw HTTP connections
- **FastAPI** sits on top, routing requests, parsing JSON, and validating request/response schemas
- **HuggingFace transformers** loads the model and tokenizer from the Hub, and runs `generate()`
- **PyTorch** is what transformers calls into for the actual GPU computation — matrix multiplications, attention, softmax

The full model loading happens once at server startup. At request time, the hot path is: decode the JSON body → apply the chat template → tokenize → call `model.generate()` → decode the output tokens → return JSON.

**OpenAI-compatible interface.** The endpoint follows the OpenAI chat completions schema: `POST /v1/chat/completions`, a `messages` array in the request body, and a `usage` object in the response with token counts. This is not just convenience — every benchmark tool, client library, and evaluation harness in the ecosystem speaks this interface. Using it means we spend no time writing adapters. Every tool we use in later posts works against our server without modification.

**In-memory telemetry.** The server records the start and end time and token counts for every request, and samples GPU utilization from `nvidia-smi` every 10 seconds. After each benchmark run, `GET /telemetry` returns all of it as JSON. This gives us server-side timing (inference only, no network) alongside GPU utilization on the same time axis, without needing any external monitoring infrastructure. Post 5 is specifically about observability and proper instrumentation; for now, this is sufficient.

The full server is about 160 lines and lives in `src/server.py` on the `01-naive` branch.

---

## Benchmarking

We want to send a realistic load at the server and measure how it behaves. Three questions shape how we do that: which prompts do we send, what do we measure, and how do we collect the measurements.

### Which prompts

We use the **ShareGPT V3 dataset** — 94,145 real conversations scraped from ChatGPT, covering code, reasoning, creative writing, factual questions, and everything in between. From each conversation, we use only the first human turn as the prompt.

Why real prompts instead of synthetic ones? Because prompt length has a direct effect on how long inference takes, and prompts with uniform length make the server look better than it is. ShareGPT's first turns range from a few words to several paragraphs. That variance is what exercises a real server.

Why first-turn only? Keeping prompts as isolated single turns makes token counts predictable and results comparable across posts. Multi-turn conversations — where each request includes the full accumulated history — create a different performance profile because the input grows with every exchange. That difference matters, and Post 2 introduces a `--conversation-mode` benchmark flag to explore it. For now, single-turn gives us a clean baseline.

We sample 200 conversations with a fixed random seed, so the same 200 prompts are used at every concurrency level and in every post. The numbers are reproducible.

### What we measure

**Throughput** — completion tokens generated per second across all requests. This is the primary efficiency signal. A server that processes 36 tok/s at concurrency 1 and still 36 tok/s at concurrency 10 is not getting more efficient with load — it is just making each user wait longer.

**Latency** — end-to-end request time from the client's perspective. We report p50 (the median user's experience), p90, and p99 (the tail). Under a queuing model, p99 diverges from p50 quickly as concurrency grows — that divergence is what tells the story.

**GPU utilization** — percentage of time the GPU's compute units are active, and how much VRAM is consumed. This is the number that explains throughput. A GPU at 30% utilization with flat throughput tells a different story than one at 95%.

### How we collect it

The benchmark script (`benchmarks/benchmark.py`) runs each concurrency level sequentially. Before each run, it calls `POST /telemetry/reset` on the server to clear accumulated data and reset the clock. After all requests complete, it calls `GET /telemetry` to fetch the server's records — per-request timing from inside the server (no network overhead) and the GPU utilization samples. This is combined with the client-side timing (which includes the network round trip through the RunPod proxy) and saved to a JSON file.

Each concurrency level gets its own file: `benchmarks/results/post-01/concurrency-1.json` through `concurrency-20.json`. The raw per-request records and GPU samples are included alongside the summary stats, so we have the time series data needed to generate graphs later.

---

## Results

Hardware: RTX 4090 (24 GB VRAM). Model: Llama 3.1 8B Instruct, loaded in bf16. 200 prompts from ShareGPT, max 256 output tokens per request. Concurrency levels: 1, 5, 10, 20.

| Concurrency | Throughput (tok/s) | p50 latency | p90 latency | p99 latency | Failed |
|-------------|-------------------|-------------|-------------|-------------|--------|
| 1 | 36.7 | 6.8s | 7.4s | 8.0s | 0/200 |
| 5 | 37.3 | 29.9s | 41.4s | 52.6s | 0/200 |
| 10 | 36.7 | 59.8s | 80.5s | 83.2s | 0/200 |
| 20 | 7.4 | 115.5s | 124.3s | 124.8s | **149/200** |

GPU utilization, sampled every 10 seconds by the server:

| Concurrency | Mean GPU util | VRAM used |
|-------------|--------------|-----------|
| 1 | 67% | 16.9 GB |
| 5 | 72% | 17.2 GB |
| 10 | 71% | 17.2 GB |
| 20 | 71% | 17.2 GB |

---

## Findings

### Throughput does not improve with concurrency

36–37 tok/s at concurrency 1, 5, and 10. Adding four more concurrent users does nothing to the total work the server gets done. The thread pool is doing its job — multiple requests are in-flight at the HTTP layer — but something at a lower level is serializing them.

That something is the forward pass.

Recall what happens for each token generated: the current sequence is multiplied through dozens of weight matrices to produce the next token probability. The GPU executes this as a single, massively parallel matrix operation — thousands of tiny computations running simultaneously on its cores. But "parallel" here means across the elements of the computation, not across multiple independent requests. By default, `transformers.generate()` runs one sequence at a time. Each call holds the GPU until it finishes. The thread pool queues additional requests, but they wait — the GPU sees them sequentially regardless of how many threads are running.

This is the core mismatch between how we think about concurrency in backend services and how GPU inference actually works. Adding threads to a database-backed service increases parallelism because the database can serve multiple queries at once. Adding threads to a `transformers.generate()` call increases the queue length, not the GPU throughput.

**How far are we from the hardware limit?** The RTX 4090 has 1008 GB/s of memory bandwidth. During the decode phase — generating one token at a time — the GPU reads the full weight matrix on every forward pass. For Llama 3.1 8B in bf16 (16 GB of weights), that sets a hard ceiling:

```
1008 GB/s ÷ 16 GB = ~63 tok/s per sequence
```

We measured 36–37 tok/s, which is about 58% of the theoretical maximum for a single sequence. The gap comes from Python overhead, memory allocation, and unoptimized attention kernels — none of which are visible in the throughput number but all of which eat into it.

63 tok/s is the ceiling for *one sequence at a time*. Batching multiple sequences into a single forward pass does not change the memory reads — the weights are the same — but it amortizes them across more output tokens. In theory, a well-batched system on this hardware can produce several hundred tokens per second total. Closing the gap between where we are and what the hardware can actually do is the thread that runs through Posts 2 and 3.

### Latency scales linearly with concurrency

p50 at concurrency 1 is 6.8 seconds. At concurrency 5 it is 29.9 seconds — roughly 5×. At concurrency 10 it is 59.8 seconds — roughly 10×.

This is the behavior of a queue, not a parallel system. Each request waits for every request ahead of it to complete before the GPU starts on it. If each request takes ~7 seconds and there are 9 ahead of you, you wait ~63 seconds before your generation even begins.

The p99 diverges further than p50 because prompt lengths vary. Longer prompts take longer to process — the prefill phase (running the input through the model to build up the context) scales with input length. Requests queued behind a long-prompt request wait proportionally longer, pushing tail latency out.

### Concurrency 20 breaks hard

149 of 200 requests hit the 300-second client timeout. At concurrency 20, with ~7 seconds per request serialised through the GPU, a request near the back of the queue waits ~140 seconds before generation even starts. Variable prompt lengths push the actual wait well past 300 for most of them.

The server did not crash. It returned timeouts because clients gave up, not because the server ran out of resources or threw exceptions. From the server's perspective, it was processing one request at a time perfectly well. The queue just grew faster than it could drain.

### The GPU is busy but not used well

GPU utilization sits at 67–72% across all concurrency levels. This is higher than the intuitive expectation — you might expect the GPU to show a sawtooth pattern: high during generation, low during the gap between requests. The gap turns out to be smaller than expected because tokenization, KV cache setup, and tensor allocation are faster than generation time for 256-token outputs.

But 67–72% with flat throughput is not a good result. The GPU is spending roughly a third of its time on per-request overhead that is paid separately for each request: setting up the attention cache, allocating output buffers, copying the next input to device. If multiple requests could be processed in a single forward pass, this overhead would be paid once per batch instead of once per request, and the useful compute fraction would increase.

The remaining headroom — that 30% — is what batching exploits. Post 2 builds the batching layer and measures what it actually recovers.

---

## What this leaves open

The thread pool approach is correct for services where the bottleneck is I/O-bound: waiting on a database, waiting on a network call, waiting on a filesystem. The event loop frees up, another request is accepted, and the system stays busy. Throughput scales with concurrency up to the I/O limit.

GPU inference is not I/O-bound in that sense. The bottleneck is compute-bound on a single device that executes one forward pass at a time. More threads does not give you more forward passes per second — it gives you a longer queue.

The fix is not more threads. It is a bigger batch. A forward pass processes a *batch* of sequences simultaneously — by default that batch is size 1. If we collect multiple requests and send them through the model together, we get multiple responses per forward pass. The per-request overhead is amortized. Throughput improves.

That is static batching. It helps — and it introduces a failure mode that is more interesting than this one.

*→ Post 2: Why your GPU is bored: batching, KV cache, and the memory wall*
