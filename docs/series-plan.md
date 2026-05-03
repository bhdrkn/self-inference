# Series plan

**Status as of:** 2 May 2026
**Active post:** Post 1 (not yet started)
**Cadence:** one post every 2 weeks
**Target completion:** end of July 2026

## Overall arc

Six posts, each motivated by a problem exposed in the previous one. The narrative spine matters as much as the technical content — each post should end with a question that the next one answers.

## Post 1 — Running an LLM at home is easy. Serving one is not.

**Branch:** `01-naive`
**Estimated effort:** ~12 hours
**Status:** not started

### Goal
Set up the worst plausible inference server. Show that running a model and serving a model are different problems. Make the reader feel the gap.

### Build
- Pull Llama 3.1 8B Instruct from Hugging Face.
- Serve it behind a FastAPI endpoint using raw `transformers.generate()`.
- One request at a time. No batching. No streaming.
- Measure: latency at concurrency 1, 5, 10, 20.
- Measure: GPU utilization during a load test.

### Expected findings
- Latency is fine at concurrency 1.
- Throughput collapses past concurrency 1 (requests queue, GPU sits idle between batches).
- Memory blows up unpredictably with longer sequences.
- p99 latency under load is grotesque.

### Hook for Post 2
"Why is the GPU at 30% utilization when the system is overwhelmed?"

---

## Post 2 — Why your GPU is bored: batching, KV cache, and the memory wall

**Branch:** `02-batching`
**Estimated effort:** ~15 hours
**Status:** planned

### Goal
The conceptual post. Earn credibility with a senior-engineer audience by being precise about what's actually happening on the GPU.

### Concepts to nail
- Prefill vs. decode phases and why they have different bottlenecks.
- The KV cache: what it is, why it dominates memory, how it grows.
- Why decode is memory-bandwidth bound, not compute bound.
- Why naive batching helps, and what breaks (head-of-line blocking with variable lengths).

### Build
- Implement static batching on top of `transformers`.
- Compare throughput vs. Post 1 at the same concurrency levels.
- Construct a benchmark that demonstrates HOL blocking (mix of short and long requests).
- **Response quality regression check**: run the same prompt solo and batched, confirm outputs match. Batching with wrong attention masks can silently corrupt outputs — 200 OK with plausible-looking gibberish. This is the one place in the series where correctness needs a guard.

### Hook for Post 3
"Static batching helps but creates a new problem. Someone already solved it."

### Conversation thread (starts here)
Head-of-line blocking is most visible with conversations: a long conversation holds up short ones in the same batch because we pad to the longest sequence in the batch. Introduce a `--conversation-mode` flag in the benchmark that sends multi-turn conversations (accumulated history across turns) instead of independent first turns. Run both modes: single-turn (comparable to Post 1) and conversation mode (reveals HOL blocking). This flag carries forward through Posts 3 and 4.

---

## Post 3 — Continuous batching and PagedAttention: how vLLM actually works

**Branch:** `03-vllm`
**Estimated effort:** ~17 hours (could expand)
**Status:** planned

### Goal
The post that signals the author looks under the hood. Don't just use vLLM — read its scheduler and block manager and explain them.

### Read
- vLLM's `core/scheduler.py` and `core/block_manager.py` (or current equivalents).
- The PagedAttention paper.
- The "Efficiently Scaling Transformer Inference" paper (Pope et al.) for context.

### Build
- Replace the Post 2 setup with vLLM.
- Re-run the same benchmarks (single-turn and conversation mode). Show the throughput delta.
- Write a short walkthrough of what the scheduler does on each step.
- **Conversation failure demonstration**: show two concrete failures on the naive/batching stack that vLLM fixes:
  1. Long conversations OOM — the full KV cache for every active sequence stays in memory simultaneously
  2. Every turn re-computes the entire conversation history from scratch — no prefix caching means quadratic cost as conversation grows
  Show vLLM's PagedAttention handling both. This is the "aha" moment for the KV cache story.

### Hook for Post 4
"One server is fine. What happens when one server isn't enough?"

---

## Post 4 — Routing inference: when one GPU isn't enough

**Branch:** `04-routing`
**Estimated effort:** ~18 hours
**Status:** planned

### Goal
The post most directly relevant to the Anthropic Inference role. Build a router that does something smarter than round-robin and explain why it matters.

### Build
- Two vLLM instances on separate GPUs.
- A router in front (start with Python; consider Rust as a stretch).
- Implement at least two routing strategies:
  - Round-robin (baseline)
  - Prefix-aware (route requests with shared prefixes to the same replica to maximize KV cache reuse)
  - Optional: queue-depth-aware
- Benchmark: measure cache hit rate and end-to-end latency for each strategy under a realistic prompt distribution.
- **Conversation thread payoff**: with prefix caching working within a single vLLM instance (Post 3), show that it breaks across instances — routing the same conversation to different replicas means each one builds its own KV cache independently, negating Post 3's gains. Prefix-aware routing fixes this by pinning conversations to the same replica. Run the benchmark in conversation mode to make this visible.

### Notes on Rust
Don't gate the post on Rust fluency. Get the Python router working first. If Rust adds clarity (latency budget, predictable memory), rewrite. If it adds friction, ship Python and write a follow-up.

### Hook for Post 5
"The router works. How do we know it's working? How do we know when it's broken?"

---

## Post 5 — Observability for inference: what to measure and why

**Branch:** `05-observability`
**Estimated effort:** ~13 hours
**Status:** planned

### Goal
Anyone can add Prometheus. The question is what to plot and why it matters.

### Concepts to cover
- Why p99 is misleading when request costs vary by 100x.
- TTFT (time to first token) vs. ITL (inter-token latency) vs. end-to-end.
- KV cache hit rate as a routing-quality signal.
- Tail latency under load and how to think about it.
- What you cannot tell from metrics alone (and what you'd need traces for).

### Build
- Instrument the Post 4 stack with Prometheus + Grafana (or OpenTelemetry, decide closer to the time).
- Generate load with a realistic mix of prompt lengths.
- Show what the metrics reveal — and what they hide.

### Hook for Post 6
"We've built a small inference platform. What did we get wrong? What didn't we touch?"

---

## Post 6 — What I'd do differently: a retrospective

**Branch:** `06-retrospective`
**Estimated effort:** ~11 hours
**Status:** planned

### Goal
Honest reflection. The post that goes in the Anthropic application.

### Cover
- What surprised me along the way.
- Where Posts 1–3 had subtle errors that Posts 4–5 forced me to confront.
- What I deliberately didn't touch (multi-node tensor parallelism, speculative decoding, quantization tradeoffs at scale, accelerator-level optimization) and why those matter.
- What this exercise revealed about the gap between application-layer distributed systems and ML infrastructure — and how much of that gap is bridgeable with a few weeks of focused work vs. how much takes years.

This is the post the application links to. It must be honest, specific, and free of bluffing.

---

## Risks and watchlist

- **Scope creep within posts.** Each post answers one question. "Just one more benchmark" is the failure mode.
- **Perfectionism on writing.** Ship at "good enough," iterate on later posts. Bad first drafts beat unfinished perfect ones.
- **Work crunches.** Build buffer: aim to finish each post 3 days before the public commit date.
- **Rewriting earlier posts after later realizations.** Don't. Note it for Post 6.
