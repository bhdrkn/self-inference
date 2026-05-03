# Post 2 Notes — Why your GPU is bored: batching, KV cache, and the memory wall

## Narrative spine

Post 1 ended with: concurrency doesn't help — the GPU serializes requests, throughput stays flat at ~37 tok/s, latency grows linearly with concurrency.

Post 2 answer: feed the GPU multiple sequences at once (batching). Throughput improves — but the fix introduces new problems that motivate Post 3.

### The three-post arc

1. **Post 1** — no batching → GPU idles between requests → throughput capped at ~37 tok/s
2. **Post 2** — static batching → GPU stays busy → throughput improves → but: KV caches for the whole batch must fit in VRAM simultaneously (memory wall) + all sequences padded to longest (HOL blocking)
3. **Post 3** — vLLM's PagedAttention → KV cache memory as non-contiguous pages → no over-allocation, no fragmentation → continuous batching solves HOL blocking

---

## Post structure

### 1. Introduction
Pick up from Post 1: concurrency doesn't improve throughput, latency scales linearly. This is the same behaviour as a queue with 1 consumer and N producers — the GPU is the consumer, users are producers. The natural next step in queuing systems is batching: instead of sending N small messages one-by-one, pack them into one. Apply the same idea to GPU inference.

### 2. Implementation
Explain the changes to `src/server.py`: a request queue + batching loop that collects up to `BATCH_SIZE` requests, pads to the longest, runs `model.generate()` once. Walk through the code.

### 3. Benchmark — throughput improvement
Run the same benchmark as Post 1 (concurrency 1, 5, 10, 20). Show the throughput increase vs Post 1 baseline. GPU utilization should climb.

### 4. Potential problems
Introduce problems as they emerge from the design — explain each concept at the point it becomes relevant:

- **Memory wall**: KV caches for all sequences in the batch must fit in VRAM simultaneously. Memory pressure = batch_size × sequence_length. Explain KV cache here: what it is, why it exists, why it dominates memory. Show the ceiling by increasing batch size until VRAM fills.
- **Padding waste**: all sequences padded to the longest in the batch — wasted compute on padding tokens. Hard to observe directly without a GPU profiler (nsight/nvtx). Mention why we can't benchmark it, explain it conceptually.
- **HOL blocking**: introduce the concept when we observe it. Like a batch in a message queue — short messages wait for the longest before the batch is acked. Short sequences wait for the longest in their batch before returning. Explain prefill vs decode here as needed.

### 5. Benchmark — demonstrating the problems
- **Memory wall**: ramp up batch size, show VRAM usage, show the point it breaks.
- **HOL blocking**: mixed short/long workload. Show short-request p99 latency is terrible even when overall throughput looks fine.
- **Padding waste**: explain why this can't be demonstrated without a profiler. Note it as a known cost.

### 6. What's next
Static batching is the right instinct but the wrong implementation. The problems (memory fragmentation, HOL blocking) all stem from treating a batch as an atomic unit. Post 3: vLLM solves both with PagedAttention (memory) and continuous batching (HOL).

---

## Key analogies

### Batch in a queue (intro)
Static batching in inference is structurally identical to batching in message queues:
- N messages packed together → downstream processes atomically → short messages wait for the longest before ack
- N sequences in one `model.generate()` call → GPU processes together → short sequences wait for longest before returning

Post 1 already framed GPU serialization in queue terms. Continue that thread here.

### Streaming → continuous batching
Kafka-style streaming solves HOL blocking by processing messages as they complete rather than waiting for the whole batch. Continuous batching (Post 3) does the same for inference: swap finished sequences out immediately, swap new ones in.

---

## What to build

- `src/server.py` — static batching: request queue + batching loop, fixed batch size, pad to longest, single `model.generate()` call
- `benchmarks/benchmark.py` — mixed short/long workload mode to demonstrate HOL blocking
- Batch size sweep to show memory wall

## Q&A — questions that came up during writing

### Does batching introduce privacy concerns between users?

No cross-contamination at the model level. Attention is computed within each sequence — sequence 5 only attends to its own tokens, enforced by the attention mask. User 1's prompt is physically present in the same batch matrix in VRAM but has zero mathematical influence on User 5's output. KV caches are also per-sequence; there is no shared state.

The legitimate privacy concern is narrower: all prompts in a batch exist simultaneously in GPU VRAM. A memory-level attack on the GPU host could read other users' tokens. That is an infrastructure concern, not a model concern.

### Does batching affect output quality or increase hallucinations?

No. Batching is mathematically equivalent to running each sequence individually. Padding tokens are masked out of attention, sampling is applied per-sequence, and the model weights are fixed. Hallucination is a function of model weights and prompt — the model doesn't know other sequences are being processed.

Small caveat: GPU floating point is not strictly deterministic. Batched matrix multiplications can differ from sequential ones by ~1e-5 in logit values — well below the threshold of changing which token gets sampled. Any output differences are indistinguishable from normal temperature-induced non-determinism.

### Where is the attention mask set?

Automatically by the tokenizer and model — nothing to set manually beyond `tokenizer.padding_side = "left"`.

`tokenizer(texts, padding=True)` returns both `input_ids` and `attention_mask` — a 0/1 matrix where `1` = real token, `0` = padding. The model's attention layers zero out scores at any position where the mask is `0`. In the server, `inputs.to(device)` moves both tensors to GPU and `model.generate(**inputs)` unpacks them both. The mask travels with the input automatically.

The one manual step: `padding_side = "left"` must be set before tokenizing. The tokenizer still generates a correct mask without it — but the padding lands on the right, and generation appends new tokens into the middle of sequences instead of at the end.

---

## Decisions made

- **Conversation mode**: not in this post. Conversation history doesn't add anything to the static batching story. Introduce it in Post 3 where prefix caching and KV cache reuse across turns becomes relevant.
- **Batch collection**: fixed size (wait for N requests), not timeout-based. Simpler to explain and reason about.
- **Quality regression check**: batching with correct attention masks produces identical outputs to single-request inference — correct by construction. No need for a dedicated benchmark; note it briefly and move on.
- **Padding waste benchmark**: not feasible without GPU profiler. Explain conceptually, note the limitation, skip the benchmark.
