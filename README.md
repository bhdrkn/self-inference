# self-inference

> A distributed systems engineer's path into AI infrastructure, learned by building.

This repo accompanies a six-part blog series in which I build a progressively more sophisticated LLM inference stack from scratch — starting with the worst possible serving setup and ending with a small, observable, multi-instance system that routes intelligently between replicas.

Each post is paired with a working implementation in this repo. The goal is not to build production inference infrastructure (you should use vLLM or similar). The goal is to **understand why production inference infrastructure looks the way it does**, by feeling the problems firsthand.

## Who this is for

Senior backend / distributed systems engineers who want to break into AI infrastructure work and are tired of "intro to LLMs" content that stops where the interesting problems start.

If you've ever shipped a high-throughput service and wondered what's actually different about serving language models — this series is for you.

## The series

| # | Post | Status | Branch |
|---|------|--------|--------|
| 1 | Running an LLM at home is easy. Serving one is not. | 🟡 In progress | `01-naive` |
| 2 | Why your GPU is bored: batching, KV cache, and the memory wall | ⚪ Planned | `02-batching` |
| 3 | Continuous batching and PagedAttention: how vLLM actually works | ⚪ Planned | `03-vllm` |
| 4 | Routing inference: when one GPU isn't enough | ⚪ Planned | `04-routing` |
| 5 | Observability for inference: what to measure and why | ⚪ Planned | `05-observability` |
| 6 | What I'd do differently: a retrospective | ⚪ Planned | `06-retrospective` |

Each branch contains the code as it exists at the end of that post. `main` always reflects the latest completed post.

## Reproducing

You will need:

- A GPU. Cheapest option: rent one on RunPod (~$0.34–0.69/hour for an RTX 4090). For Posts 3+ you'll want an A100 40GB occasionally.
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv) for dependency management.
- A Hugging Face account and access token (for pulling open model weights).

Each post's branch has its own README with setup steps and benchmark commands.

## Why I'm writing this

I'm a senior software engineer with a decade of distributed systems experience, currently working on agentic LLM systems. I want to move deeper into the infrastructure side of AI — and the most honest way to learn it is to build it badly first, then better, in public.

If you're on a similar path, follow along. If you spot something wrong, open an issue.

## License

MIT. Use anything here however you like.
