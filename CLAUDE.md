# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Project context

This is a personal learning project: a six-part blog series titled **"Self-Hosting Inference: A Distributed Systems Engineer's Path Into AI Infrastructure."**.

The point of this repo is not to build production inference infrastructure. It is to build a progressively harder inference stack at home in order to **understand the problems**, then write about what was learned. Code quality matters; production-readiness does not. Optimize for clarity and demonstrable understanding, not robustness.

## Audience

Engineers with backend/distributed systems experience breaking into AI infra. Assume the reader has shipped services under load but may not know LLM-specific concepts — explain those in context as they come up. Skip general "intro to LLMs" framing. Focus on substance.

## The six posts

1. **Naive serving** — FastAPI + raw `transformers`. Make it fall over under load. Show why.
2. **Batching, KV cache, the memory wall** — implement static batching by hand. Show its failure mode.
3. **vLLM internals** — switch to vLLM, read the scheduler/block manager source, explain it.
4. **Multi-instance routing** — two vLLM instances behind a custom router. Smarter than round-robin.
5. **Observability** — TTFT, ITL, p99 under variable request cost. What to plot and why.
6. **Retrospective** — what surprised me, what I got wrong, what I didn't touch.

Current status and detailed plan: see `docs/series-plan.md`.

## Repo conventions

- **Python 3.11+**, dependencies managed with `uv`. No `requirements.txt`; use `pyproject.toml`. Never use `pip` directly — add dependencies via `pyproject.toml` and `uv sync`.
- **One branch per post**: `01-naive`, `02-batching`, etc. `master` is the default branch (not `main`) and always reflects the latest completed post.
- **Code lives in `src/`**, benchmarks in `benchmarks/`, helper scripts in `scripts/`.
- **Posts live in `posts/`** as markdown. Notes and drafts live in `docs/posts/`.
- **Rust is planned for Post 4 only** (the router). Don't introduce Rust before then. If Rust feels like a stretch when we get there, write Post 4's router in Python first and consider Rust as a follow-up.
- **No production-grade abstractions.** Don't add config systems, plugin architectures, or framework wrappers unless a post specifically motivates them. The reader should be able to read the code top-to-bottom in 20 minutes.
- **Benchmarks must be reproducible.** Every benchmark gets a script in `benchmarks/` and writes raw output to `benchmarks/results/`. Numbers in posts cite the script that produced them.

## Writing style

- **No competence-signaling language.** Don't describe the target audience as "senior engineers", "competent engineers", or similar. The posts assume distributed systems experience without flattering the reader. Describe what the reader knows (e.g. "has shipped services under load") not what rank they hold.

## Working style

- **Be direct.** The author prefers honesty over reassurance. Push back on bad ideas. Name when something is scope creep, perfectionism, or premature optimization.
- **Don't over-engineer.** If a post can be served by 100 lines of Python, don't write 400.
- **Document the surprises.** When a benchmark or experiment produces an unexpected result, write it down in `docs/posts/post-N-notes.md` immediately. Those surprises are the post's most valuable content.
- **No fabricated numbers.** If a benchmark wasn't run, don't write what it "would" show. Run it.
- **Honest gaps.** When the author lacks experience in something (e.g., accelerator-level optimization, multi-node tensor parallelism), name it rather than bluffing. The retrospective post depends on this.

## Server lifecycle

Always use the scripts in `scripts/` for server start/stop — never start `src/server.py` directly with `python` or `&`. The start script writes a PID file that the stop script depends on.

```bash
# Start (env vars set before the command, not inside the script)
MOCK_MODE=true BATCH_SIZE=4 scripts/start-local-server.sh

# Stop
scripts/stop-local-server.sh
```

If the server was accidentally started outside the script, `lsof -ti :8000 | xargs kill` cleans it up — but fix the root cause, don't make a habit of it.

### Two local testing modes

**Mock mode** (`MOCK_MODE=true`): skips model loading entirely, sleeps for `MOCK_LATENCY` seconds per request. Use this first — fast startup, no GPU/CPU required, good for verifying server logic, batching behaviour, and telemetry correctness.

**Local 1B model** (default, CPU): loads `meta-llama/Llama-3.2-1B-Instruct` on CPU. Slow inference but real model outputs. Use this after mock mode passes, to verify the full pipeline end-to-end before deploying to RunPod.

Always test in this order: mock → local 1B → RunPod. Don't skip to RunPod to save time — it costs money and obscures whether a bug is in the server logic or the GPU environment.

## What to do when starting a session

1. Read `docs/series-plan.md` to see current post status.
2. Read the active post's notes file (`docs/posts/post-N-notes.md`) for context on what's been tried.
3. Confirm the current branch matches the post being worked on.
4. Ask before introducing new dependencies, new directories, or new architectural patterns.

## What to skip

- Don't add CI, pre-commit hooks, or tooling beyond `uv` and `ruff` unless the author asks.
- Don't write tests unless a specific post calls for them. This is a learning artifact, not a production codebase.
- Don't add Docker or Kubernetes manifests. RunPod handles the GPU; the code runs locally or on a single rented instance.
- Don't refactor "for elegance." Each branch should reflect what was understood at the time.
