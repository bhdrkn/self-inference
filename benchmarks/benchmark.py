"""
Benchmark script for the self-inference series.

Sends N requests sampled from the ShareGPT dataset to a running inference
server at configurable concurrency levels, measures latency and throughput,
and saves results as JSON — one file per concurrency level.

GPU utilization and per-request server-side timings are collected via the
server's /telemetry endpoints (reset before each run, fetch after).

Used across all posts — the server changes (naive → batching → vLLM),
the benchmark stays the same.

Usage:
    python benchmarks/benchmark.py \
        --host localhost --port 8000 \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --dataset benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 200 \
        --concurrency 1 5 10 20 \
        --output-dir benchmarks/results/post-01
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    messages: list[dict]
    expected_output_tokens: int  # from ShareGPT ground truth (used for sampling only)


def load_sharegpt(path: str, num_prompts: int, seed: int = 42) -> list[Prompt]:
    """Sample num_prompts conversations from the ShareGPT dataset."""
    with open(path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(data)

    prompts = []
    for convo in data:
        turns = convo.get("conversations", [])
        if len(turns) < 2:
            continue
        if turns[0].get("from") not in ("human", "user"):
            continue

        human_turn = turns[0]["value"]
        gpt_turn = turns[1]["value"] if turns[1].get("from") in ("gpt", "assistant") else ""

        if not human_turn or not gpt_turn:
            continue

        prompts.append(Prompt(
            messages=[{"role": "user", "content": human_turn}],
            expected_output_tokens=len(gpt_turn.split()),
        ))

        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        print(f"Warning: only found {len(prompts)} usable prompts (requested {num_prompts})")

    return prompts


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class RequestRecord:
    t_start: float       # seconds since benchmark start (client clock)
    t_end: float         # seconds since benchmark start (client clock)
    latency_s: float     # end-to-end including network
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: str = ""


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: Prompt,
    max_tokens: int,
    t_run_start: float,
) -> RequestRecord:
    payload = {
        "model": model,
        "messages": prompt.messages,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.json()
            t1 = time.perf_counter()
            latency = t1 - t0
            if resp.status != 200:
                return RequestRecord(
                    round(t0 - t_run_start, 3), round(t1 - t_run_start, 3),
                    round(latency, 3), 0, 0, False, str(body),
                )
            usage = body.get("usage", {})
            return RequestRecord(
                t_start=round(t0 - t_run_start, 3),
                t_end=round(t1 - t_run_start, 3),
                latency_s=round(latency, 3),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                success=True,
            )
    except Exception as e:
        t1 = time.perf_counter()
        return RequestRecord(
            t_start=round(t0 - t_run_start, 3),
            t_end=round(t1 - t_run_start, 3),
            latency_s=round(t1 - t0, 3),
            prompt_tokens=0, completion_tokens=0,
            success=False, error=str(e),
        )


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------

async def reset_telemetry(session: aiohttp.ClientSession, base_url: str) -> None:
    """Clear server-side telemetry and reset its clock before a benchmark run."""
    try:
        async with session.post(f"{base_url}/telemetry/reset") as resp:
            await resp.read()
    except Exception as e:
        print(f"  Warning: could not reset telemetry: {e}")


async def fetch_telemetry(session: aiohttp.ClientSession, base_url: str) -> dict:
    """Fetch server-side request records and GPU samples after a benchmark run."""
    try:
        async with session.get(f"{base_url}/telemetry") as resp:
            return await resp.json()
    except Exception as e:
        print(f"  Warning: could not fetch telemetry: {e}")
        return {"requests": [], "gpu_samples": []}


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------

async def run_concurrency_level(
    base_url: str,
    model: str,
    prompts: list[Prompt],
    concurrency: int,
    max_tokens: int,
) -> dict:
    """Send all prompts at a fixed concurrency limit. Returns result dict."""
    completions_url = f"{base_url}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await reset_telemetry(session, base_url)

        t_run_start = time.perf_counter()

        async def bounded(prompt):
            async with semaphore:
                return await send_request(session, completions_url, model, prompt, max_tokens, t_run_start)

        results = await asyncio.gather(*[bounded(p) for p in prompts])
        total_time = time.perf_counter() - t_run_start

        telemetry = await fetch_telemetry(session, base_url)

    successful = [r for r in results if r.success]
    failed = len(results) - len(successful)

    if not successful:
        print(f"  concurrency={concurrency}: all {len(results)} requests failed")
        return {
            "concurrency": concurrency,
            "summary": {"error": "all requests failed"},
            "client_requests": [asdict(r) for r in results],
            "server_requests": telemetry["requests"],
            "gpu_samples": telemetry["gpu_samples"],
        }

    latencies = [r.latency_s for r in successful]
    completion_tokens = sum(r.completion_tokens for r in successful)
    prompt_tokens = sum(r.prompt_tokens for r in successful)

    summary = {
        "num_prompts": len(prompts),
        "num_successful": len(successful),
        "num_failed": failed,
        "total_time_s": round(total_time, 3),
        "throughput_tokens_per_s": round(completion_tokens / total_time, 2),
        "throughput_requests_per_s": round(len(successful) / total_time, 3),
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_p50_s": round(float(np.percentile(latencies, 50)), 3),
        "latency_p90_s": round(float(np.percentile(latencies, 90)), 3),
        "latency_p99_s": round(float(np.percentile(latencies, 99)), 3),
    }

    print(
        f"  concurrency={concurrency}: "
        f"{summary['throughput_tokens_per_s']} tok/s | "
        f"p50={summary['latency_p50_s']}s p99={summary['latency_p99_s']}s | "
        f"failed={failed} | gpu_samples={len(telemetry['gpu_samples'])}"
    )

    return {
        "concurrency": concurrency,
        "summary": summary,
        "client_requests": [asdict(r) for r in results],
        "server_requests": telemetry["requests"],
        "gpu_samples": telemetry["gpu_samples"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark an inference server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--https", action="store_true", help="Use HTTPS (required for RunPod proxy)")
    parser.add_argument("--model", required=True, help="Model name to send in requests")
    parser.add_argument("--dataset", required=True, help="Path to ShareGPT JSON file")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=256, help="max_tokens per request")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--output-dir", required=True, help="Directory to save per-concurrency JSON results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scheme = "https" if args.https else "http"
    base_url = f"{scheme}://{args.host}:{args.port}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    print(f"Loading {args.num_prompts} prompts from ShareGPT...")
    prompts = load_sharegpt(args.dataset, args.num_prompts, args.seed)
    print(f"Loaded {len(prompts)} prompts. Target: {base_url}")
    print(f"Concurrency levels: {args.concurrency}")
    print()

    for concurrency in args.concurrency:
        print(f"Running concurrency={concurrency}...")
        result = asyncio.run(run_concurrency_level(
            base_url, args.model, prompts, concurrency, args.max_tokens,
        ))

        output = {**meta, **result}
        output_path = output_dir / f"concurrency-{concurrency}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to {output_path}")
        print()


if __name__ == "__main__":
    main()
