"""
Benchmark script for the self-inference series.

Sends N requests sampled from the ShareGPT dataset to a running inference
server at configurable concurrency levels, measures latency and throughput,
and saves results as JSON.

Used across all posts — the server changes (naive → batching → vLLM),
the benchmark stays the same.

Usage:
    python benchmarks/benchmark.py \
        --host localhost --port 8000 \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --dataset benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 200 \
        --concurrency 1 5 10 20 \
        --output benchmarks/results/post-01/results.json
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
        # Need at least one human turn followed by a GPT turn
        if len(turns) < 2:
            continue
        if turns[0].get("from") not in ("human", "user"):
            continue

        # Use the first human turn as the prompt
        human_turn = turns[0]["value"]
        gpt_turn = turns[1]["value"] if turns[1].get("from") in ("gpt", "assistant") else ""

        if not human_turn or not gpt_turn:
            continue

        prompts.append(Prompt(
            messages=[{"role": "user", "content": human_turn}],
            expected_output_tokens=len(gpt_turn.split()),  # rough word count as proxy
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
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    success: bool
    error: str = ""


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: Prompt,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": prompt.messages,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.json()
            latency = time.perf_counter() - t0
            if resp.status != 200:
                return RequestResult(0, 0, latency, False, str(body))
            usage = body.get("usage", {})
            return RequestResult(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_s=latency,
                success=True,
            )
    except Exception as e:
        return RequestResult(0, 0, time.perf_counter() - t0, False, str(e))


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------

async def run_concurrency_level(
    url: str,
    model: str,
    prompts: list[Prompt],
    concurrency: int,
    max_tokens: int,
) -> dict:
    """Send all prompts with a fixed concurrency limit. Returns summary dict."""
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async def bounded(prompt):
        async with semaphore:
            return await send_request(session, url, model, prompt, max_tokens)

    t_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[bounded(p) for p in prompts])
    total_time = time.perf_counter() - t_start

    successful = [r for r in results if r.success]
    failed = len(results) - len(successful)

    if not successful:
        print(f"  concurrency={concurrency}: all {len(results)} requests failed")
        return {"concurrency": concurrency, "error": "all requests failed"}

    latencies = [r.latency_s for r in successful]
    completion_tokens = sum(r.completion_tokens for r in successful)
    prompt_tokens = sum(r.prompt_tokens for r in successful)

    summary = {
        "concurrency": concurrency,
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
        f"failed={failed}"
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark an inference server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True, help="Model name to send in requests")
    parser.add_argument("--dataset", required=True, help="Path to ShareGPT JSON file")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=256, help="max_tokens per request")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--output", required=True, help="Path to save JSON results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_prompts} prompts from ShareGPT...")
    prompts = load_sharegpt(args.dataset, args.num_prompts, args.seed)
    print(f"Loaded {len(prompts)} prompts. Target: {url}")
    print(f"Concurrency levels: {args.concurrency}")
    print()

    all_results = []
    for concurrency in args.concurrency:
        print(f"Running concurrency={concurrency}...")
        result = asyncio.run(run_concurrency_level(
            url, args.model, prompts, concurrency, args.max_tokens
        ))
        all_results.append(result)

    output = {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
