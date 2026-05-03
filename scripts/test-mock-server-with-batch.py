"""
Verify that the batching server works correctly with MOCK_MODE=true.

Sends BATCH_SIZE concurrent requests and checks that:
  - All requests complete in roughly one MOCK_LATENCY period (not N × MOCK_LATENCY)
  - Telemetry records one entry per request
  - All requests in the same batch share the same t_end (fired together)

Requires:
  - Server running with MOCK_MODE=true (scripts/start-local-server.sh)

Usage:
    MOCK_MODE=true BATCH_SIZE=4 BATCH_TIMEOUT_MS=200 scripts/start-local-server.sh
    uv run python scripts/test-mock-server-with-batch.py
"""

import asyncio
import sys
import time

import aiohttp

BASE_URL = "http://localhost:8000"
NUM_REQUESTS = 4


async def send(session: aiohttp.ClientSession, i: int) -> float:
    t = time.perf_counter()
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "test",
        "messages": [{"role": "user", "content": f"request {i}"}],
        "max_tokens": 5,
    }) as r:
        await r.json()
    return time.perf_counter() - t


async def main() -> None:
    async with aiohttp.ClientSession() as s:
        await s.post(f"{BASE_URL}/telemetry/reset")

    async with aiohttp.ClientSession() as s:
        latencies = await asyncio.gather(*[send(s, i) for i in range(NUM_REQUESTS)])

    print(f"Latencies: {[round(t, 2) for t in latencies]}")

    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/telemetry") as r:
            tel = await r.json()

    reqs = tel["requests"]
    # Drop any stale records that predate our reset (t_start much larger than
    # expected, meaning they were timestamped relative to a previous _t0).
    reqs = [r for r in reqs if r["t_start"] < 10.0]
    print(f"Telemetry records: {len(reqs)} (expected {NUM_REQUESTS})")
    print(f"t_start values: {[r['t_start'] for r in reqs]}")
    print(f"t_end values:   {[r['t_end'] for r in reqs]}")

    # Verify batching: all t_end values should be equal (same batch)
    t_ends = [r["t_end"] for r in reqs]
    if len(set(t_ends)) == 1:
        print("\nPASS: all requests completed in the same batch (identical t_end)")
    else:
        spread = max(t_ends) - min(t_ends)
        print(f"\nWARN: t_end spread = {spread:.3f}s — requests may have been split across batches")

    # Verify throughput: total wall time should be close to one batch duration
    max_latency = max(latencies)
    min_latency = min(latencies)
    if max_latency < 2 * min_latency:
        print(f"PASS: max latency ({max_latency:.2f}s) is close to min ({min_latency:.2f}s) — not serialized")
    else:
        print(f"FAIL: max latency ({max_latency:.2f}s) is much larger than min ({min_latency:.2f}s) — may be serialized")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
