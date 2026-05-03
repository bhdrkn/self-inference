"""
Post 2 — static batching inference server.

Extends the naive Post 1 server with a request queue and a batching loop.
Incoming requests are held in a queue; a background thread collects up to
BATCH_SIZE of them (or fires after BATCH_TIMEOUT_MS, whichever comes first),
pads to the longest sequence in the batch, and runs model.generate() once.

This keeps the GPU fed between requests and improves throughput — at the cost
of head-of-line blocking and padding waste, which we benchmark in Post 2.

Configuration (environment variables):
    MODEL_NAME          HuggingFace model ID (default: Llama 3.2 1B for local dev)
    BATCH_SIZE          Max requests per batch (default: 8)
    BATCH_TIMEOUT_MS    Max wait for a full batch in milliseconds (default: 100)
    MOCK_MODE           Set to "true" to skip model loading and simulate inference
    MOCK_LATENCY        Simulated generation time in seconds when MOCK_MODE=true (default: 2.0)
"""

import asyncio
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
BATCH_TIMEOUT_MS = float(os.environ.get("BATCH_TIMEOUT_MS", "100"))
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"
MOCK_LATENCY = float(os.environ.get("MOCK_LATENCY", "2.0"))
GPU_SAMPLE_INTERVAL = float(os.environ.get("GPU_SAMPLE_INTERVAL", "10.0"))

model = None
tokenizer = None
device = None


# ---------------------------------------------------------------------------
# In-memory telemetry (unchanged from Post 1)
# ---------------------------------------------------------------------------

_telemetry_lock = threading.Lock()
_telemetry_requests: list[dict] = []
_telemetry_gpu_samples: list[dict] = []
_t0: float = time.perf_counter()


def _now() -> float:
    return round(time.perf_counter() - _t0, 3)


async def _gpu_sampler():
    """Background task: sample GPU utilization every GPU_SAMPLE_INTERVAL seconds."""
    while True:
        await asyncio.sleep(GPU_SAMPLE_INTERVAL)
        t = _now()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            sample = {
                "t": t,
                "utilization_pct": int(parts[0]),
                "memory_used_mib": int(parts[1]),
                "memory_free_mib": int(parts[2]),
            }
        except Exception:
            sample = {"t": t, "utilization_pct": 0, "memory_used_mib": 0, "memory_free_mib": 0}

        with _telemetry_lock:
            _telemetry_gpu_samples.append(sample)


# ---------------------------------------------------------------------------
# Request queue and batching loop
#
# Each incoming HTTP request places a _BatchItem on _request_queue and blocks
# on its future. The batching thread drains the queue, groups items into
# batches, runs inference once per batch, then resolves each future.
# ---------------------------------------------------------------------------

@dataclass
class _BatchItem:
    request: "ChatCompletionRequest"
    t_start: float
    future: Future


_request_queue: queue.Queue[_BatchItem] = queue.Queue()


def _collect_batch() -> list[_BatchItem]:
    """
    Collect up to BATCH_SIZE items from the queue.
    Blocks until at least one item arrives, then waits up to BATCH_TIMEOUT_MS
    for more before returning whatever has accumulated.
    """
    timeout_s = BATCH_TIMEOUT_MS / 1000.0
    deadline = None
    batch = []

    while len(batch) < BATCH_SIZE:
        if deadline is None:
            # Block indefinitely for the first item.
            item = _request_queue.get()
            batch.append(item)
            deadline = time.perf_counter() + timeout_s
        else:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = _request_queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break

    return batch


def _run_batch(batch: list[_BatchItem]) -> None:
    """Run inference for a batch of requests and resolve each future."""
    if MOCK_MODE:
        time.sleep(MOCK_LATENCY)
        for item in batch:
            t_end = _now()
            completion_tokens = item.request.max_tokens
            text = "mock token " * completion_tokens
            _record_telemetry(item.t_start, t_end, 0, completion_tokens)
            item.future.set_result((text, 0, completion_tokens))
        return

    import torch

    # Build input texts for all requests in the batch.
    texts = [
        tokenizer.apply_chat_template(
            [m.model_dump() for m in item.request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in batch
    ]

    # Tokenize with left-padding so all sequences align on the right.
    # Left-padding is required for batched generation — the model attends
    # to the rightmost tokens first, so prompt tokens must be flush-right.
    tokenizer.padding_side = "left"
    input_device = next(model.parameters()).device
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(input_device)

    prompt_lengths = inputs.attention_mask.sum(dim=1).tolist()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max(item.request.max_tokens for item in batch),
            pad_token_id=tokenizer.eos_token_id,
        )

    t_end = _now()

    for i, item in enumerate(batch):
        prompt_tokens = int(prompt_lengths[i])
        completion_tokens = outputs[i].shape[-1] - inputs.input_ids.shape[-1]
        text = tokenizer.decode(outputs[i][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        _record_telemetry(item.t_start, t_end, prompt_tokens, completion_tokens)
        item.future.set_result((text, prompt_tokens, completion_tokens))


def _record_telemetry(t_start: float, t_end: float, prompt_tokens: int, completion_tokens: int) -> None:
    with _telemetry_lock:
        _telemetry_requests.append({
            "t_start": t_start,
            "t_end": t_end,
            "latency_s": round(t_end - t_start, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })


def _batching_loop() -> None:
    """Background thread: collect batches and run inference until stopped."""
    while True:
        batch = _collect_batch()
        try:
            _run_batch(batch)
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device

    if MOCK_MODE:
        print("MOCK_MODE=true — skipping model load, inference will sleep for "
              f"{MOCK_LATENCY}s per request")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading {MODEL_NAME} on {device} ({dtype}) ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)
        model.eval()
        print(f"Model ready. Device map: {getattr(model, 'hf_device_map', device)}")

    # Start the batching loop in a background daemon thread.
    t = threading.Thread(target=_batching_loop, daemon=True)
    t.start()
    print(f"Batching loop started (BATCH_SIZE={BATCH_SIZE}, BATCH_TIMEOUT_MS={BATCH_TIMEOUT_MS})")

    sampler = asyncio.create_task(_gpu_sampler())
    yield
    sampler.cancel()
    try:
        await sampler
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


# --- Request / response types (OpenAI chat completions schema) ---

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 256


class ChatCompletionResponse(BaseModel):
    model: str
    choices: list[dict]
    usage: dict


# --- Endpoint ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    t_start = _now()
    future: Future = Future()
    _request_queue.put(_BatchItem(request=request, t_start=t_start, future=future))

    # Await the future without blocking the event loop.
    loop = asyncio.get_event_loop()
    text, prompt_tokens, completion_tokens = await loop.run_in_executor(None, future.result)

    return ChatCompletionResponse(
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@app.get("/telemetry")
async def get_telemetry():
    with _telemetry_lock:
        return {
            "requests": list(_telemetry_requests),
            "gpu_samples": list(_telemetry_gpu_samples),
        }


@app.post("/telemetry/reset")
async def reset_telemetry():
    global _t0
    with _telemetry_lock:
        _telemetry_requests.clear()
        _telemetry_gpu_samples.clear()
        _t0 = time.perf_counter()
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
