"""
Post 1 — naive inference server.

A minimal FastAPI server wrapping transformers.generate(). No batching,
no streaming, no tricks. Requests are handled concurrently via a thread
pool, but the GPU serializes them at the CUDA stream level.

Configuration (environment variables):
    MODEL_NAME      HuggingFace model ID (default: Llama 3.2 1B for local dev)
    MAX_WORKERS     Thread pool size (default: 4)
    MOCK_MODE       Set to "true" to skip model loading and simulate inference
    MOCK_LATENCY    Simulated generation time in seconds when MOCK_MODE=true (default: 2.0)
"""

import asyncio
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; no-op in production where env vars are injected directly

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Default to Llama 3.2 1B for local dev. Override with MODEL_NAME env var.
# Production: MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"
MOCK_LATENCY = float(os.environ.get("MOCK_LATENCY", "2.0"))
GPU_SAMPLE_INTERVAL = float(os.environ.get("GPU_SAMPLE_INTERVAL", "10.0"))

model = None
tokenizer = None
device = None
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# ---------------------------------------------------------------------------
# In-memory telemetry
#
# Stores request records and GPU samples since the last reset.
# benchmark.py calls POST /telemetry/reset before each concurrency run,
# then GET /telemetry after all requests complete to fetch the data.
#
# _t0 is set on reset and used as the reference for all timestamps,
# so request records and GPU samples share the same time axis.
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
        # Use bf16 on GPU (memory-efficient, no quality loss for inference).
        # Fall back to fp32 on CPU — older Intel CPUs have no native bf16 support.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading {MODEL_NAME} on {device} ({dtype}) ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # device_map="auto" loads weights directly onto GPU, avoiding a CPU→GPU copy
        # which can fail silently for large models. Falls back gracefully on CPU-only.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)
        model.eval()
        print(f"Model ready. Device map: {getattr(model, 'hf_device_map', device)}")

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


# --- Inference (runs in thread pool, off the event loop) ---

def _run_inference(request: ChatCompletionRequest) -> tuple[str, int, int]:
    """Returns (generated_text, prompt_tokens, completion_tokens)."""
    t_start = _now()

    if MOCK_MODE:
        time.sleep(MOCK_LATENCY)
        prompt_tokens, completion_tokens = 0, request.max_tokens
        text = "mock token " * completion_tokens
    else:
        import torch

        # apply_chat_template renders the conversation to a string.
        # We tokenize separately so we always get a plain tensor, not a BatchEncoding.
        text = tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Place inputs on the same device as the model's first layer.
        # With device_map="auto", model.device may be "meta" — use hf_device_map instead.
        input_device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(input_device)
        prompt_tokens = inputs.input_ids.shape[-1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_tokens = output.shape[-1] - prompt_tokens
        text = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)

    t_end = _now()
    with _telemetry_lock:
        _telemetry_requests.append({
            "t_start": t_start,
            "t_end": t_end,
            "latency_s": round(t_end - t_start, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    return text, prompt_tokens, completion_tokens


# --- Endpoints ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    loop = asyncio.get_event_loop()
    text, prompt_tokens, completion_tokens = await loop.run_in_executor(
        executor, _run_inference, request
    )
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
    """Return all recorded request and GPU sample data since the last reset."""
    with _telemetry_lock:
        return {
            "requests": list(_telemetry_requests),
            "gpu_samples": list(_telemetry_gpu_samples),
        }


@app.post("/telemetry/reset")
async def reset_telemetry():
    """Clear all telemetry and reset the clock. Call before each benchmark run."""
    global _t0
    with _telemetry_lock:
        _telemetry_requests.clear()
        _telemetry_gpu_samples.clear()
        _t0 = time.perf_counter()
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
