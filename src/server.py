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

model = None
tokenizer = None
device = None
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


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

    yield


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
    if MOCK_MODE:
        time.sleep(MOCK_LATENCY)
        text = "mock token " * request.max_tokens
        return text, 0, request.max_tokens

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
    prompt_len = inputs.input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion_tokens = output.shape[-1] - prompt_len
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return text, prompt_len, completion_tokens


# --- Endpoint ---

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
