"""OpenAI-compatible chat API for Meowtrix-0.8B (FastAPI, non-streaming).

    MEOWTRIX_MODEL=exported/meowtrix-0.8b uv run uvicorn serve.app:app --port 8000

Extensions over the vanilla OpenAI schema:
  - ``num_loops``:        per-request recurrent loop override.
  - ``enable_thinking``:  renders the prompt with the <think> block open;
                          the returned message exposes ``reasoning_content``.
  - ``stream``:           accepted but not yet implemented (TODO).
"""

from __future__ import annotations

import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from meowtrix import load_meowtrix_model

MODEL_PATH = os.environ.get("MEOWTRIX_MODEL", "exported/meowtrix-0.8b")
MODEL_ID = "meowtrix-0.8b"
THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)

state: dict[str, Any] = {}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = _device()
    state["device"] = device
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    state["model"] = load_meowtrix_model(MODEL_PATH, torch_dtype=torch.bfloat16).to(device)
    state["model"].eval()
    yield
    state.clear()


app = FastAPI(title="Meowtrix-0.8B API", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = Field(default=512, ge=1, le=8192)
    num_loops: int | None = Field(default=None, ge=1)
    enable_thinking: bool = True
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "meowtrix"}],
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": state.get("device")}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        # TODO: SSE streaming.
        return {"error": "stream=true is not implemented yet"}

    tokenizer = state["tokenizer"]
    model = state["model"]

    text = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=req.enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(state["device"])
    prompt_tokens = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature or None,
        top_p=req.top_p,
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_loops=req.num_loops or model.config.num_loops,
            **{k: v for k, v in gen_kwargs.items() if v is not None},
        )
    completion_ids = out[0][prompt_tokens:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True)

    reasoning, content = None, raw
    m = THINK_RE.match(raw)
    if m:
        reasoning, content = m.group(1).strip(), raw[m.end() :].strip()

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_ids.shape[0]),
            "total_tokens": int(prompt_tokens + completion_ids.shape[0]),
        },
    }
