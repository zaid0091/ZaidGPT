"""
FastAPI Streaming Server for ChatGPT Local Assistant.
Optimized for multi-threaded, high-throughput CPU inference with KV-caching and zero latency.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict
from threading import Thread

# Redirect HuggingFace cache strictly to D: drive
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "huggingface"))
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

# Enable optimal multi-threaded CPU acceleration (8 threads optimal for Intel i7-1355U)
num_threads = min(os.cpu_count() or 4, 8)
torch.set_num_threads(num_threads)
try:
    torch.set_num_interop_threads(num_threads)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Initialize FastAPI App
app = FastAPI(title="ChatGPT Local Assistant", version="2.0")

# Mount Static Files
WEB_DIR = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Singleton Engine
ENGINE = {}


def get_engine():
    if "model" not in ENGINE:
        print(f"[*] Loading Neural Engine ({MODEL_ID}) in bfloat16 with {num_threads} CPU threads...")
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        ENGINE["model"] = model
        ENGINE["tokenizer"] = tokenizer
        print("[OK] Neural Engine is online and accelerated!")
    return ENGINE["model"], ENGINE["tokenizer"]


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]


@app.on_event("startup")
async def startup_warmup():
    print("[*] Pre-warming Neural Engine...")
    model, tokenizer = get_engine()
    # 1-token warm up pass to prime OpenMP threads and CPU cache
    try:
        inputs = tokenizer("Hello", return_tensors="pt")
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=1, use_cache=True, do_sample=False)
        print("[OK] Neural Engine pre-warmed for instant response!")
    except Exception as e:
        print("[!] Warmup exception:", e)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = WEB_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "engine": MODEL_ID,
        "threads": num_threads,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cache_dir": CACHE_DIR,
    }


SYSTEM_PROMPT = """You are an expert AI assistant.
Follow these rules for every response:
1. Think step-by-step before answering complex questions.
2. Structure your answers with clear headings, bullet points, and clean markdown code blocks.
3. Be direct, factual, and concise without unnecessary fluff."""


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate_events():
        try:
            model, tokenizer = get_engine()

            # Format multi-turn conversation with system prompt
            formatted_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            # Use sliding context window (last 6 messages) to prevent topic bleeding in multi-turn chats
            raw_history = [m for m in req.messages if m.get("role") in ["user", "assistant"]]
            recent_history = raw_history[-6:] if len(raw_history) > 6 else raw_history

            for m in recent_history:
                formatted_messages.append({"role": m["role"], "content": m["content"]})

            prompt_text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt_text, return_tensors="pt")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # High-speed greedy search with repetition penalty & KV-caching
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=384,
                do_sample=False,
                repetition_penalty=1.15,
                use_cache=True,  # KV-Cache for O(1) step inference
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def run_generation():
                with torch.inference_mode():
                    model.generate(**gen_kwargs)

            def stream_producer():
                try:
                    for new_token in streamer:
                        if new_token:
                            loop.call_soon_threadsafe(queue.put_nowait, new_token)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            thread_gen = Thread(target=run_generation, daemon=True)
            thread_stream = Thread(target=stream_producer, daemon=True)
            thread_gen.start()
            thread_stream.start()

            while True:
                token = await queue.get()
                if token is None:
                    break
                
                # Drain any accumulated tokens in batch for smooth throughput
                chunk = token
                while not queue.empty():
                    next_tok = queue.get_nowait()
                    if next_tok is None:
                        payload = json.dumps({"token": chunk})
                        yield f"data: {payload}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    chunk += next_tok

                payload = json.dumps({"token": chunk})
                yield f"data: {payload}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            err_payload = json.dumps({"token": f"\n\n[Error: {str(e)}]"})
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🚀 Starting ChatGPT Assistant Server...")
    print(f"⚡ CPU Threads: {num_threads} | Model: {MODEL_ID}")
    print("🌐 Open in your browser: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
