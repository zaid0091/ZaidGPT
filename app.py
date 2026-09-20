"""
ZaidGPT Full-Stack Web Application Backend.
FastAPI Streaming server supporting SSE token-by-token generation and engine switching.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

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
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# Initialize FastAPI App
app = FastAPI(title="ZaidGPT Web UI", version="2.0")

# Mount Static Files
WEB_DIR = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Global Engine Cache
ENGINE_CACHE = {}


def get_instruct_engine(model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    if "instruct" not in ENGINE_CACHE:
        print(f"[*] Loading Instruct Engine ({model_id})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            dtype=torch.float32,
        )
        model.eval()
        ENGINE_CACHE["instruct"] = {"model": model, "tokenizer": tokenizer, "id": model_id}
        print("[OK] Instruct Engine online!")
    return ENGINE_CACHE["instruct"]


def get_finetuned_engine(model_path: str = "checkpoints/zaidgpt_finetuned"):
    if "finetuned" not in ENGINE_CACHE:
        print(f"[*] Loading Fine-Tuned Engine ({model_path})...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        ENGINE_CACHE["finetuned"] = {"model": model, "tokenizer": tokenizer, "path": model_path}
        print("[OK] Fine-Tuned Engine online!")
    return ENGINE_CACHE["finetuned"]


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    engine: str = "instruct"  # instruct | finetuned | scratch


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = WEB_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cache_dir": CACHE_DIR,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate_events():
        try:
            if req.engine == "instruct":
                engine = get_instruct_engine()
                model = engine["model"]
                tokenizer = engine["tokenizer"]

                # Apply chat template
                formatted_messages = [
                    {
                        "role": "system",
                        "content": "You are ZaidGPT, an expert AI assistant specializing in software engineering, system design, and coding. Provide direct, structured, and helpful answers.",
                    }
                ]
                for m in req.messages:
                    if m.get("role") in ["user", "assistant"]:
                        formatted_messages.append({"role": m["role"], "content": m["content"]})

                prompt_text = tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = tokenizer(prompt_text, return_tensors="pt")
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                gen_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id,
                )

                thread = Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()

                for new_token in streamer:
                    if new_token:
                        payload = json.dumps({"token": new_token})
                        yield f"data: {payload}\n\n"
                        await asyncio.sleep(0.005)

                thread.join()

            elif req.engine == "finetuned":
                engine = get_finetuned_engine()
                model = engine["model"]
                tokenizer = engine["tokenizer"]

                last_user_msg = req.messages[-1]["content"] if req.messages else ""
                prompt = f"User: {last_user_msg}\nAssistant: "
                inputs = tokenizer(prompt, return_tensors="pt")
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                gen_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                )

                thread = Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()

                for new_token in streamer:
                    if new_token:
                        payload = json.dumps({"token": new_token})
                        yield f"data: {payload}\n\n"
                        await asyncio.sleep(0.005)

                thread.join()

            else:
                # Fallback / From scratch
                engine = get_instruct_engine()
                model = engine["model"]
                tokenizer = engine["tokenizer"]
                inputs = tokenizer(req.messages[-1]["content"], return_tensors="pt")
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                thread = Thread(
                    target=model.generate,
                    kwargs=dict(**inputs, streamer=streamer, max_new_tokens=200),
                )
                thread.start()
                for token in streamer:
                    if token:
                        payload = json.dumps({"token": token})
                        yield f"data: {payload}\n\n"
                        await asyncio.sleep(0.005)
                thread.join()

            yield "data: [DONE]\n\n"

        except Exception as e:
            err_payload = json.dumps({"token": f"\n\n[Error: {str(e)}]"})
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🚀 Starting ZaidGPT Full-Stack Web Server...")
    print("🌐 Open in your browser: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
