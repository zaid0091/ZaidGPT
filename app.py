import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict
from threading import Thread
from contextlib import asynccontextmanager

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
MODEL_NAME = "Qwen2.5-Coder-7B-Instruct (GGUF Q4_K_M)"

num_threads = min(os.cpu_count() or 4, 8)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Checking Neural Engine...")
    try:
        if os.path.exists(MODEL_PATH):
            llm = get_engine()
            llm.create_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            print("[OK] Neural Engine pre-warmed for instant response!")
        else:
            print(f"[!] Model file not found at {MODEL_PATH}...")
    except Exception as e:
        print("[!] Startup check:", e)
    yield


app = FastAPI(title="ChatGPT Local Assistant", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

ENGINE = {}


def get_engine():
    if "llm" not in ENGINE:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Please ensure the GGUF file exists."
            )
        print(f"[*] Loading GGUF Neural Engine ({os.path.basename(MODEL_PATH)}) with {num_threads} CPU threads...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=num_threads,
            n_batch=512,
            verbose=False,
        )
        ENGINE["llm"] = llm
        print("[OK] GGUF Neural Engine is online and AVX2-accelerated!")
    return ENGINE["llm"]


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = WEB_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    model_ready = os.path.exists(MODEL_PATH)
    return {
        "status": "online" if model_ready else "downloading",
        "engine": MODEL_NAME,
        "threads": num_threads,
        "device": "cpu (AVX2)",
        "model_path": MODEL_PATH,
    }


SYSTEM_PROMPT = """You are an expert AI software engineer, architect, and technical assistant.
Follow these rules for every response:
1. Think step-by-step before answering complex questions.
2. Provide clean, robust, and production-ready code with complete implementations (avoid placeholders or omissions).
3. Structure your answers with clear headings, bullet points, and clean markdown code blocks.
4. Be direct, factual, and concise without unnecessary fluff."""


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate_events():
        try:
            llm = get_engine()

            formatted_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            raw_history = [
                m for m in req.messages
                if m.get("role") in ["user", "assistant"]
                and m.get("content", "").strip()
                and not m.get("content", "").startswith("[Error:")
                and "Connection to the local server was temporarily interrupted" not in m.get("content", "")
            ]
            recent_history = raw_history[-6:] if len(raw_history) > 6 else raw_history

            for m in recent_history:
                formatted_messages.append({"role": m["role"], "content": m["content"]})

            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            is_cancelled = False

            def run_generation():
                try:
                    response_stream = llm.create_chat_completion(
                        messages=formatted_messages,
                        max_tokens=768,
                        temperature=0.6,
                        top_p=0.9,
                        repeat_penalty=1.15,
                        stream=True,
                    )
                    for chunk in response_stream:
                        if is_cancelled:
                            break
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            loop.call_soon_threadsafe(queue.put_nowait, content)
                except Exception as ex:
                    if not is_cancelled:
                        loop.call_soon_threadsafe(queue.put_nowait, f"\n\n[Error: {str(ex)}]")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            thread = Thread(target=run_generation, daemon=True)
            thread.start()

            try:
                while True:
                    token = await queue.get()
                    if token is None:
                        break

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

            except (asyncio.CancelledError, GeneratorExit):
                is_cancelled = True
                return

        except (asyncio.CancelledError, GeneratorExit):
            return
        except Exception as e:
            err_payload = json.dumps({"token": f"\n\n[Error: {str(e)}]"})
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🚀 Starting ChatGPT Assistant Server (GGUF Engine)...")
    print(f"⚡ CPU Threads: {num_threads} | Model: {MODEL_NAME}")
    print("🌐 Open in your browser: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
