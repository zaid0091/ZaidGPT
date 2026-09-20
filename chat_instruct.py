"""
Terminal CLI Assistant for ChatGPT Local Engine.
Optimized for high-throughput multi-threaded CPU inference.
"""

import os
import sys

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

# Multi-threaded CPU performance
num_threads = os.cpu_count() or 4
torch.set_num_threads(num_threads)

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def chat(model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 65)
    print("ChatGPT Local Terminal Assistant")
    print(f"[*] Engine: {model_id} | CPU Threads: {num_threads} | Device: {device.upper()}")
    print("Type your message and press Enter. Type 'exit' or 'quit' to quit.")
    print("=" * 65 + "\n")

    print(f"[*] Loading {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            dtype=torch.float32,
        )
        if device == "cpu":
            model.to("cpu")
        model.eval()
        print("[OK] Assistant Engine is online and ready!\n")
    except Exception as e:
        print(f"[!] Error loading model {model_id}: {e}")
        return

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI assistant.\n"
                "Follow these rules for every response:\n"
                "1. Think step-by-step before answering complex questions.\n"
                "2. Structure your answers with clear headings, bullet points, and clean markdown code blocks.\n"
                "3. Be direct, factual, and concise without unnecessary fluff."
            ),
        }
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if user_input.lower() in ["clear", "reset"]:
                messages = [messages[0]]
                print("[*] Conversation memory cleared.")
                continue

            messages.append({"role": "user", "content": user_input})

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            print("\nChatGPT: ", end="", flush=True)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": response_text.strip()})

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Have a great day!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChatGPT Terminal Chat")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Model ID",
    )
    args = parser.parse_args()
    chat(model_id=args.model)
