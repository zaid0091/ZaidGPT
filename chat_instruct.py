"""
ZaidGPT Supercharged Instruct Assistant (Powered by Qwen2.5-0.5B-Instruct / SmolLM2).
Provides direct, structured, ChatGPT-grade explanations and code on CPU/GPU.
"""

import os
import sys

# Redirect HuggingFace cache to D: drive to prevent C: drive full errors
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "huggingface"))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def chat(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 65)
    print("🤖 ZaidGPT Instruct AI Assistant (ChatGPT-Grade Reasoning & Code)")
    print(f"[*] Engine: {model_id} | Running on: {device.upper()}")
    print("Type your message and press Enter. Type 'exit' or 'quit' to quit.")
    print("=" * 65 + "\n")

    print(f"[*] Loading {model_id} (first time will download ~600MB)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model.to("cpu")
        model.eval()
        print("[✓] ZaidGPT Engine is online and ready!\n")
    except Exception as e:
        print(f"[!] Error loading model {model_id}: {e}")
        return

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [
        {
            "role": "system",
            "content": "You are ZaidGPT, an expert, concise, and helpful AI assistant specializing in software engineering, programming, system design, and general problem solving. Provide complete, accurate, structured, and helpful answers.",
        }
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye from ZaidGPT!")
                break
            if user_input.lower() in ["clear", "reset"]:
                messages = [messages[0]]
                print("[*] Conversation memory cleared.")
                continue

            # Append user message
            messages.append({"role": "user", "content": user_input})

            # Format with modern chat template
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            print("\nZaidGPT: ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

            # Extract assistant response and append to history for multi-turn chat
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": response_text.strip()})

        except (KeyboardInterrupt, EOFError):
            print("\nExiting ZaidGPT chat. Have a great day!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZaidGPT Instruct Chat")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Foundation Instruct model (e.g., Qwen/Qwen2.5-0.5B-Instruct or HuggingFaceTB/SmolLM2-360M-Instruct)",
    )
    args = parser.parse_args()
    chat(model_id=args.model)
