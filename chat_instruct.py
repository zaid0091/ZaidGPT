import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from llama_cpp import Llama

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "models"))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "qwen2.5-coder-7b-instruct-q4_k_m.gguf")

num_threads = min(os.cpu_count() or 4, 8)


def chat(model_path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"[!] Error: Model file not found at {model_path}")
        print("[*] Please wait for the GGUF model download to complete.")
        return

    print("\n" + "=" * 65)
    print("ChatGPT Local Terminal Assistant (GGUF Engine)")
    print(f"[*] Model: {os.path.basename(model_path)} | CPU Threads: {num_threads} | AVX2: ON")
    print("Type your message and press Enter. Type 'exit' or 'quit' to quit.")
    print("=" * 65 + "\n")

    print(f"[*] Loading model...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=num_threads,
            n_batch=512,
            verbose=False,
        )
        print("[OK] Assistant Engine is online and ready!\n")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI software engineer and technical assistant.\n"
                "Follow these rules for every response:\n"
                "1. Think step-by-step before answering complex questions.\n"
                "2. Provide clean, robust, and production-ready code with complete implementations.\n"
                "3. Structure your answers with clear headings, bullet points, and clean markdown code blocks.\n"
                "4. Be direct, factual, and concise."
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

            print("\nChatGPT: ", end="", flush=True)
            full_response = ""

            response_stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=768,
                temperature=0.6,
                top_p=0.9,
                repeat_penalty=1.15,
                stream=True,
            )

            for chunk in response_stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_response += content

            print()
            messages.append({"role": "assistant", "content": full_response.strip()})

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error during generation: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ChatGPT Local Assistant CLI (GGUF)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to GGUF model file")
    args = parser.parse_args()
    chat(args.model)
