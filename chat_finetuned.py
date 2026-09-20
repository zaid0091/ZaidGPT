"""
Interactive Streaming Chat for Fine-Tuned ZaidGPT.
Connects to the fine-tuned model and generates fluent, ChatGPT-level responses.
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


def chat():
    model_path = "checkpoints/zaidgpt_finetuned"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("ZaidGPT Fine-Tuned Assistant (Fluent English & Code)")
    print("Type your message and press Enter. Type 'exit' or 'quit' to quit.")
    print("=" * 60 + "\n")

    try:
        print(f"[*] Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()
        print("[*] Model ready!\n")
    except Exception as e:
        print(f"[!] Error loading fine-tuned model: {e}")
        print("Please run fine-tuning first: python finetune.py")
        return

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            prompt = f"User: {user_input}\nAssistant: "
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            print("\nZaidGPT: ", end="", flush=True)
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Have a great day!")
            break


if __name__ == "__main__":
    chat()
