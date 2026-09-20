import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict
import torch

from config import GPTConfig, setup_system
from model.transformer import ZaidGPT
from tokenizer.tokenizer import CharacterTokenizer


def load_model_and_tokenizer(checkpoint_path: str = None, vocab_path: str = "data/vocab.json", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        if Path("checkpoints/zaidgpt_best.pt").exists():
            checkpoint_path = "checkpoints/zaidgpt_best.pt"
        elif Path("checkpoints/zaidgpt_latest.pt").exists():
            checkpoint_path = "checkpoints/zaidgpt_latest.pt"
        else:
            checkpoint_path = "checkpoints/zaidgpt_colab_best.pt"

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}.")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

    # 1. Load Tokenizer: Prefer vocab embedded directly inside checkpoint
    if "vocab" in checkpoint:
        tokenizer = CharacterTokenizer()
        tokenizer.vocab = checkpoint["vocab"]
        tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.vocab)}
        tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.vocab)}
    elif isinstance(checkpoint.get("config"), dict) and checkpoint["config"].get("vocab_size") == 65:
        # Exact 65-character Shakespeare vocabulary from Karpathy char-rnn
        shakespeare_vocab = list("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        tokenizer = CharacterTokenizer()
        tokenizer.vocab = shakespeare_vocab
        tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.vocab)}
        tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.vocab)}
    elif Path(vocab_path).exists():
        tokenizer = CharacterTokenizer(vocab_file=Path(vocab_path))
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}.")
    raw_config = checkpoint["config"]
    if isinstance(raw_config, dict):
        config = GPTConfig(**raw_config)
    else:
        config = raw_config

    config.device = device
    setup_system(config)

    model = ZaidGPT(config)
    
    state_dict = checkpoint["model_state_dict"]
    # Universal compatibility remap (net.0 -> c_fc, net.2 -> c_proj)
    remapped_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace(".mlp.net.0.", ".mlp.c_fc.").replace(".mlp.net.2.", ".mlp.c_proj.")
        remapped_state_dict[new_k] = v

    model.load_state_dict(remapped_state_dict)
    model.to(device)
    model.eval()

    print(f"[ZaidGPT] Loaded checkpoint from {checkpoint_path} (Trained for {checkpoint.get('iter_num', 'N/A')} steps)")
    if checkpoint.get('iter_num', 0) < 50:
        print("[!] Note: This checkpoint was saved at Step 1 (model is still training in the background). Please wait for training to finish for coherent responses!")
    return model, tokenizer, config


def generate_stream(
    model: ZaidGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.4,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    delay: float = 0.01,
):
    """
    Streams generated tokens with repetition penalty, temperature scaling, and top-k/top-p filtering.
    """
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    prompt_tokens_len = len(tokens)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            # Apply repetition penalty to prevent repeating character loops
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
        next_char = tokenizer.decode(idx_next[0])
        
        # Stream character to stdout safely
        try:
            sys.stdout.write(next_char)
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write("?")
            sys.stdout.flush()

        if delay > 0:
            time.sleep(delay)

        # Stop if user turnaround encountered in newly generated response
        new_text = tokenizer.decode(idx[0][prompt_tokens_len:])
        if "\nUser:" in new_text or "\n\nUser:" in new_text:
            break

    sys.stdout.write("\n")
    sys.stdout.flush()


def interactive_chat(model: ZaidGPT, tokenizer: CharacterTokenizer, temperature: float = 0.35, top_k: int = 40):
    """
    Runs an intelligent, context-aware interactive chat session with ZaidGPT.
    Uses RAG (Retrieval-Augmented Generation) to answer ANY question accurately.
    """
    from data.rag_engine import KnowledgeRAGEngine
    rag = KnowledgeRAGEngine()

    print("\n" + "=" * 60)
    print("🤖 ZaidGPT Supercharged AI Assistant (Ready to Answer Anything)")
    print("Type your message and press Enter. Type 'exit' or 'quit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Search relevant knowledge for the user query
            context = rag.search(user_input, top_k=1)
            
            if context and len(context) > 40:
                # Include relevant context to ground ZaidGPT's answer
                prompt = f"Knowledge: {context[:250]}\n\nUser: {user_input}\nAssistant: "
            else:
                prompt = f"User: {user_input}\nAssistant: "

            print("\nZaidGPT: ", end="", flush=True)
            generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=300,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=1.15,
                delay=0.01,
            )

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Have a great day!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZaidGPT Text Generation and Interactive Chat")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--vocab", type=str, default="data/vocab.json", help="Path to vocabulary file")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt to complete")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K sampling limit")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling threshold")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat session")
    args = parser.parse_args()

    try:
        model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.vocab)
        
        if args.interactive or args.prompt is None:
            interactive_chat(model, tokenizer, temperature=args.temperature, top_k=args.top_k)
        else:
            print(f"\n--- Prompt: {args.prompt} ---")
            print("--- Generated Response ---")
            sys.stdout.write(args.prompt)
            generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
    except Exception as e:
        print(f"\n[Error] {e}")
        print("Please train the model first by running: python train.py --preset tiny")
