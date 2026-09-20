import math
import os
import time
from pathlib import Path
import torch

from config import GPTConfig, get_preset, setup_system
from data.dataset import TextDataset
from model.transformer import ZaidGPT


def get_lr(it: int, config: GPTConfig) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.
    """
    # 1) Linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / max(1, config.warmup_iters)
    # 2) If it > max_iters, return min learning rate
    if it > config.max_iters:
        return config.min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model: ZaidGPT, dataset: TextDataset, config: GPTConfig):
    """
    Computes accurate train and validation loss averaged over multiple batches.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = dataset.get_batch(split, config.batch_size, config.block_size, config.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(preset: str = "tiny", max_iters: int = None):
    # 1. Configuration & System setup
    config = get_preset(preset)
    if max_iters is not None:
        config.max_iters = max_iters

    setup_system(config)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 2. Dataset & Vocabulary
    dataset = TextDataset(
        data_path=os.path.join(config.data_dir, "train.txt"),
        vocab_path=os.path.join(config.data_dir, "vocab.json"),
    )
    config.vocab_size = dataset.tokenizer.vocab_size

    # 3. Model Initialization
    print(f"\n[Model] Initializing ZaidGPT ({preset.upper()} Preset)...")
    model = ZaidGPT(config)
    model.to(config.device)
    num_params = model.get_num_params()
    print(f"[Model] Total parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # 4. Optimizer Setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    # 5. Training Loop
    print("\n" + "=" * 60)
    print(f">>> Starting ZaidGPT Training on {config.device.upper()}")
    print(f"Iterations: {config.max_iters} | Batch size: {config.batch_size} | Context: {config.block_size}")
    print("=" * 60 + "\n")

    best_val_loss = float("inf")
    t0 = time.time()

    for iter_num in range(1, config.max_iters + 1):
        # Update learning rate per schedule
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate train & val loss periodically
        if iter_num % config.eval_interval == 0 or iter_num == 1:
            losses = estimate_loss(model, dataset, config)
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"Step {iter_num:4d}/{config.max_iters} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {dt:.2f}s",
                flush=True
            )

            # Save best validation checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint_path = Path(config.checkpoint_dir) / "zaidgpt_best.pt"
                from dataclasses import asdict
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "vocab": dataset.tokenizer.vocab,
                    "config": asdict(config),
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"  [*] Saved best checkpoint: val_loss={best_val_loss:.4f} -> {checkpoint_path}", flush=True)

            # Save latest checkpoint periodically after meaningful learning
            if iter_num >= 100:
                latest_path = Path(config.checkpoint_dir) / "zaidgpt_latest.pt"
                from dataclasses import asdict
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "val_loss": losses["val"],
                    "train_loss": losses["train"],
                    "vocab": dataset.tokenizer.vocab,
                    "config": asdict(config),
                }, latest_path)

            # Quick sample generation preview during training
            if iter_num % (config.eval_interval * 2) == 0 or iter_num == config.max_iters:
                print("\n" + "-" * 35 + " [Live Generation Sample] " + "-" * 35)
                sample_prompt = "User: Who are you?\nAssistant:"
                prompt_tokens = dataset.tokenizer.encode(sample_prompt)
                x_prompt = torch.tensor(prompt_tokens, dtype=torch.long, device=config.device).unsqueeze(0)
                generated_tokens = model.generate(x_prompt, max_new_tokens=80, temperature=0.7, top_k=20)
                decoded_text = dataset.tokenizer.decode(generated_tokens[0])
                print(decoded_text.strip())
                print("-" * 95 + "\n")

        # Fetch training batch & compute gradients
        x, y = dataset.get_batch("train", config.batch_size, config.block_size, config.device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    print("\n" + "=" * 60)
    print(f"[DONE] Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: checkpoints/zaidgpt_best.pt")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ZaidGPT Language Model from Scratch")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "base"], help="Model preset (tiny or base)")
    parser.add_argument("--iters", type=int, default=None, help="Number of training iterations (overrides preset)")
    args = parser.parse_args()

    train(preset=args.preset, max_iters=args.iters)