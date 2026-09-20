import os
import torch
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 256        # Set dynamically by tokenizer
    block_size: int = 128        # Context window length (tokens)
    n_embd: int = 192            # Embedding dimension
    n_head: int = 6              # Number of attention heads (n_embd % n_head == 0)
    n_layer: int = 6             # Number of Transformer blocks
    dropout: float = 0.1         # Regularization dropout rate
    bias: bool = False           # True: bias in Linears and LayerNorms, like GPT-2. False: like LLaMA/modern GPTs

    # Training settings
    batch_size: int = 16         # Number of sequences per batch
    learning_rate: float = 6e-4  # Max learning rate
    min_lr: float = 6e-5         # Min learning rate for cosine decay
    warmup_iters: int = 100      # Warmup iterations
    max_iters: int = 2000        # Maximum training iterations
    weight_decay: float = 1e-2   # AdamW weight decay
    grad_clip: float = 1.0       # Gradient norm clipping
    eval_interval: int = 200     # Evaluation step interval
    eval_iters: int = 10         # Number of batches to average for eval loss
    
    # Paths & System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"


# Preset for fast iterations & lower resource usage on 8GB RAM / CPU
def get_preset(name: str = "base") -> GPTConfig:
    """
    Returns a configured GPTConfig preset:
    - 'tiny': ~1.8M parameters, fast CPU training with 128 context window.
    - 'base': ~3.5M parameters, high capacity for 8GB RAM.
    """
    if name.lower() == "tiny":
        return GPTConfig(
            block_size=128,
            n_embd=192,
            n_head=6,
            n_layer=4,
            batch_size=16,
            learning_rate=6e-4,
            max_iters=2000,
        )
    elif name.lower() == "base":
        return GPTConfig(
            block_size=128,
            n_embd=192,
            n_head=6,
            n_layer=6,
            batch_size=16,
            learning_rate=6e-4,
            max_iters=2500,
        )
    else:
        return GPTConfig()


def setup_system(config: GPTConfig):
    """Configures PyTorch CPU thread count and device settings for optimal Intel CPU throughput."""
    if config.device == "cpu":
        # Utilize available CPU cores effectively without thrashing
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(max(1, min(num_cores, 8)))
        print(f"[System] Running on CPU with {torch.get_num_threads()} worker threads.")
    else:
        print(f"[System] CUDA Device Detected: {torch.cuda.get_device_name(0)}")
