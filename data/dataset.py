import os
from pathlib import Path
from typing import Tuple
import torch

from tokenizer.tokenizer import CharacterTokenizer


class TextDataset:
    """
    Manages loading, tokenizing, splitting, and batch extraction for training ZaidGPT.
    """

    def __init__(self, data_path: str = "data/train.txt", vocab_path: str = "data/vocab.json", split_ratio: float = 0.9):
        self.data_path = Path(data_path)
        self.vocab_path = Path(vocab_path)
        self.split_ratio = split_ratio

        # 1. Load raw text
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()

        # 2. Build or load tokenizer
        self.tokenizer = CharacterTokenizer(self.raw_text)
        self.tokenizer.save(self.vocab_path)
        print(f"[Dataset] Vocabulary size: {self.tokenizer.vocab_size} unique characters.")

        # 3. Encode entire dataset into tensor
        tokens = self.tokenizer.encode(self.raw_text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        print(f"[Dataset] Total tokens loaded: {len(self.data)}")

        # 4. Train/Val split
        n = int(self.split_ratio * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:] if n < len(self.data) else self.data
        print(f"[Dataset] Train tokens: {len(self.train_data)} | Val tokens: {len(self.val_data)}")

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a random mini-batch of inputs (x) and autoregressive next-token targets (y).
        """
        data = self.train_data if split == "train" else self.val_data
        
        # Guard against short text lengths
        max_start = len(data) - block_size - 1
        if max_start <= 0:
            # Replicate/pad if dataset is very small relative to block size
            pad_needed = block_size + 2 - len(data)
            data = torch.cat([data, data[:pad_needed]])
            max_start = len(data) - block_size - 1

        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        if device != "cpu":
            # Pin memory / move to target device asynchronously if on CUDA
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y