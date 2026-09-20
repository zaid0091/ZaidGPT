# ==============================================================================
# 🚀 ZaidGPT: Google Colab Free GPU Supercharged Training Script
# ==============================================================================
# Run this in a free Google Colab GPU instance (Runtime -> Change runtime type -> T4 GPU).
# This trains a multi-million parameter ZaidGPT on massive text datasets in minutes!
# ==============================================================================

import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.nn import functional as F
import urllib.request

# -----------------------------------------------------------------------------
# 1. Hardware & System Setup
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Running on device: {device.upper()}")
if device == "cuda":
    print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[!] Warning: GPU not detected. Go to Runtime -> Change runtime type -> Select T4 GPU.")

# -----------------------------------------------------------------------------
# 2. Compile & Load Master Tech Stack, Coding & Official Documentation Corpus
# -----------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
data_file = "data/train.txt"

# If train.txt is small or missing, auto-scrape all real official docs & algorithms
if not os.path.exists(data_file) or os.path.getsize(data_file) < 150000:
    print("[*] Compiling Master Knowledge, Coding & Architecture Dataset...")
    try:
        from data.download_knowledge import generate_immense_knowledge_corpus
        corpus_text = generate_immense_knowledge_corpus()
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(corpus_text)
    except Exception as e:
        print(f"[!] Note: {e}")
    try:
        from data.web_docs_scraper import RealWebDocsScraper
        scraper = RealWebDocsScraper()
        scraper.run()
    except Exception as e:
        print(f"[!] Scraper note: {e}")

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()

print(f"[*] Master Dataset Loaded! Total Knowledge Characters: {len(text):,}")

# Build Character Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Save Vocab
import json
with open("data/vocab.json", "w", encoding="utf-8") as f:
    json.dump({"vocab": chars}, f)

data_tensor = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]
print(f"[*] Total Dataset Tokens: {len(data_tensor):,} | Vocab Size: {vocab_size}")

# -----------------------------------------------------------------------------
# 3. Model Architecture (Scaled to ~12M Parameters: 8 Layers, 8 Heads, 256 Embd, 256 Block Size)
# -----------------------------------------------------------------------------
@dataclass
class ColabGPTConfig:
    vocab_size: int = vocab_size
    block_size: int = 256        # Long 256-token context window
    n_embd: int = 256            # 256 embedding dimension
    n_head: int = 8              # 8 attention heads
    n_layer: int = 8             # 8 Transformer blocks
    dropout: float = 0.1
    bias: bool = False
    batch_size: int = 64         # Batched for GPU Tensor Cores
    learning_rate: float = 8e-4
    min_lr: float = 8e-5
    warmup_iters: int = 200
    max_iters: int = 4000        # 4,000 deep GPU steps
    weight_decay: float = 1e-2
    eval_interval: int = 250
    eval_iters: int = 40

config = ColabGPTConfig()

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ZaidGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=30):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# 4. Training Engine with Mixed Precision (AMP)
# -----------------------------------------------------------------------------
model = ZaidGPT(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"[*] ZaidGPT Model Initialized! Total Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - config.block_size - 1, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(split)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float("inf")
t0 = time.time()

print("\n" + "=" * 60)
print(f"🚀 Training ZaidGPT on {device.upper()} (3,000 Iterations)...")
print("=" * 60)

for iter_num in range(1, config.max_iters + 1):
    # Cosine learning rate decay
    if iter_num < config.warmup_iters:
        lr = config.learning_rate * iter_num / config.warmup_iters
    else:
        decay_ratio = (iter_num - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = config.min_lr + coeff * (config.learning_rate - config.min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % config.eval_interval == 0 or iter_num == 1:
        losses = estimate_loss()
        dt = time.time() - t0
        t0 = time.time()
        print(f"Step {iter_num:4d}/{config.max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e} | Time: {dt:.2f}s", flush=True)

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "vocab": chars,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }, "checkpoints/zaidgpt_colab_best.pt")

    x, y = get_batch("train")
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        logits, loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

print("\n" + "=" * 60)
print(f"🎉 Training Complete! Best Val Loss: {best_val_loss:.4f}")
print("Saved checkpoint to: checkpoints/zaidgpt_colab_best.pt")
print("=" * 60)

# Sample generation
print("\n[Generated Sample from ZaidGPT]:")
start_context = torch.tensor(encode("To be, or not to be"), dtype=torch.long, device=device).unsqueeze(0)
output_tokens = model.generate(start_context, max_new_tokens=200, temperature=0.6, top_k=20)
print(decode(output_tokens[0].tolist()))
