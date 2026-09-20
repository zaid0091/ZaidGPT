# 🧠 ZaidGPT: Custom Transformer LLM from Scratch

**ZaidGPT** is a complete, standalone Generative Pre-trained Transformer (GPT) language model built from first principles using PyTorch.

Engineered specifically for balanced performance on systems with **8GB RAM** and **Intel UHD Graphics / Multi-Core CPUs**.

---

## 🏛️ Architecture Overview

ZaidGPT implements a modern Decoder-Only Transformer architecture:
- **Token & Positional Embeddings**: Vector representations with learnable position encoding.
- **Pre-LayerNorm Transformer Blocks**: Ensures stable gradient flow throughout deep layers.
- **Multi-Head Causal Self-Attention**: Batched Query, Key, and Value projections with lower-triangular causal masking.
- **Feed-Forward MLP**: $4\times$ expansion with GELU non-linear activations.
- **Weight Tying**: Shares weights between token embeddings and the output LM head.
- **Sampling Engine**: Supports Temperature scaling, Top-$k$ filtering, and Nucleus (Top-$p$) sampling with real-time token streaming.

---

## ⚡ Quick Start

### 1. Run Automated Unit Tests
To verify all neural network blocks, attention mechanisms, and tokenizer pipelines:
```bash
python -m model.test_model
```

### 2. Train ZaidGPT
Train the model on your training dataset with the CPU-optimized preset:
```bash
# Fast training (~800K parameters, 1000 steps)
python train.py --preset tiny --iters 1000

# Base training (~3.5M parameters, higher capacity)
python train.py --preset base --iters 2500
```

### 3. Interactive Chat with ZaidGPT
Launch the interactive terminal chat interface to converse with your model in real time:
```bash
python generate.py --interactive
```

### 4. Single-Prompt Generation
```bash
python generate.py --prompt "User: What is artificial intelligence?\nAssistant:" --temperature 0.7
```

---

## ⚙️ Model Presets

| Preset | Parameters | Layers | Heads | Embedding Dim | Context Size | Ideal For |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tiny`** | ~800K | 4 | 4 | 128 | 64 | Lightning-fast CPU training & testing |
| **`base`** | ~3.5M | 6 | 6 | 192 | 128 | High-capacity reasoning within 8GB RAM |

---

## 📂 Project Structure

```
ZaidGPT/
├── config.py              # Hyperparameters, presets, and CPU thread tuning
├── train.py               # Training loop with AdamW, cosine decay, and checkpointing
├── generate.py            # Streaming text generation & interactive terminal chat
├── model/
│   ├── transformer.py     # CausalSelfAttention, FeedForward, TransformerBlock, ZaidGPT
│   ├── language_model.py  # Model exports
│   └── test_model.py      # Architecture unit tests
├── tokenizer/
│   └── tokenizer.py       # CharacterTokenizer with JSON vocabulary persistence
├── data/
│   ├── dataset.py         # Text loading, train/val split, and batch sampling
│   ├── train.txt          # Multi-turn conversational & technical corpus
│   └── vocab.json         # Serialized vocabulary mappings
└── checkpoints/
    └── zaidgpt_best.pt    # Best model weights & training state
```
