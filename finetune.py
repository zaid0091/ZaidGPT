"""
Fine-Tuning Engine for ZaidGPT.
Takes a pre-trained foundation model (e.g., GPT-2, SmolLM, or Qwen) and fine-tunes it
on our custom ZaidGPT conversational, coding, and tech documentation dataset.
"""

import os
import sys
from pathlib import Path

# Redirect HuggingFace cache to D: drive to prevent C: drive full errors
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "huggingface"))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


def load_custom_dataset(data_path: str = "data/train.txt") -> Dataset:
    """Loads and formats the custom knowledge dataset into training samples."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split into individual dialogue / knowledge chunks
    samples = [s.strip() for s in raw_text.split("\n\n") if len(s.strip()) > 30]
    print(f"[*] Loaded {len(samples):,} fine-tuning training examples from {data_path}")
    return Dataset.from_dict({"text": samples})


def run_finetuning(
    base_model_name: str = "gpt2",
    output_dir: str = "checkpoints/zaidgpt_finetuned",
    epochs: int = 3,
    batch_size: int = 4,
):
    print("\n" + "=" * 60)
    print(f"[*] Fine-Tuning '{base_model_name}' into ZaidGPT")
    print("=" * 60 + "\n")

    # 1. Load Pre-Trained Tokenizer & Model (Already speaks fluent English & code)
    print(f"[*] Downloading base foundation model weights: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    print(f"[*] Pre-trained model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # 2. Prepare Dataset
    dataset = load_custom_dataset("data/train.txt")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    # Freeze lower layers for ultra-low memory CPU fine-tuning (only train top 2 transformer layers)
    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False
    for block in model.transformer.h[:-2]:
        for param in block.parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Memory Optimization: Training {trainable_params:,} of {total_params:,} parameters (Top Layers)")

    # 3. Training Arguments (Optimized for 8GB RAM CPU/Laptop)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    # 4. Start Fine-Tuning
    print("\n[*] Starting Fine-Tuning...")
    trainer.train()

    # 5. Save Final Model & Tokenizer
    print(f"\n[*] Saving fine-tuned ZaidGPT to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Fine-Tuning Complete! Model saved to: {output_dir}")
    print("Chat with your model using: python chat_finetuned.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained model into ZaidGPT")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model (e.g. gpt2, Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    args = parser.parse_args()

    run_finetuning(base_model_name=args.model, epochs=args.epochs)
