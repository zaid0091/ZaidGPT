import torch
from config import GPTConfig
from model.transformer import ZaidGPT, CausalSelfAttention, FeedForward, TransformerBlock
from tokenizer.tokenizer import CharacterTokenizer


def test_tokenizer():
    print("Testing Tokenizer...")
    text = "Hello ZaidGPT! 123"
    tok = CharacterTokenizer(text)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text, f"Decoded text '{decoded}' does not match original '{text}'"
    print("[PASS] Tokenizer test passed.")


def test_transformer_block():
    print("Testing Transformer Block...")
    config = GPTConfig(block_size=32, n_embd=64, n_head=4, n_layer=2)
    block = TransformerBlock(config)
    x = torch.randn(2, 16, config.n_embd)
    out = block(x)
    assert out.shape == x.shape, f"Block shape mismatch: {out.shape} vs {x.shape}"
    print("[PASS] Transformer Block test passed.")


def test_model_forward():
    print("Testing ZaidGPT Forward & Backward Pass...")
    config = GPTConfig(vocab_size=60, block_size=32, n_embd=64, n_head=4, n_layer=2)
    model = ZaidGPT(config)

    x = torch.randint(0, 60, (2, 16))
    y = torch.randint(0, 60, (2, 16))

    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 60), f"Logits shape mismatch: {logits.shape}"
    assert loss is not None and loss.item() > 0, "Loss computation failed"

    loss.backward()
    print("[PASS] Model forward & backward test passed.")


def test_model_generation():
    print("Testing Generation...")
    config = GPTConfig(vocab_size=60, block_size=32, n_embd=64, n_head=4, n_layer=2)
    model = ZaidGPT(config)
    prompt = torch.randint(0, 60, (1, 4))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=10)
    assert generated.shape == (1, 14), f"Generation shape mismatch: {generated.shape}"
    print("[PASS] Generation test passed.")


if __name__ == "__main__":
    test_tokenizer()
    test_transformer_block()
    test_model_forward()
    test_model_generation()
    print("\n[SUCCESS] ALL TESTS PASSED!")