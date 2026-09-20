import torch
import torch.nn as nn


# ==========================================
# 1. Configuration
# ==========================================

vocab_size = 50
embedding_size = 16


# ==========================================
# 2. Create embedding layer
# ==========================================

embedding = nn.Embedding(
    vocab_size,
    embedding_size
)


# ==========================================
# 3. Example token IDs
# ==========================================

tokens = torch.tensor([
    5,
    12,
    25,
    40
])


# ==========================================
# 4. Convert token IDs to vectors
# ==========================================

vectors = embedding(tokens)


print("Token IDs:")
print(tokens)

print("\nEmbedding vectors:")
print(vectors)

print("\nShape:")
print(vectors.shape)