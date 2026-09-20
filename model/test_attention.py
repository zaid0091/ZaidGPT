import torch

from self_attention import SelfAttention


embedding_dim = 16
head_size = 8


attention = SelfAttention(
    embedding_dim,
    head_size
)


x = torch.randn(
    2,
    4,
    embedding_dim
)


output = attention(x)


print("Input:")
print(x.shape)

print("\nOutput:")
print(output.shape)