import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        head_size,
        context_length
    ):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            )
        )


    def forward(self, x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)


        # Attention scores

        scores = Q @ K.transpose(
            -2,
            -1
        )


        # Scale

        scores = scores / (
            K.shape[-1] ** 0.5
        )


        # Causal mask

        T = x.shape[1]

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )


        # Softmax

        weights = torch.softmax(
            scores,
            dim=-1
        )


        # Weighted values

        output = weights @ V


        return output