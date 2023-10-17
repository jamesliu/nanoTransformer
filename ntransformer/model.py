import torch
import torch.nn as nn

# Assuming multihead_attention_batched is defined somewhere within your attention layers script
from .attention_layers import multihead_attention_batched

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # Initialize the projection matrices for multi-head attention
        # Note: We're not using Linear layers here because 'multihead_attention_batched'
        # expects the actual matrices, not layers.
        self.P_q = nn.Parameter(torch.randn(heads, dim, dim // heads))  # [heads, dim, dim_head]
        self.P_k = nn.Parameter(torch.randn(heads, dim, dim // heads))
        self.P_v = nn.Parameter(torch.randn(heads, dim, dim // heads))
        self.P_o = nn.Parameter(torch.randn(heads, dim, dim // heads))

        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim),  # This could be expanded to a larger dimension if you prefer
            nn.ReLU(),  # Using ReLU for simplicity
            nn.Linear(dim, dim),
        )

        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

        self.attention = multihead_attention_batched

    def forward(self, x, mask):
        """
        print("X shape:", x.shape)
        print("Mask shape:", mask.shape)
        print("P_q shape:", self.P_q.shape)
        print("P_k shape:", self.P_k.shape)
        print("P_v shape:", self.P_v.shape)
        print("P_o shape:", self.P_o.shape)
        """
        # Applying the multihead attention with the projection matrices
        attn_output = multihead_attention_batched(x, x, mask, self.P_q, self.P_k, self.P_v, self.P_o)

        # Add & Norm
        x = self.layer_norm1(attn_output + x)

        # Feedforward network
        ff_output = self.feed_forward(x)

        # Add & Norm
        x = self.layer_norm2(ff_output + x)

        return x

# Make sure to initialize and use this TransformerBlock correctly in your Transformer model.

class Transformer(nn.Module):
    def __init__(self, num_layers, model_dim, heads):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, heads) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return x  # This is the output after the transformer blocks.

# Usage:
# model = Transformer(num_layers=3, model_dim=512, heads=8)
# output = model(input_tensor, mask)
