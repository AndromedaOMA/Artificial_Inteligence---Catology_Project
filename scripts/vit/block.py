import torch
from .attention import Attention
from .mlp import MLP


class Block(torch.nn.Module):
    def __init__(self, dimension, no_attn_heads, qkv_bias=True, proj_p=0., attn_p=0.):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dimension, eps=1e-6)
        self.attn = Attention(
            dimension,
            no_attn_heads=no_attn_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        self.norm2 = torch.nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = MLP(
            input_size=dimension,
            output_size=dimension
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
