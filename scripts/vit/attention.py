import torch


class Attention(torch.nn.Module):
    def __init__(self, embed_dimension, no_attn_heads=6, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.no_attn_heads = no_attn_heads
        self.embed_dimension = embed_dimension

        if embed_dimension % no_attn_heads != 0:
            raise ValueError("Embedding embed_dimension must be divisible by the number of attention heads.")

        self.head_embed_dimension = embed_dimension // no_attn_heads
        self.scale = self.head_embed_dimension ** -0.5

        self.qkv = torch.nn.Linear(embed_dimension, embed_dimension * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_p)
        self.projection = torch.nn.Linear(embed_dimension, embed_dimension)
        self.projection_drop = torch.nn.Dropout(proj_p)

    def forward(self, x):  # n_samples, n_patches, embed_dimension
        n_samples, n_tokens, embed_dimension = x.shape

        if embed_dimension != self.embed_dimension:
            raise ValueError("Input embed_dimension does not match model embed_dimension")

        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.no_attn_heads, self.head_embed_dimension)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2)
        weighted_avg = weighted_avg.flatten(2)

        x = self.projection(weighted_avg)
        x = self.projection_drop(x)

        return x
