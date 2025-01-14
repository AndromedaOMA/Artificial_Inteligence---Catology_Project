import torch
from .patch_embed import PatchEmbed
from .block import Block


class ViT(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 input_channels=3,
                 no_classes=8,
                 embed_dimension=6,
                 depth=6,
                 no_attn_heads=6,
                 qkv_bias=True,
                 proj_p=0.,
                 attn_p=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            embed_dimension=embed_dimension
        )
        self.class_tokens = torch.nn.Parameter(torch.zeros(
            1, 1, embed_dimension
        ))
        self.position_embed = torch.nn.Parameter(torch.zeros(
            1, 1 + self.patch_embed.no_of_patches, embed_dimension
        ))
        self.position_drop = torch.nn.Dropout(p=proj_p)
        self.blocks = torch.nn.ModuleList([
            Block(
                dimension=embed_dimension,
                no_attn_heads=no_attn_heads,
                qkv_bias=qkv_bias,
                proj_p=proj_p,
                attn_p=attn_p
            ) for _ in range(depth)
        ])
        self.norm = torch.nn.LayerNorm(embed_dimension, eps=1e-6)
        self.head = torch.nn.Linear(embed_dimension, no_classes)

    def forward(self, x):
        no_samples = x.shape[0]
        x = self.patch_embed(x)

        class_token = self.class_tokens.expand(
            no_samples, -1, -1
        )
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embed

        x = self.position_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        class_token_final = x[:, 0]
        x = self.head(class_token_final)

        return x
