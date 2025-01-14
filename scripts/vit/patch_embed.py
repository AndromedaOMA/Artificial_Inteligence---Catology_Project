import torch


class PatchEmbed(torch.nn.Module):
    def __init__(self, image_size, patch_size, input_channels=1, embed_dimension=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.no_of_patches = (image_size // patch_size) ** 2
        self.projection = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dimension,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):       #(batch_size, in_chanells, img_size, img_size)
        x = self.projection(x)  #(batch_size, embed_dim, n_patches, n_patches)
        x = x.flatten(2)        #(batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   #(batch_size, n_patches, embed_dim)
        return x
