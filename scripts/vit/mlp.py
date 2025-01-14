import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=125),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=125, out_features=output_size),
            torch.nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
