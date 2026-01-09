import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.e_lambda)
        return x * torch.sigmoid(y)