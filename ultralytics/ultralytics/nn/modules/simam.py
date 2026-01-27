import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x.var(dim=[2, 3], keepdim=True, unbiased=False) + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)