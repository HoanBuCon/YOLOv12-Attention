import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.spatial(torch.cat([avg, max], dim=1)))
        return x * sa


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)
        return x * torch.sigmoid((x - mean) / torch.sqrt(var + self.e_lambda))
