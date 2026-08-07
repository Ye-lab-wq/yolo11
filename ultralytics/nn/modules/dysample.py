import torch
import torch.nn as nn
import torch.nn.functional as F

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = in_channels

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)