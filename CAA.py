import torch
import torch.nn as nn

class CAA(nn.Module):
    """Context Anchor Attention 模块 – 即插即用版"""
    def __init__(self, channels=256, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size),
                                padding=(0, h_kernel_size//2), groups=channels)
        self.h_bn = nn.BatchNorm2d(channels)
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1),
                                padding=(v_kernel_size//2, 0), groups=channels)
        self.v_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.bn1(attn)
        attn = self.relu(attn)
        attn = self.h_conv(attn)
        attn = self.h_bn(attn)
        attn = self.relu(attn)
        attn = self.v_conv(attn)
        attn = self.v_bn(attn)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.bn2(attn)
        attn = self.act(attn)
        out = x * attn + identity
        return out