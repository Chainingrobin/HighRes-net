"""Convolutional Block Attention Module (CBAM) building blocks."""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        hidden = max(1, channels // max(1, reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),  # bias=True here
        )
        self.gate = nn.Sigmoid()
        self._init_passthrough()

    def _init_passthrough(self):
        """Initialize with zero weights and positive bias so the gate starts open."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final_conv = self.mlp[2]
        nn.init.constant_(final_conv.bias, 1.5)

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.gate(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SpatialAttention kernel_size must be odd")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=True)  # bias=True here
        self.gate = nn.Sigmoid()
        self._init_passthrough()

    def _init_passthrough(self):
        """Initialize with zero weights and positive bias so the gate starts open."""
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 1.5)

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        attn = self.gate(self.conv(pooled))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(
            channels=channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x