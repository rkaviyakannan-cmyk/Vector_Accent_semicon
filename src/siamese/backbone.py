"""
src/siamese/backbone.py

MobileNetV3-Small based shared backbone adapter with 1-channel grayscale input support,
Inverted Bottleneck blocks, Squeeze-and-Excitation (SE/ECAM) channel attention, and CBAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation (SE/ECAM) channel attention module."""
    def __init__(self, channels, squeeze_factor=4):
        super().__init__()
        squeezed_channels = max(8, channels // squeeze_factor)
        self.fc1 = nn.Conv2d(channels, squeezed_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezed_channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM) combining Channel & Spatial Attention."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        red_channels = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, red_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_channels, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.channel_sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_s, max_s], dim=1)
        spatial_att = self.spatial_sigmoid(self.spatial_conv(spatial_in))
        x = x * spatial_att

        return x


class InvertedResidualBlock(nn.Module):
    """Inverted bottleneck block with depthwise separable convolutions and SE attention."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio=3, use_se=True):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True)
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True)
        ])

        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))

        # Pointwise linear projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3SmallBackbone(nn.Module):
    """
    Lightweight MobileNetV3-Small inspired backbone for 128x128 grayscale patch feature extraction.
    Outputs feature maps of size (B, 96, 16, 16).
    """
    def __init__(self, in_channels=1, use_cbam=True):
        super().__init__()
        self.use_cbam = use_cbam

        # 1-channel Grayscale Stem Adapter
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False), # -> (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )

        # Inverted Bottleneck Sequence
        self.b1 = InvertedResidualBlock(16, 24, stride=2, expand_ratio=3, use_se=True) # -> (24, 32, 32)
        self.b2 = InvertedResidualBlock(24, 40, stride=2, expand_ratio=3, use_se=True) # -> (40, 16, 16)
        self.b3 = InvertedResidualBlock(40, 96, stride=1, expand_ratio=4, use_se=True) # -> (96, 16, 16)

        if use_cbam:
            self.cbam = CBAM(96)
        else:
            self.cbam = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.cbam(x)
        return x
