"""
src/siamese/cir.py

Lightweight CIR (Context-aware Intensity/Representation) module.
Enhances multi-scale receptive fields and structural channel correlations for FinFET SEM features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CIRModule(nn.Module):
    """
    Context-aware Intensity/Representation (CIR) module.
    Applies multi-branch receptive field projections (1x1, 3x3 depthwise, 5x5 depthwise)
    followed by channel excitation and feature aggregation.
    """
    def __init__(self, channels=96, reduction=4):
        super().__init__()
        branch_channels = channels // 3

        # Branch 1: 1x1 conv for local intensity
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 3x3 depthwise + 1x1 for fine scale structures
        self.b2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 5x5 depthwise + 1x1 for broader context
        self.b3 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Aggregation 1x1 conv
        self.agg = nn.Sequential(
            nn.Conv2d(branch_channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Channel Attention Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)

        cat = torch.cat([out1, out2, out3], dim=1)
        agg = self.agg(cat)

        att = self.se(agg)
        out = agg * att

        return self.act(x + out)
