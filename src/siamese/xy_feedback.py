"""
src/siamese/xy_feedback.py

X/Y Feedback Head:
Predicts sub-pixel coordinate offsets (dx, dy) from fused correlation features.
"""

import torch
import torch.nn as nn

class XYFeedbackHead(nn.Module):
    """
    Sub-pixel offset regression heads for delta_x and delta_y.
    Input: Fused correlation features (B, in_channels, H, W)
    Outputs: dx (B, 1), dy (B, 1) in candidate patch pixel units.
    """
    def __init__(self, in_channels=128):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared intermediate FC
        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True)
        )
        
        # X offset regressor head
        self.head_dx = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        # Y offset regressor head
        self.head_dy = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        feat = self.pool(x).view(x.size(0), -1)
        shared = self.shared_fc(feat)
        
        dx = self.head_dx(shared)
        dy = self.head_dy(shared)
        
        return dx, dy
