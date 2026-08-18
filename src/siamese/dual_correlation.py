"""
src/siamese/dual_correlation.py

Dual Correlation Module:
Computes both Channel-wise Correlation (channel-level feature similarity)
and Pixel-wise Correlation (spatial cross-correlation) followed by learnable correlation fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualCorrelationModule(nn.Module):
    def __init__(self, channels=96, out_channels=128):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        # Learnable Fusion Network
        # Input channel dim = channels (channel corr expanded) + channels (pixel corr pooled) = 2 * channels
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_r, F_c):
        """
        F_r: Reference feature map (B, C, H, W)
        F_c: Candidate feature map (B, C, H, W)
        Returns:
            F_fused: Fused correlation features (B, out_channels, H, W)
            C_channel: Channel correlation vector (B, C)
            C_pixel: Pixel correlation map (B, C, H, W)
        """
        B, C, H, W = F_r.shape
        eps = 1e-7

        # 1. Channel-wise Correlation
        # Pool spatial dimensions to get channel descriptors (B, C, 1, 1)
        r_chan = F.adaptive_avg_pool2d(F_r, 1)
        c_chan = F.adaptive_avg_pool2d(F_c, 1)
        
        # Cosine similarity across channels
        r_chan_norm = F.normalize(r_chan, dim=1, eps=eps)
        c_chan_norm = F.normalize(c_chan, dim=1, eps=eps)
        
        # Elementwise channel similarity (B, C, 1, 1)
        C_channel_vec = r_chan_norm * c_chan_norm
        # Expand across spatial dimensions (B, C, H, W)
        C_channel_map = C_channel_vec.expand(-1, -1, H, W)

        # 2. Pixel-wise Correlation
        # Channel-wise normalized features
        r_pix_norm = F.normalize(F_r, dim=1, eps=eps)
        c_pix_norm = F.normalize(F_c, dim=1, eps=eps)
        
        # Local spatial cross-correlation (B, C, H, W)
        C_pixel_map = r_pix_norm * c_pix_norm

        # 3. Learnable Fusion
        concat_corr = torch.cat([C_channel_map, C_pixel_map], dim=1) # (B, 2*C, H, W)
        F_fused = self.fusion(concat_corr) # (B, out_channels, H, W)

        return F_fused, C_channel_vec.squeeze(-1).squeeze(-1), C_pixel_map
