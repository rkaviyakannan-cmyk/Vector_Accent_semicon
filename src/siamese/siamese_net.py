"""
src/siamese/siamese_net.py

Complete FinFET Drift-Sense Deep Siamese Network architecture.
Integrates:
- Shared MobileNetV3-Small backbone (1-channel adapter)
- SE/ECAM & CBAM Attention
- CIR (Context-aware Intensity/Representation) Module
- Dual Correlation Module (Channel-wise & Pixel-wise)
- Similarity Score Head
- X/Y Feedback Regressor Heads (dx, dy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV3SmallBackbone
from .cir import CIRModule
from .dual_correlation import DualCorrelationModule
from .xy_feedback import XYFeedbackHead

class FinFETSiameseNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 use_cbam=True,
                 use_cir=True,
                 use_dual_correlation=True,
                 use_xy_feedback=True):
        super().__init__()
        
        self.use_cbam = use_cbam
        self.use_cir = use_cir
        self.use_dual_correlation = use_dual_correlation
        self.use_xy_feedback = use_xy_feedback
        
        # Shared Backbone (MobileNetV3-Small)
        self.backbone = MobileNetV3SmallBackbone(in_channels=in_channels, use_cbam=use_cbam)
        
        # CIR Module
        if use_cir:
            self.cir = CIRModule(channels=96)
        else:
            self.cir = nn.Identity()
            
        # Dual Correlation Module
        if use_dual_correlation:
            self.dual_corr = DualCorrelationModule(channels=96, out_channels=128)
            corr_dim = 128
        else:
            self.dual_corr = None
            corr_dim = 96
            
        # Similarity Head
        self.similarity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(corr_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # X/Y Feedback Offset Regressor Head
        if use_xy_feedback:
            self.xy_head = XYFeedbackHead(in_channels=corr_dim)
        else:
            self.xy_head = None

    def forward_one(self, x):
        """Passes a single patch through the shared backbone + CIR."""
        feat = self.backbone(x)
        feat = self.cir(feat)
        return feat

    def forward(self, ref_patch, cand_patch):
        """
        Forward pass for a reference patch and candidate patch pair.
        
        Parameters:
            ref_patch (Tensor): (B, 1, 128, 128)
            cand_patch (Tensor): (B, 1, 128, 128)
            
        Returns:
            similarity (Tensor): (B, 1) score in range [0, 1]
            dx (Tensor): (B, 1) predicted X offset in patch pixels
            dy (Tensor): (B, 1) predicted Y offset in patch pixels
        """
        # NaN / Inf assertions to protect training
        assert torch.isfinite(ref_patch).all(), "NaN/Inf detected in ref_patch input!"
        assert torch.isfinite(cand_patch).all(), "NaN/Inf detected in cand_patch input!"

        # Extract features using shared weights
        F_r = self.forward_one(ref_patch)
        F_c = self.forward_one(cand_patch)

        # Correlation computation
        if self.use_dual_correlation and self.dual_corr is not None:
            F_fused, _, _ = self.dual_corr(F_r, F_c)
        else:
            # Fallback simple absolute difference
            F_fused = torch.abs(F_r - F_c)

        # Similarity score
        similarity = self.similarity_head(F_fused)

        # X/Y Feedback predictions
        if self.use_xy_feedback and self.xy_head is not None:
            dx, dy = self.xy_head(F_fused)
        else:
            batch_size = ref_patch.size(0)
            dx = torch.zeros(batch_size, 1, device=ref_patch.device)
            dy = torch.zeros(batch_size, 1, device=ref_patch.device)

        return similarity, dx, dy
