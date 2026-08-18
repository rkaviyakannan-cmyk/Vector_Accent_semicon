"""
src/training/loss.py

Combined Loss Function for FinFET Siamese Network:
- Similarity Loss: BCE Loss on predicted similarity score vs binary ground-truth label
- Coordinate Loss: Smooth L1 Loss on dx and dy offset predictions (applied on positive samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FinFETCombinedLoss(nn.Module):
    def __init__(self, lambda_similarity=1.0, lambda_coordinate=1.0, eps=1e-7):
        super().__init__()
        self.lambda_similarity = lambda_similarity
        self.lambda_coordinate = lambda_coordinate
        self.eps = eps
        self.bce_loss = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_similarity, pred_dx, pred_dy, label, gt_dx, gt_dy):
        """
        Parameters:
            pred_similarity (Tensor): (B, 1) in range [0, 1]
            pred_dx (Tensor): (B, 1) predicted dx offset
            pred_dy (Tensor): (B, 1) predicted dy offset
            label (Tensor): (B, 1) 1.0 for positive pair, 0.0 for negative
            gt_dx (Tensor): (B, 1) ground-truth dx offset
            gt_dy (Tensor): (B, 1) ground-truth dy offset
        """
        # Clamp similarity to prevent log(0)
        sim_clamped = torch.clamp(pred_similarity, self.eps, 1.0 - self.eps)
        loss_sim = self.bce_loss(sim_clamped, label)

        # Coordinate loss applied only on positive samples (label == 1)
        pos_mask = (label == 1.0).float()
        
        loss_dx_raw = self.smooth_l1(pred_dx, gt_dx)
        loss_dy_raw = self.smooth_l1(pred_dy, gt_dy)
        
        loss_coord_raw = loss_dx_raw + loss_dy_raw
        
        num_pos = pos_mask.sum()
        if num_pos > 0:
            loss_coord = (loss_coord_raw * pos_mask).sum() / (num_pos + self.eps)
        else:
            loss_coord = torch.tensor(0.0, device=pred_similarity.device)

        total_loss = self.lambda_similarity * loss_sim + self.lambda_coordinate * loss_coord
        
        return total_loss, loss_sim, loss_coord
