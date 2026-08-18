"""
src/training/replay.py

Hard-Negative Error Replay Memory.
Stores false-positive candidates (high predicted similarity at incorrect locations)
and replays them during training to suppress repeating FinFET ambiguity.
"""

import numpy as np
import torch

class ReplayMemory:
    def __init__(self, capacity=2000, replay_ratio=0.25):
        self.capacity = capacity
        self.replay_ratio = replay_ratio
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def add(self, ref_patch, cand_patch, pred_sim, cand_x, cand_y, gt_x, gt_y):
        """
        Adds a sample to replay memory if it represents a hard-negative error
        (high similarity at spatially incorrect location).
        """
        error_x = float(gt_x - cand_x)
        error_y = float(gt_y - cand_y)
        total_error = float(np.hypot(error_x, error_y))

        # Hard negative condition: Spatial error > 30px AND predicted similarity > 0.4
        if total_error > 30.0 and float(pred_sim) > 0.4:
            item = {
                "ref_patch": ref_patch.detach().cpu().clone(),
                "cand_patch": cand_patch.detach().cpu().clone(),
                "pred_sim": float(pred_sim),
                "cand_x": float(cand_x),
                "cand_y": float(cand_y),
                "gt_x": float(gt_x),
                "gt_y": float(gt_y),
                "total_error": total_error,
                "label": torch.tensor([0.0], dtype=torch.float32),
                "dx": torch.tensor([0.0], dtype=torch.float32),
                "dy": torch.tensor([0.0], dtype=torch.float32)
            }
            
            if len(self.memory) >= self.capacity:
                # Evict item with lowest error/similarity score
                min_idx = min(range(len(self.memory)), key=lambda i: self.memory[i]["total_error"] * self.memory[i]["pred_sim"])
                self.memory[min_idx] = item
            else:
                self.memory.append(item)

    def sample(self, batch_size):
        """Samples a batch of hard negatives from replay memory."""
        if not self.memory:
            return None

        sample_size = min(batch_size, len(self.memory))
        indices = np.random.choice(len(self.memory), size=sample_size, replace=False)
        sampled_items = [self.memory[i] for i in indices]

        ref_batch = torch.stack([item["ref_patch"] for item in sampled_items])
        cand_batch = torch.stack([item["cand_patch"] for item in sampled_items])
        label_batch = torch.stack([item["label"] for item in sampled_items])
        dx_batch = torch.stack([item["dx"] for item in sampled_items])
        dy_batch = torch.stack([item["dy"] for item in sampled_items])

        return {
            "ref_patch": ref_batch,
            "cand_patch": cand_batch,
            "label": label_batch,
            "dx": dx_batch,
            "dy": dy_batch
        }
