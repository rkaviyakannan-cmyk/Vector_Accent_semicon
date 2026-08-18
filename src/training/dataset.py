"""
src/training/dataset.py

PyTorch Dataset for FinFET Siamese training and validation.
Supports online patch extraction, independent stochastic data augmentation,
and creation of positive, hard-negative, and easy-negative candidate patches.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class FinFETSiameseDataset(Dataset):
    def __init__(self,
                 manifest_df,
                 input_size=128,
                 is_training=True,
                 hard_neg_ratio=0.5,
                 aug_prob=0.5,
                 seed=42):
        
        self.df = manifest_df.reset_index(drop=True)
        self.input_size = input_size
        self.is_training = is_training
        self.hard_neg_ratio = hard_neg_ratio
        self.aug_prob = aug_prob
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.df)

    def _independent_augment(self, patch, rng):
        """Applies independent stochastic intensity/noise augmentation."""
        patch = patch.astype(np.float32)

        # 1. Random Brightness / Contrast adjustment
        if rng.rand() < self.aug_prob:
            alpha = rng.uniform(0.8, 1.2) # Contrast
            beta = rng.uniform(-0.1, 0.1)  # Brightness
            patch = np.clip(patch * alpha + beta, 0.0, 1.0)

        # 2. Random Gamma transformation
        if rng.rand() < self.aug_prob:
            gamma = rng.uniform(0.8, 1.2)
            patch = np.clip(np.power(patch + 1e-7, gamma), 0.0, 1.0)

        # 3. Random Gaussian / Poisson noise
        if rng.rand() < self.aug_prob:
            noise_sigma = rng.uniform(0.01, 0.03)
            noise = rng.normal(0, noise_sigma, patch.shape).astype(np.float32)
            patch = np.clip(patch + noise, 0.0, 1.0)

        return patch

    def _crop_and_resize(self, img, cx, cy, crop_w, crop_h):
        """Crops sub-window centered at (cx, cy) and resizes to self.input_size."""
        h, w = img.shape[:2]
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        x1 = int(round(cx - half_w))
        y1 = int(round(cy - half_h))
        x2 = int(round(cx + half_w))
        y2 = int(round(cy + half_h))

        # Pad if out of bounds
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                            cv2.BORDER_REFLECT)
            crop = img_padded[y1 + pad_top:y2 + pad_top, x1 + pad_left:x2 + pad_left]
        else:
            crop = img[y1:y2, x1:x2]

        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((crop_h, crop_w), dtype=img.dtype)

        resized = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        return resized

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            raise ValueError(f"Could not load image pair: {row['reference_path']} or {row['search_path']}")

        gt_x = float(row['gt_x'])
        gt_y = float(row['gt_y'])

        # Reference patch (full reference center)
        ref_patch = cv2.resize(ref_img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        ref_patch_f = ref_patch.astype(np.float32) / 255.0

        # Sample type: 50% positive pair, 50% negative pair during training
        is_positive = True
        if self.is_training and self.rng.rand() > 0.5:
            is_positive = False

        if is_positive:
            # Positive sample: Candidate center near ground truth (small jitter)
            if self.is_training:
                offset_x = float(self.rng.uniform(-10.0, 10.0))
                offset_y = float(self.rng.uniform(-10.0, 10.0))
            else:
                offset_x, offset_y = 0.0, 0.0

            cand_x = gt_x + offset_x
            cand_y = gt_y + offset_y

            # Ground truth offset to be predicted by X/Y head
            dx_gt = float(gt_x - cand_x)
            dy_gt = float(gt_y - cand_y)
            label = 1.0

        else:
            # Negative sample (Hard or Easy)
            if self.rng.rand() < self.hard_neg_ratio:
                # Hard negative: nearby repeating structure 50px to 200px away
                angle = self.rng.uniform(0, 2 * np.pi)
                dist = self.rng.uniform(50.0, 250.0)
                cand_x = float(np.clip(gt_x + dist * np.cos(angle), 50.0, 950.0))
                cand_y = float(np.clip(gt_y + dist * np.sin(angle), 50.0, 950.0))
            else:
                # Easy negative: random location away from GT
                cand_x = float(self.rng.uniform(50.0, 950.0))
                cand_y = float(self.rng.uniform(50.0, 950.0))

            dx_gt = 0.0
            dy_gt = 0.0
            label = 0.0

        # Extract candidate patch around (cand_x, cand_y) with nominal footprint 100x100
        cand_patch = self._crop_and_resize(search_img, cand_x, cand_y, crop_w=100, crop_h=100)
        cand_patch_f = cand_patch.astype(np.float32) / 255.0

        # Independent stochastic augmentation for training
        if self.is_training:
            ref_rng = np.random.RandomState(self.rng.randint(0, 1000000))
            search_rng = np.random.RandomState(self.rng.randint(0, 1000000))

            ref_patch_f = self._independent_augment(ref_patch_f, ref_rng)
            cand_patch_f = self._independent_augment(cand_patch_f, search_rng)

        # Convert to PyTorch tensors (1, H, W)
        ref_t = torch.from_numpy(ref_patch_f).unsqueeze(0)
        cand_t = torch.from_numpy(cand_patch_f).unsqueeze(0)
        label_t = torch.tensor([label], dtype=torch.float32)
        dx_t = torch.tensor([dx_gt], dtype=torch.float32)
        dy_t = torch.tensor([dy_gt], dtype=torch.float32)

        return {
            "ref_patch": ref_t,
            "cand_patch": cand_t,
            "label": label_t,
            "dx": dx_t,
            "dy": dy_t,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "cand_x": cand_x,
            "cand_y": cand_y,
            "pair_id": str(row['pair_id'])
        }
