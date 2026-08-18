"""
src/training/trainer.py

Complete PyTorch Trainer for FinFET Drift-Sense Siamese Network.
Features:
- NaN/Inf assertion guards and safe batch skipping
- Gradient clipping
- Learning rate scheduling & early stopping
- Hard-Negative Error Replay Memory integration
- Best/Last checkpoint saving and JSON configuration serialization
"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .loss import FinFETCombinedLoss
from .replay import ReplayMemory

class FinFETTrainer:
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 output_dir="models/checkpoints",
                 batch_size=16,
                 epochs=20,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 gradient_clip_norm=5.0,
                 lambda_similarity=1.0,
                 lambda_coordinate=1.0,
                 use_replay=True,
                 replay_capacity=2000,
                 replay_ratio=0.25,
                 device=None):
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_clip_norm = gradient_clip_norm
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.loss_fn = FinFETCombinedLoss(lambda_similarity=lambda_similarity,
                                          lambda_coordinate=lambda_coordinate).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        
        self.use_replay = use_replay
        if use_replay:
            self.replay = ReplayMemory(capacity=replay_capacity, replay_ratio=replay_ratio)
        else:
            self.replay = None
            
        os.makedirs(self.output_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_sim_loss, total_coord_loss = 0.0, 0.0, 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            ref = batch["ref_patch"].to(self.device)
            cand = batch["cand_patch"].to(self.device)
            label = batch["label"].to(self.device)
            gt_dx = batch["dx"].to(self.device)
            gt_dy = batch["dy"].to(self.device)

            # NaN / Inf guard
            if not (torch.isfinite(ref).all() and torch.isfinite(cand).all()):
                print(f"[Epoch {epoch}] Warning: Skipping batch with non-finite inputs!")
                continue

            self.optimizer.zero_grad()
            
            sim, pred_dx, pred_dy = self.model(ref, cand)
            
            loss, loss_sim, loss_coord = self.loss_fn(sim, pred_dx, pred_dy, label, gt_dx, gt_dy)

            # Replay Memory Integration
            if self.use_replay and self.replay is not None:
                # Add hard negatives to memory
                for i in range(ref.size(0)):
                    self.replay.add(ref[i], cand[i], sim[i].item(),
                                    batch["cand_x"][i].item(), batch["cand_y"][i].item(),
                                    batch["gt_x"][i].item(), batch["gt_y"][i].item())
                
                # Sample and compute replay loss if memory has enough samples
                replay_batch = self.replay.sample(int(round(ref.size(0) * self.replay.replay_ratio)))
                if replay_batch is not None:
                    r_ref = replay_batch["ref_patch"].to(self.device)
                    r_cand = replay_batch["cand_patch"].to(self.device)
                    r_label = replay_batch["label"].to(self.device)
                    r_gt_dx = replay_batch["dx"].to(self.device)
                    r_gt_dy = replay_batch["dy"].to(self.device)

                    r_sim, r_dx, r_dy = self.model(r_ref, r_cand)
                    r_loss, _, _ = self.loss_fn(r_sim, r_dx, r_dy, r_label, r_gt_dx, r_gt_dy)
                    loss = loss + 0.5 * r_loss

            if not torch.isfinite(loss):
                print(f"[Epoch {epoch}] Warning: NaN/Inf detected in loss! Skipping batch update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_sim_loss += loss_sim.item()
            total_coord_loss += loss_coord.item()
            n_batches += 1

        n_batches = max(1, n_batches)
        return total_loss / n_batches, total_sim_loss / n_batches, total_coord_loss / n_batches

    def validate(self):
        self.model.eval()
        total_loss, total_sim_loss, total_coord_loss = 0.0, 0.0, 0.0
        n_batches = 0
        correct_sim = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                ref = batch["ref_patch"].to(self.device)
                cand = batch["cand_patch"].to(self.device)
                label = batch["label"].to(self.device)
                gt_dx = batch["dx"].to(self.device)
                gt_dy = batch["dy"].to(self.device)

                sim, pred_dx, pred_dy = self.model(ref, cand)
                loss, loss_sim, loss_coord = self.loss_fn(sim, pred_dx, pred_dy, label, gt_dx, gt_dy)

                if torch.isfinite(loss):
                    total_loss += loss.item()
                    total_sim_loss += loss_sim.item()
                    total_coord_loss += loss_coord.item()
                    n_batches += 1

                preds_binary = (sim >= 0.5).float()
                correct_sim += (preds_binary == label).sum().item()
                total_samples += label.size(0)

        n_batches = max(1, n_batches)
        accuracy = correct_sim / max(1, total_samples)
        return total_loss / n_batches, total_sim_loss / n_batches, total_coord_loss / n_batches, accuracy

    def train(self):
        print(f"Starting training on device: {self.device}")
        history = []
        best_val_loss = float("inf")
        
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_sim, train_coord = self.train_epoch(epoch)
            val_loss, val_sim, val_coord, val_acc = self.validate()
            self.scheduler.step()
            dt = time.time() - t0
            
            print(f"Epoch {epoch:02d}/{self.epochs:02d} [{dt:.1f}s] - "
                  f"Train Loss: {train_loss:.4f} (Sim: {train_sim:.4f}, Coord: {train_coord:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (Sim: {val_sim:.4f}, Coord: {val_coord:.4f}, Acc: {val_acc*100:.1f}%)")

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_sim_loss": train_sim,
                "train_coord_loss": train_coord,
                "val_loss": val_loss,
                "val_sim_loss": val_sim,
                "val_coord_loss": val_coord,
                "val_accuracy": val_acc,
                "lr": self.scheduler.get_last_lr()[0]
            })

            # Checkpoint saving
            last_ckpt = os.path.join(self.output_dir, "last_model.pt")
            torch.save(self.model.state_dict(), last_ckpt)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = os.path.join(self.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_ckpt)

        # Save history CSV
        df_hist = pd.DataFrame(history)
        hist_csv = os.path.join(self.output_dir, "training_history.csv")
        df_hist.to_csv(hist_csv, index=False)
        
        # Save model config JSON
        cfg = {
            "backbone": "mobilenet_v3_small",
            "in_channels": 1,
            "input_size": self.train_dataset.input_size,
            "use_cbam": self.model.use_cbam,
            "use_cir": self.model.use_cir,
            "use_dual_correlation": self.model.use_dual_correlation,
            "use_xy_feedback": self.model.use_xy_feedback,
            "best_val_loss": best_val_loss,
            "epochs": self.epochs
        }
        cfg_path = os.path.join(self.output_dir, "model_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"Training completed successfully! Checkpoints saved to {self.output_dir}")
        return hist_csv
