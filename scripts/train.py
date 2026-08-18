"""
train.py

Training entry point CLI script for the FinFET Drift-Sense Siamese Model.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import pandas as pd
import torch

from src.siamese import FinFETSiameseNet
from src.training import FinFETSiameseDataset, FinFETTrainer

def train_model(config_path="configs/train.yaml",
                train_manifest="data/manifests/train.csv",
                val_manifest="data/manifests/validation.csv",
                output_dir="models/checkpoints"):
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})
    r_cfg = cfg.get("replay", {})

    print("=== FinFET Siamese Network Training ===")
    print(f"Config: {config_path}")
    print(f"Train Manifest: {train_manifest}")
    print(f"Val Manifest: {val_manifest}")

    df_train = pd.read_csv(train_manifest)
    df_val = pd.read_csv(val_manifest)

    input_size = m_cfg.get("input_size", 128)
    seed = cfg.get("seed", 42)

    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = FinFETSiameseDataset(df_train, input_size=input_size, is_training=True, seed=seed)
    val_ds = FinFETSiameseDataset(df_val, input_size=input_size, is_training=False, seed=seed)

    model = FinFETSiameseNet(
        in_channels=1,
        use_cbam=m_cfg.get("use_cbam", True),
        use_cir=m_cfg.get("use_cir", True),
        use_dual_correlation=m_cfg.get("use_dual_correlation", True),
        use_xy_feedback=m_cfg.get("use_xy_feedback", True)
    )

    trainer = FinFETTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=output_dir,
        batch_size=t_cfg.get("batch_size", 16),
        epochs=t_cfg.get("epochs", 15),
        learning_rate=t_cfg.get("learning_rate", 3e-4),
        weight_decay=t_cfg.get("weight_decay", 1e-4),
        gradient_clip_norm=t_cfg.get("gradient_clip_norm", 5.0),
        lambda_similarity=t_cfg.get("similarity_loss_weight", 1.0),
        lambda_coordinate=t_cfg.get("coordinate_loss_weight", 1.0),
        use_replay=r_cfg.get("enabled", True),
        replay_capacity=r_cfg.get("capacity", 2000),
        replay_ratio=r_cfg.get("replay_ratio", 0.25)
    )

    hist_csv = trainer.train()
    print(f"Training finished! Logs saved to {hist_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FinFET Siamese Model")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to config yaml")
    parser.add_argument("--train-manifest", default="data/manifests/train.csv", help="Train manifest CSV")
    parser.add_argument("--val-manifest", default="data/manifests/validation.csv", help="Val manifest CSV")
    parser.add_argument("--output-dir", default="models/checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    train_model(args.config, args.train_manifest, args.val_manifest, args.output_dir)
