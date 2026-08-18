"""
create_splits.py

Performs pair-level stratified train/validation/test splitting (70% / 15% / 15%)
with deterministic seed (default: 42) and duplicate reference pattern hash grouping.
Outputs: train.csv, validation.csv, test.csv
"""

import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

def get_image_hash(img_path):
    with Image.open(img_path) as img:
        arr = np.array(img)
    return hashlib.md5(arr.tobytes()).hexdigest()

def create_splits(manifest_csv="data/manifests/all_pairs.csv",
                  output_dir="data/manifests",
                  seed=42,
                  train_ratio=0.70,
                  val_ratio=0.15,
                  test_ratio=0.15):
    
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest CSV {manifest_csv} does not exist. Run prepare_dataset.py first.")
        
    df = pd.read_csv(manifest_csv)
    
    # Compute image hash to group identical reference patterns and avoid leakage
    df['ref_hash'] = df['reference_path'].apply(get_image_hash)
    
    # Group unique reference hashes along with their architecture
    hash_df = df.groupby('ref_hash').agg({
        'pair_id': list,
        'architecture': 'first'
    }).reset_index()
    
    np.random.seed(seed)
    
    train_hashes, val_hashes, test_hashes = [], [], []
    
    # Stratify by architecture across unique reference pattern groups
    for arch, group in hash_df.groupby('architecture'):
        hashes = group['ref_hash'].values
        np.random.shuffle(hashes)
        
        n_arch = len(hashes)
        n_train = max(1, int(round(n_arch * train_ratio)))
        n_val = max(1, int(round(n_arch * val_ratio)))
        
        train_h = hashes[:n_train]
        val_h = hashes[n_train:n_train + n_val]
        test_h = hashes[n_train + n_val:]
        
        # If test set empty due to rounding, re-allocate
        if len(test_h) == 0 and len(train_h) > 2:
            test_h = train_h[-1:]
            train_h = train_h[:-1]
            
        train_hashes.extend(train_h)
        val_hashes.extend(val_h)
        test_hashes.extend(test_h)
        
    train_df = df[df['ref_hash'].isin(train_hashes)].drop(columns=['ref_hash'])
    val_df = df[df['ref_hash'].isin(val_hashes)].drop(columns=['ref_hash'])
    test_df = df[df['ref_hash'].isin(test_hashes)].drop(columns=['ref_hash'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "validation.csv")
    test_csv = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Splits successfully created with seed {seed}:")
    print(f"  Train: {len(train_df)} pairs ({len(train_df)/len(df)*100:.1f}%) -> {train_csv}")
    print(f"  Val:   {len(val_df)} pairs ({len(val_df)/len(df)*100:.1f}%) -> {val_csv}")
    print(f"  Test:  {len(test_df)} pairs ({len(test_df)/len(df)*100:.1f}%) -> {test_csv}")
    
    return train_csv, val_csv, test_csv

if __name__ == "__main__":
    create_splits()
