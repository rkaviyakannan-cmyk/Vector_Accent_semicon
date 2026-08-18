"""
build_pairs.py

Wrapper script to execute prepare_dataset and create_splits in sequence.
"""

from prepare_dataset import prepare_dataset
from create_splits import create_splits

def build_pairs():
    print("=== Step 1: Preparing Master Dataset Manifest ===")
    manifest_csv = prepare_dataset()
    
    print("\n=== Step 2: Creating Train/Validation/Test Manifests ===")
    create_splits(manifest_csv)
    
if __name__ == "__main__":
    build_pairs()
