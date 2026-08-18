"""
prepare_dataset.py

Scans the dataset directory (FINALDATASET/data), validates all image and metadata pairs,
and builds the master manifest CSV file at data/manifests/all_pairs.csv.
"""

import os
import glob
import json
import pandas as pd
from PIL import Image

def get_pair_id(path, prefix):
    base = os.path.basename(path)
    num_str = base.replace(prefix, '').split('.')[0]
    return f"{int(num_str):03d}"

def prepare_dataset(data_dir="FINALDATASET/data", output_dir="data/manifests"):
    os.makedirs(output_dir, exist_ok=True)
    
    ref_files = glob.glob(os.path.join(data_dir, "references", "*"))
    search_files = glob.glob(os.path.join(data_dir, "search_images", "*"))
    meta_files = glob.glob(os.path.join(data_dir, "metadata", "*"))
    
    ref_map = {get_pair_id(p, "reference_"): p for p in ref_files}
    search_map = {get_pair_id(p, "search_"): p for p in search_files}
    meta_map = {get_pair_id(p, "metadata_"): p for p in meta_files}
    
    all_ids = sorted(list(set(ref_map.keys()) | set(search_map.keys()) | set(meta_map.keys())))
    
    records = []
    for pid in all_ids:
        if pid not in ref_map or pid not in search_map or pid not in meta_map:
            print(f"Warning: Pair ID {pid} is incomplete. Missing: "
                  f"ref={pid not in ref_map}, search={pid not in search_map}, meta={pid not in meta_map}")
            continue
            
        ref_path = ref_map[pid]
        search_path = search_map[pid]
        meta_path = meta_map[pid]
        
        # Validate readability
        try:
            with Image.open(ref_path) as img:
                w_ref, h_ref = img.size
            with Image.open(search_path) as img:
                w_src, h_src = img.size
        except Exception as e:
            print(f"Error reading images for pair {pid}: {e}")
            continue
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        gt_x = float(meta['gt_x'])
        gt_y = float(meta['gt_y'])
        gt_box = meta.get('gt_box', [gt_x - 50.0, gt_y - 50.0, 100.0, 100.0])
        gt_x1, gt_y1, box_w, box_h = gt_box
        gt_x2 = gt_x1 + box_w
        gt_y2 = gt_y1 + box_h
        
        arch = meta.get('architecture', 'unknown')
        seed = meta.get('seed', 0)
        
        record = {
            "pair_id": pid,
            "reference_path": os.path.relpath(ref_path).replace("\\", "/"),
            "search_path": os.path.relpath(search_path).replace("\\", "/"),
            "metadata_path": os.path.relpath(meta_path).replace("\\", "/"),
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_x1": gt_x1,
            "gt_y1": gt_y1,
            "gt_x2": gt_x2,
            "gt_y2": gt_y2,
            "box_width": box_w,
            "box_height": box_h,
            "architecture": arch,
            "seed": seed,
            "ref_width": w_ref,
            "ref_height": h_ref,
            "search_width": w_src,
            "search_height": h_src,
            "detector_noise_ref": meta.get("detector_noise_sigma_ref", 2.0),
            "detector_noise_search": meta.get("detector_noise_sigma_search", 5.0)
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    out_csv = os.path.join(output_dir, "all_pairs.csv")
    df.to_csv(out_csv, index=False)
    print(f"Successfully processed {len(df)} pairs. Master manifest saved to {out_csv}")
    return out_csv

if __name__ == "__main__":
    prepare_dataset()
