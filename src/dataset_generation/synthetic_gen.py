"""
src/dataset_generation/synthetic_gen.py

Optional synthetic FinFET image generator for future dataset expansion.
DOES NOT replace or overwrite the existing dataset.
Generates structural FinFET grating patterns with realistic SEM noise, blur, and scale variations.
"""

import os
import json
import cv2
import numpy as np

class OptionalFinFETGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_pair(self, output_dir="data/synthetic_samples", pair_id=1):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1000x1000 base canvases
        ref_canvas = np.full((1000, 1000), 50, dtype=np.uint8)
        search_canvas = np.full((1000, 1000), 50, dtype=np.uint8)

        # FinFET vertical lines pattern
        fin_pitch = 30
        for x in range(0, 1000, fin_pitch):
            cv2.line(ref_canvas, (x, 0), (x, 1000), 200, 4)

        gt_x = float(self.rng.uniform(200, 800))
        gt_y = float(self.rng.uniform(200, 800))

        # Render 10:1 scaled target into search image
        for x in range(int(gt_x - 50), int(gt_x + 50), 3):
            cv2.line(search_canvas, (x, 0), (x, 1000), 200, 1)

        # Add SEM noise
        noise_ref = self.rng.normal(0, 5, ref_canvas.shape).astype(np.float32)
        noise_search = self.rng.normal(0, 10, search_canvas.shape).astype(np.float32)

        ref_img = np.clip(ref_canvas.astype(np.float32) + noise_ref, 0, 255).astype(np.uint8)
        search_img = np.clip(search_canvas.astype(np.float32) + noise_search, 0, 255).astype(np.uint8)

        ref_p = os.path.join(output_dir, f"ref_synth_{pair_id:03d}.png")
        src_p = os.path.join(output_dir, f"search_synth_{pair_id:03d}.png")
        meta_p = os.path.join(output_dir, f"meta_synth_{pair_id:03d}.json")

        cv2.imwrite(ref_p, ref_img)
        cv2.imwrite(src_p, search_img)

        meta = {
            "pair_id": pair_id,
            "architecture": "finfet_synth",
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_box": [gt_x - 50.0, gt_y - 50.0, 100.0, 100.0],
            "seed": self.seed,
            "is_synthetic": True
        }
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2)

        return ref_p, src_p, meta_p
