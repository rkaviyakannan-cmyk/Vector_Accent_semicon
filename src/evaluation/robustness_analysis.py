"""
src/evaluation/robustness_analysis.py

Controlled SEM Noise Robustness Experiment for FinFET Drift-Sense Localization System.
Applies independent Gaussian noise realizations to reference and search test images,
evaluates complete candidate search + Siamese model + X/Y feedback pipeline,
and saves results to results/metrics/noise_robustness.csv.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd

from src.localization import FinFETLocalizer

class NoiseRobustnessAnalyzer:
    def __init__(self,
                 manifest_path="data/manifests/test.csv",
                 checkpoint_path="models/checkpoints/best_model.pt",
                 config_path="models/model_config.json",
                 output_dir="results",
                 noise_levels=None):
        
        self.manifest_path = manifest_path
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.output_dir = output_dir

        if noise_levels is None:
            self.noise_levels = [
                {"name": "Low Noise", "sigma": 5.0},
                {"name": "Medium Noise", "sigma": 15.0},
                {"name": "High Noise", "sigma": 30.0}
            ]
        else:
            self.noise_levels = noise_levels

        # Load model config
        self.model_cfg = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.model_cfg = json.load(f)

        # Initialize localizer
        self.localizer = FinFETLocalizer(
            model_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            model_config=self.model_cfg
        )

    def _add_independent_gaussian_noise(self, img_uint8, sigma, seed):
        """Applies independent Gaussian noise to image with uint8 clipping."""
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, img_uint8.shape)
        noisy_img = np.clip(img_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_img

    def run_experiment(self):
        """Runs noise robustness evaluation across all noise levels."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest {self.manifest_path} not found.")

        df_manifest = pd.read_csv(self.manifest_path)
        robustness_records = []

        noisy_img_dir = os.path.join(self.output_dir, "robustness", "noise")
        os.makedirs(noisy_img_dir, exist_ok=True)

        print("=== Running SEM Noise Robustness Experiment ===")

        for level in self.noise_levels:
            name = level["name"]
            sigma = level["sigma"]
            print(f"Evaluating {name} (sigma = {sigma:.1f})...")

            errors = []
            similarities = []
            runtimes = []
            top1_recalls = []
            top3_recalls = []

            for idx, row in df_manifest.iterrows():
                pair_id = str(row['pair_id'])
                ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
                search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
                gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])

                if ref_img is None or search_img is None:
                    continue

                # Independent noise realizations (ref_seed != search_seed)
                seed_ref = 42 + idx * 2
                seed_search = 42 + idx * 2 + 10000

                ref_noisy = self._add_independent_gaussian_noise(ref_img, sigma, seed_ref)
                search_noisy = self._add_independent_gaussian_noise(search_img, sigma, seed_search)

                # Save sample noisy image for inspection
                if idx == 0:
                    cv2.imwrite(os.path.join(noisy_img_dir, f"ref_noisy_{name.lower().replace(' ', '_')}_{pair_id}.png"), ref_noisy)
                    cv2.imwrite(os.path.join(noisy_img_dir, f"search_noisy_{name.lower().replace(' ', '_')}_{pair_id}.png"), search_noisy)

                # Measure runtime and run localization pipeline
                t0 = time.perf_counter()
                result = self.localizer.localize(ref_noisy, search_noisy)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                pred_x, pred_y = result['x'], result['y']
                error_px = float(np.hypot(pred_x - gt_x, pred_y - gt_y))

                errors.append(error_px)
                similarities.append(result['similarity'])
                runtimes.append(dt_ms)

                # Candidate recall evaluation (search tolerance = 20.0 px)
                candidates = result['candidates']
                in_top1 = False
                in_top3 = False
                for c in candidates:
                    dist = np.hypot(c['raw_x'] - gt_x, c['raw_y'] - gt_y)
                    if dist <= 20.0:
                        in_top3 = True
                        if c['rank'] == 1:
                            in_top1 = True

                top1_recalls.append(1.0 if in_top1 else 0.0)
                top3_recalls.append(1.0 if in_top3 else 0.0)

            err_arr = np.array(errors)
            n_samples = len(err_arr)

            mean_err = float(np.mean(err_arr))
            median_err = float(np.median(err_arr))

            succ_5 = float(np.mean(err_arr <= 5.0)) * 100.0
            succ_10 = float(np.mean(err_arr <= 10.0)) * 100.0
            succ_20 = float(np.mean(err_arr <= 20.0)) * 100.0
            succ_50 = float(np.mean(err_arr <= 50.0)) * 100.0

            top1_rec = float(np.mean(top1_recalls)) * 100.0
            top3_rec = float(np.mean(top3_recalls)) * 100.0

            rec = {
                "noise_level": name,
                "noise_parameter": sigma,
                "num_samples": n_samples,
                "mean_error": mean_err,
                "median_error": median_err,
                "success_5px": succ_5,
                "success_10px": succ_10,
                "success_20px": succ_20,
                "success_50px": succ_50,
                "candidate_recall_top1": top1_rec,
                "candidate_recall_top3": top3_rec,
                "mean_similarity": float(np.mean(similarities)),
                "mean_runtime_ms": float(np.mean(runtimes))
            }
            robustness_records.append(rec)

        df_robustness = pd.DataFrame(robustness_records)
        out_csv = os.path.abspath(os.path.join(self.output_dir, "metrics", "noise_robustness.csv"))
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w', encoding='utf-8') as f:
            df_robustness.to_csv(f, index=False)
        print(f"Saved noise robustness metrics to {out_csv}")

        return df_robustness

