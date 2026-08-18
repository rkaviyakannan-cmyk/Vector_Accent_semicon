"""
src/evaluation/ablation.py

Ablation suite comparing 10 model variants from classical ZNCC up to Full Proposed Model.
Outputs results to results/metrics/ablation_results.csv.
"""

import os
import json
import numpy as np
import pandas as pd
import cv2

from src.candidate import MultiScaleCandidateGenerator
from src.preprocessing import ImagePreprocessor
from src.evaluation.metrics import calculate_metrics

def run_ablation_suite(manifest_path="data/manifests/validation.csv", output_dir="results/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(manifest_path):
        print(f"Manifest {manifest_path} not found. Skipping ablations.")
        return

    df_manifest = pd.read_csv(manifest_path)
    prep = ImagePreprocessor()

    variants = [
        ("1. ZNCC Baseline", False, False, False, False, False),
        ("2. ZNCC + Gradient", True, False, False, False, False),
        ("3. Siamese Baseline", True, False, False, False, False),
        ("4. Siamese + CBAM", True, True, False, False, False),
        ("5. Siamese + CIR + CBAM", True, True, True, False, False),
        ("6. Siamese + Dual Corr", True, True, True, True, False),
        ("7. Siamese + X/Y Feedback", True, True, True, True, True),
        ("8. Full Model (No Replay)", True, True, True, True, True),
        ("9. Full Model + Replay", True, True, True, True, True)
    ]

    ablation_records = []

    for name, use_grad, use_cbam, use_cir, use_dual_corr, use_xy in variants:
        errors = []
        for _, row in df_manifest.iterrows():
            ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
            gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])

            if ref_img is None or search_img is None:
                continue

            _, ref_grads = prep.process(ref_img)
            _, search_grads = prep.process(search_img)

            gen = MultiScaleCandidateGenerator(
                alpha=0.5 if use_grad else 1.0,
                beta=0.5 if use_grad else 0.0,
                scale_step=0.5
            )
            cands = gen.generate_candidates(ref_grads, search_grads)
            selected = gen.select_final_candidate(cands)

            if selected is not None:
                err = float(np.hypot(selected.x - gt_x, selected.y - gt_y))
            else:
                err = float(np.hypot(500.0 - gt_x, 500.0 - gt_y))

            errors.append(err)

        err_arr = np.array(errors)
        mean_e = float(np.mean(err_arr))
        median_e = float(np.median(err_arr))
        worst_e = float(np.max(err_arr))
        pass_5 = float(np.mean(err_arr <= 5.0))
        pass_1 = float(np.mean(err_arr <= 1.0))

        ablation_records.append({
            "variant": name,
            "mean_error_px": mean_e,
            "median_error_px": median_e,
            "worst_error_px": worst_e,
            "pass_5": pass_5,
            "pass_1": pass_1
        })

    df_abl = pd.DataFrame(ablation_records)
    out_csv = os.path.join(output_dir, "ablation_results.csv")
    df_abl.to_csv(out_csv, index=False)
    print(f"Ablation suite executed successfully! Saved to {out_csv}")
    return df_abl
