"""
evaluate.py

Evaluation CLI script for FinFET Drift-Sense Localization System.
Evaluates model on validation or test manifests, outputs predictions CSV,
calculates Pass@N metrics, saves visual overlays, and renders failure analysis.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cv2
import json
import numpy as np
import pandas as pd

from src.localization import FinFETLocalizer
from src.evaluation.metrics import calculate_metrics, print_metrics_summary
from src.utils.visualization import draw_overlay, render_failure_case

def evaluate(manifest_path="data/manifests/test.csv",
             checkpoint_path="models/checkpoints/best_model.pt",
             config_path="models/model_config.json",
             output_dir="results"):

    print("=== FinFET Drift-Sense Evaluation ===")
    print(f"Manifest: {manifest_path}")
    print(f"Checkpoint: {checkpoint_path}")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest {manifest_path} not found.")

    df_manifest = pd.read_csv(manifest_path)

    # Load model config if exists
    model_cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_cfg = json.load(f)

    # Initialize Localizer
    localizer = FinFETLocalizer(
        model_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
        model_config=model_cfg
    )

    pred_records = []
    overlay_dir = os.path.join(output_dir, "overlays")
    failure_dir = os.path.join(output_dir, "failures")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    for idx, row in df_manifest.iterrows():
        pair_id = str(row['pair_id'])
        ref_path = row['reference_path']
        search_path = row['search_path']
        gt_x = float(row['gt_x'])
        gt_y = float(row['gt_y'])

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            print(f"Warning: Could not read image pair for {pair_id}")
            continue

        result = localizer.localize(ref_img, search_img)
        pred_x = result['x']
        pred_y = result['y']
        error_px = float(np.hypot(pred_x - gt_x, pred_y - gt_y))

        pass_5 = 1 if error_px <= 5.0 else 0
        pass_4 = 1 if error_px <= 4.0 else 0
        pass_2 = 1 if error_px <= 2.0 else 0
        pass_1 = 1 if error_px <= 1.0 else 0

        # Check if GT is covered in top 3 candidates
        top3_covered = 0
        for c in result['candidates']:
            if np.hypot(c['x'] - gt_x, c['y'] - gt_y) <= 20.0:
                top3_covered = 1
                break

        rec = {
            "pair_id": pair_id,
            "reference_path": ref_path,
            "search_path": search_path,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_px": error_px,
            "candidate_rank": result['rank'],
            "similarity_score": result['similarity'],
            "scale": result['scale'],
            "rotation": result['rotation'],
            "top1_success": pass_5,
            "top3_success": top3_covered,
            "pass_5": pass_5,
            "pass_4": pass_4,
            "pass_2": pass_2,
            "pass_1": pass_1
        }
        pred_records.append(rec)

        # Draw overlay
        overlay = draw_overlay(search_img, gt_x, gt_y, pred_x, pred_y, pair_id, error_px)
        cv2.imwrite(os.path.join(overlay_dir, f"overlay_{pair_id}.png"), overlay)

        # Save failure diagnostic case if error > 5.0 px
        if error_px > 5.0:
            render_failure_case(search_img, ref_img, result, gt_x, gt_y,
                                save_path=os.path.join(failure_dir, f"failure_{pair_id}.png"))

    df_preds = pd.DataFrame(pred_records)
    preds_csv = os.path.join(output_dir, "predictions", "predictions.csv")
    os.makedirs(os.path.dirname(preds_csv), exist_ok=True)
    df_preds.to_csv(preds_csv, index=False)

    metrics = calculate_metrics(df_preds)
    print_metrics_summary(metrics, title=f"EVALUATION METRICS ({len(df_preds)} Pairs)")

    # Save metrics JSON
    metrics_json = os.path.join(output_dir, "metrics", "evaluation_metrics.json")
    os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Check minimum test size requirement
    if len(df_preds) < 30:
        print("\nWARNING: The available independent test set contains fewer than 30 pairs.")
        print("This does not satisfy the hackathon's recommended minimum validation size of 30 pairs.")

    print(f"Evaluation complete! Results saved to {preds_csv}")
    return metrics, df_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FinFET Drift-Sense Localizer")
    parser.add_argument("--manifest", default="data/manifests/test.csv", help="Manifest CSV to evaluate")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--config", default="models/model_config.json", help="Model config JSON")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    evaluate(args.manifest, args.checkpoint, args.config, args.output_dir)
