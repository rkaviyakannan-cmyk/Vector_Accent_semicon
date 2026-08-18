"""
infer.py

Inference CLI script for FinFET Drift-Sense Localization System.
Supports single-pair inference and batch manifest inference.
DOES NOT USE GROUND TRUTH DURING INFERENCE.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.localization import FinFETLocalizer

def save_infer_visualization(ref_img, search_img, result, viz_path, ref_path="", search_path="", gt_x=None, gt_y=None):
    """Renders a comprehensive multi-panel localization visualization for single-pair inference."""
    os.makedirs(os.path.dirname(os.path.abspath(viz_path)), exist_ok=True)
    
    fig = plt.figure(figsize=(14, 11), dpi=200)
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1.5, 1.2])

    pred_x, pred_y = result['x'], result['y']
    scale = result['scale']
    rot = result['rotation']
    rank = result['rank']
    sim = result['similarity']
    candidates = result.get('candidates', [])

    # Candidate box footprint in search image
    box_w = scale * 10.0
    box_h = scale * 10.0
    if result.get('selected_candidate'):
        sc = result['selected_candidate']
        if 'candidate_obj' in sc and hasattr(sc['candidate_obj'], 'width'):
            box_w = sc['candidate_obj'].width
            box_h = sc['candidate_obj'].height

    # Panel 1: Reference Image
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.imshow(ref_img, cmap='gray')
    ax_ref.set_title(f"Reference Image\n{os.path.basename(ref_path) if ref_path else ''}", fontsize=10, fontweight='bold')
    ax_ref.axis('off')

    # Panel 2: Full Search Image with GT, Prediction, Bounding Box, Top-3 candidates
    ax_search = fig.add_subplot(gs[0, 1:])
    ax_search.imshow(search_img, cmap='gray')

    # Predicted Bounding Box (Cyan rectangle)
    rect_x = pred_x - box_w / 2.0
    rect_y = pred_y - box_h / 2.0
    rect = patches.Rectangle((rect_x, rect_y), box_w, box_h, linewidth=2, edgecolor='cyan', facecolor='none', label='Predicted BBox')
    ax_search.add_patch(rect)

    # Predicted Location (Cyan Crosshair)
    ax_search.plot(pred_x, pred_y, 'cx', markersize=14, markeredgewidth=2, label=f"Prediction ({pred_x:.1f}, {pred_y:.1f})")

    # Top-3 candidates
    for c in candidates:
        cx_c = c.get('x', c.get('raw_x', 0.0))
        cy_c = c.get('y', c.get('raw_y', 0.0))
        r_c = c.get('rank', 1)
        score_c = c.get('fused_score', c.get('similarity', 0.0))
        color_c = 'cyan' if r_c == rank else 'yellow'
        ax_search.plot(cx_c, cy_c, marker='o', color=color_c, markersize=7)
        ax_search.text(cx_c + 12, cy_c + 12, f"R{r_c} ({score_c:.2f})", color='yellow', fontsize=8.5, fontweight='bold')

    # Ground Truth Marker if available
    if gt_x is not None and gt_y is not None:
        ax_search.plot(gt_x, gt_y, 'g*', markersize=14, label=f"GT ({gt_x:.1f}, {gt_y:.1f})")
        ax_search.plot([gt_x, pred_x], [gt_y, pred_y], 'r--', linewidth=2, label=f"Error: {np.hypot(pred_x-gt_x, pred_y-gt_y):.2f} px")
        gt_str = f"x = {gt_x:.2f}, y = {gt_y:.2f}"
    else:
        gt_str = "Not Available"

    ax_search.set_title("Full Search Image (Predicted Location & Top-3 Candidates)", fontsize=10, fontweight='bold')
    ax_search.legend(loc='upper right', fontsize=8.5)
    ax_search.axis('off')

    # Panel 3: Zoomed Crop extracted ACTUALLY from Search Image
    ax_crop = fig.add_subplot(gs[1, 0])
    h_s, w_s = search_img.shape[:2]
    x1 = max(0, int(round(pred_x - box_w / 2.0)))
    y1 = max(0, int(round(pred_y - box_h / 2.0)))
    x2 = min(w_s, int(round(pred_x + box_w / 2.0)))
    y2 = min(h_s, int(round(pred_y + box_h / 2.0)))
    search_crop = search_img[y1:y2, x1:x2]
    if search_crop.size == 0:
        search_crop = np.zeros((128, 128), dtype=np.uint8)
    else:
        search_crop = cv2.resize(search_crop, (128, 128))

    ax_crop.imshow(search_crop, cmap='gray')
    ax_crop.set_title(f"Detected Search Crop\n(Centered at {pred_x:.1f}, {pred_y:.1f})", fontsize=9.5, fontweight='bold')
    ax_crop.axis('off')

    # Panel 4: Reference Crop vs Detected Search Crop Side-by-Side
    ax_ref_crop = fig.add_subplot(gs[1, 1])
    ref_crop = cv2.resize(ref_img, (128, 128))
    ax_ref_crop.imshow(ref_crop, cmap='gray')
    ax_ref_crop.set_title("Reference Crop\n(Normalized)", fontsize=9.5, fontweight='bold')
    ax_ref_crop.axis('off')

    ax_comp = fig.add_subplot(gs[1, 2])
    comp_diff = cv2.absdiff(ref_crop, search_crop)
    ax_comp.imshow(comp_diff, cmap='inferno')
    ax_comp.set_title("Abs Intensity Difference\n(Ref vs Search Crop)", fontsize=9.5, fontweight='bold')
    ax_comp.axis('off')

    # Panel 5: Diagnostic Summary Info Box
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    cand_str = ", ".join([f"R{c['rank']}:({c['x']:.1f},{c['y']:.1f},s={c['fused_score']:.3f})" for c in candidates[:3]]) if candidates else "N/A"

    info_text = (
        f"FINFET LOCALIZATION INFERENCE VISUALIZATION\n"
        f"--------------------------------------------------------------------------------------------------------\n"
        f"Predicted Location (x, y) : ({pred_x:.2f} px, {pred_y:.2f} px) | Ground Truth : {gt_str}\n"
        f"Similarity Score          : {sim:.4f}                          | Selected Rank : Rank {rank}\n"
        f"Estimated Scale           : {scale:.2f} : 1                        | Estimated Rot : {rot:.2f} deg\n"
        f"Top-3 Candidates          : {cand_str}\n"
        f"Visualization File Saved  : {os.path.abspath(viz_path)}"
    )

    ax_text.text(0.01, 0.15, info_text, fontsize=9.5, family='monospace',
                 bbox=dict(facecolor='#e1f5fe', alpha=0.9, boxstyle='round,pad=0.6'))

    plt.tight_layout()
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization successfully saved to: {os.path.abspath(viz_path)}")

def run_single_inference(ref_path, search_path, checkpoint_path, config_path=None, gt_x=None, gt_y=None, save_visualization=False, output_path=None):
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image {ref_path} not found.")
    if not os.path.exists(search_path):
        raise FileNotFoundError(f"Search image {search_path} not found.")

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    model_cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            model_cfg = json.load(f)

    localizer = FinFETLocalizer(
        model_path=checkpoint_path if (checkpoint_path and os.path.exists(checkpoint_path)) else None,
        model_config=model_cfg
    )

    t0 = time.time()
    result = localizer.localize(ref_img, search_img)
    dt_ms = (time.time() - t0) * 1000.0

    print("========================================")
    print("FINFET DRIFT-SENSE LOCALIZATION")
    print("========================================")
    print(f"Reference:\n{ref_path}\n")
    print(f"Search:\n{search_path}\n")
    print(f"Predicted Centre:")
    print(f"x = {result['x']:8.2f} px")
    print(f"y = {result['y']:8.2f} px\n")
    print(f"Similarity Score:")
    print(f"{result['similarity']:.4f}\n")
    print(f"Estimated Scale:")
    print(f"{result['scale']:.2f} : 1\n")
    print(f"Estimated Rotation:")
    print(f"{result['rotation']:.2f} degrees\n")
    print(f"Selected Candidate:")
    print(f"Rank {result['rank']}\n")
    print(f"Runtime:")
    print(f"{dt_ms:.2f} ms")

    if gt_x is not None and gt_y is not None:
        err = float(np.hypot(result['x'] - gt_x, result['y'] - gt_y))
        print(f"\nGround Truth:")
        print(f"x = {gt_x:8.2f}")
        print(f"y = {gt_y:8.2f}\n")
        print(f"Localization Error:")
        print(f"{err:.2f} px\n")
        print(f"Pass@5: {'YES' if err <= 5.0 else 'NO'}")
        print(f"Pass@4: {'YES' if err <= 4.0 else 'NO'}")
        print(f"Pass@2: {'YES' if err <= 2.0 else 'NO'}")
        print(f"Pass@1: {'YES' if err <= 1.0 else 'NO'}")
    else:
        print(f"\nGround Truth:\nNot Available\n")

    print("========================================")

    if save_visualization:
        viz_out = output_path if (output_path and (output_path.endswith('.png') or output_path.endswith('.jpg'))) else "results/overlays/visualization.png"
        save_infer_visualization(ref_img, search_img, result, viz_out, ref_path=ref_path, search_path=search_path, gt_x=gt_x, gt_y=gt_y)

    return result

def run_batch_inference(input_manifest, output_csv, checkpoint_path, config_path=None):
    df_in = pd.read_csv(input_manifest)
    model_cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            model_cfg = json.load(f)

    localizer = FinFETLocalizer(
        model_path=checkpoint_path if (checkpoint_path and os.path.exists(checkpoint_path)) else None,
        model_config=model_cfg
    )

    out_records = []
    for idx, row in df_in.iterrows():
        ref_p = row['reference_path']
        search_p = row['search_path']
        ref_img = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_p, cv2.IMREAD_GRAYSCALE)

        t0 = time.time()
        res = localizer.localize(ref_img, search_img)
        dt_ms = (time.time() - t0) * 1000.0

        rec = {
            "reference_path": ref_p,
            "search_path": search_p,
            "pred_x": res['x'],
            "pred_y": res['y'],
            "similarity_score": res['similarity'],
            "scale": res['scale'],
            "rotation": res['rotation'],
            "selected_rank": res['rank'],
            "runtime_ms": dt_ms
        }
        if "gt_x" in row and "gt_y" in row:
            rec["gt_x"] = float(row["gt_x"])
            rec["gt_y"] = float(row["gt_y"])
            rec["error_px"] = float(np.hypot(res['x'] - float(row['gt_x']), res['y'] - float(row['gt_y'])))
        out_records.append(rec)

    df_out = pd.DataFrame(out_records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Batch inference complete for {len(df_out)} pairs. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinFET Localization Inference CLI")
    parser.add_argument("--reference", help="Path to reference image PNG")
    parser.add_argument("--search", help="Path to search image PNG")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--config", default="models/model_config.json", help="Model config JSON")
    parser.add_argument("--input-manifest", help="Input manifest CSV for batch mode")
    parser.add_argument("--output", default="results/predictions.csv", help="Output path (CSV for batch mode, image PNG for visualization)")
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save localization visualization"
    )
    args = parser.parse_args()

    if args.reference and args.search:
        run_single_inference(
            args.reference,
            args.search,
            args.checkpoint,
            args.config,
            save_visualization=args.save_visualization,
            output_path=args.output
        )
    elif args.input_manifest:
        run_batch_inference(args.input_manifest, args.output, args.checkpoint, args.config)
    else:
        # Default run on pair 001 for demonstration
        print("No input provided. Running demonstration inference on pair 001...")
        run_single_inference(
            "FINALDATASET/data/references/reference_001.png",
            "FINALDATASET/data/search_images/search_001.png",
            args.checkpoint,
            args.config,
            save_visualization=args.save_visualization,
            output_path=args.output
        )
