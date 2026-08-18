"""
src/evaluation/metrics.py

Evaluation metrics calculation for FinFET Drift-Sense localization:
- Mean Error, Median Error, Worst-Case Error, Standard Deviation
- Threshold accuracy: Pass@5, Pass@4, Pass@2, Pass@1, Pass@0.5
- Top-1 success and Top-3 candidate coverage
"""

import numpy as np
import pandas as pd

def calculate_metrics(predictions_df):
    """
    Computes summary metrics from a predictions DataFrame.
    DataFrame must contain: 'gt_x', 'gt_y', 'pred_x', 'pred_y', 'top1_success', 'top3_success' (optional).
    """
    df = predictions_df.copy()
    
    if "error_px" not in df.columns:
        df["error_px"] = np.hypot(df["pred_x"] - df["gt_x"], df["pred_y"] - df["gt_y"])

    errors = df["error_px"].values
    n_total = len(errors)

    if n_total == 0:
        return {}

    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    worst_err = float(np.max(errors))
    std_err = float(np.std(errors))

    pass_5 = float(np.mean(errors <= 5.0))
    pass_4 = float(np.mean(errors <= 4.0))
    pass_2 = float(np.mean(errors <= 2.0))
    pass_1 = float(np.mean(errors <= 1.0))
    pass_0_5 = float(np.mean(errors <= 0.5))

    top1_acc = float(df["top1_success"].mean()) if "top1_success" in df.columns else pass_5
    top3_acc = float(df["top3_success"].mean()) if "top3_success" in df.columns else pass_5

    return {
        "count": n_total,
        "mean_error_px": mean_err,
        "median_error_px": median_err,
        "worst_error_px": worst_err,
        "std_error_px": std_err,
        "pass_5": pass_5,
        "pass_4": pass_4,
        "pass_2": pass_2,
        "pass_1": pass_1,
        "pass_0_5": pass_0_5,
        "top1_success": top1_acc,
        "top3_success": top3_acc
    }

def print_metrics_summary(metrics_dict, title="EVALUATION SUMMARY"):
    print(f"\n========================================")
    print(f" {title}")
    print(f"========================================")
    print(f"Total Samples Evaluated : {metrics_dict.get('count', 0)}")
    print(f"Mean Error             : {metrics_dict.get('mean_error_px', 0.0):.4f} px")
    print(f"Median Error           : {metrics_dict.get('median_error_px', 0.0):.4f} px")
    print(f"Worst-Case Error       : {metrics_dict.get('worst_error_px', 0.0):.4f} px")
    print(f"Std Deviation          : {metrics_dict.get('std_error_px', 0.0):.4f} px")
    print(f"----------------------------------------")
    print(f"Pass@5  (<= 5.0 px)    : {metrics_dict.get('pass_5', 0.0)*100:.2f}%")
    print(f"Pass@4  (<= 4.0 px)    : {metrics_dict.get('pass_4', 0.0)*100:.2f}%")
    print(f"Pass@2  (<= 2.0 px)    : {metrics_dict.get('pass_2', 0.0)*100:.2f}%")
    print(f"Pass@1  (<= 1.0 px)    : {metrics_dict.get('pass_1', 0.0)*100:.2f}%")
    print(f"Pass@0.5(<= 0.5 px)    : {metrics_dict.get('pass_0_5', 0.0)*100:.2f}%")
    print(f"----------------------------------------")
    print(f"Top-1 Success Rate     : {metrics_dict.get('top1_success', 0.0)*100:.2f}%")
    print(f"Top-3 Candidate Coverage: {metrics_dict.get('top3_success', 0.0)*100:.2f}%")
    print(f"========================================\n")
