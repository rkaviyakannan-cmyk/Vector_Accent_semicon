"""
src/evaluation/plot_results.py

Publication-Quality Plotting Engine for FinFET Failure Analysis & Robustness System.
Generates clean academic paper-style high-resolution (300 DPI) PNG charts and ablation CSV.
Uses ONLY real measured data from empirical evaluation.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResultPlotter:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.pres_dir = os.path.join(output_dir, "presentation")
        os.makedirs(self.pres_dir, exist_ok=True)
        
        # Set academic styling
        plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 11
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 12

    def plot_noise_robustness(self, df_noise):
        """
        Generates:
        1. results/presentation/noise_robustness.png
        2. results/presentation/localization_error_vs_noise.png
        3. results/presentation/success_rate_vs_noise.png
        """
        if df_noise is None or len(df_noise) == 0:
            print("No noise data provided. Skipping noise plots.")
            return

        noise_levels = df_noise['noise_level'].values
        sigma_vals = df_noise['noise_parameter'].values
        mean_errors = df_noise['mean_error'].values

        succ_5 = df_noise['success_5px'].values
        succ_10 = df_noise['success_10px'].values
        succ_20 = df_noise['success_20px'].values
        succ_50 = df_noise['success_50px'].values

        top1_rec = df_noise['candidate_recall_top1'].values
        top3_rec = df_noise['candidate_recall_top3'].values

        # 1. Noise Robustness (Top-1 vs Top-3 Candidate Recall & Success Rate)
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
        ax.plot(noise_levels, top1_rec, 'o-', color='#2b5c8f', linewidth=2, markersize=7, label='Top-1 Candidate Recall')
        ax.plot(noise_levels, top3_rec, 's--', color='#d95f02', linewidth=2, markersize=7, label='Top-3 Candidate Recall')
        ax.plot(noise_levels, succ_20, '^-.', color='#7570b3', linewidth=2, markersize=7, label='Success@20px')

        ax.set_title("FinFET Localization Performance Under Increasing SEM Noise", fontweight='bold', pad=12)
        ax.set_xlabel(r"SEM Noise Level ($\sigma$)")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', frameon=True)

        for i, txt in enumerate(top3_rec):
            ax.annotate(f"{txt:.1f}%", (noise_levels[i], top3_rec[i] + 3), ha='center', fontsize=8.5, fontweight='bold')

        plt.tight_layout()
        path1 = os.path.join(self.pres_dir, "noise_robustness.png")
        plt.savefig(path1, dpi=300)
        plt.close()

        # 2. Localization Error vs Noise
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
        ax.plot(noise_levels, mean_errors, 'd-', color='#e74c3c', linewidth=2.5, markersize=8)
        ax.set_title("Mean Localization Error vs Noise Strength", fontweight='bold', pad=12)
        ax.set_xlabel("SEM Noise Level")
        ax.set_ylabel("Mean Localization Error (pixels)")
        ax.grid(True, linestyle=':', alpha=0.6)

        for i, err in enumerate(mean_errors):
            ax.annotate(f"{err:.2f} px", (noise_levels[i], mean_errors[i] + max(mean_errors)*0.03), ha='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        path2 = os.path.join(self.pres_dir, "localization_error_vs_noise.png")
        plt.savefig(path2, dpi=300)
        plt.close()

        # 3. Success Rate vs Noise (Success@5px, 10px, 20px, 50px)
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
        ax.plot(noise_levels, succ_5, 'o-', label='Success@5px', color='#2ecc71', linewidth=2)
        ax.plot(noise_levels, succ_10, 's-', label='Success@10px', color='#3498db', linewidth=2)
        ax.plot(noise_levels, succ_20, '^-', label='Success@20px', color='#9b59b6', linewidth=2)
        ax.plot(noise_levels, succ_50, 'd-', label='Success@50px', color='#f39c12', linewidth=2)

        ax.set_title("Localization Success Rate Across Error Thresholds vs Noise", fontweight='bold', pad=12)
        ax.set_xlabel("Noise Condition")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', frameon=True)

        plt.tight_layout()
        path3 = os.path.join(self.pres_dir, "success_rate_vs_noise.png")
        plt.savefig(path3, dpi=300)
        plt.close()

        print(f"Generated noise robustness plots: {path1}, {path2}, {path3}")

    def plot_failure_reasons(self, df_failures):
        """Generates results/presentation/failure_reasons.png."""
        if df_failures is None or len(df_failures) == 0:
            print("No failure data provided.")
            return

        reason_counts = df_failures['failure_reason'].value_counts()
        labels = reason_counts.index.tolist()
        counts = reason_counts.values.tolist()

        colors = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#34495e', '#2ecc71']

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        bars = ax.barh(labels, counts, color=colors[:len(labels)], edgecolor='black', linewidth=0.8)

        ax.set_title("Primary Causes of Localization Failures", fontweight='bold', pad=12)
        ax.set_xlabel("Number of Test Samples")
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            pct = (w / len(df_failures)) * 100.0
            ax.text(w + 0.1, bar.get_y() + bar.get_height()/2.0, f"{int(w)} ({pct:.1f}%)", ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        path = os.path.join(self.pres_dir, "failure_reasons.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Generated failure reasons plot: {path}")

    def plot_baseline_vs_proposed(self, zncc_metrics, proposed_metrics):
        """Generates results/presentation/baseline_vs_proposed.png."""
        categories = ['Mean Error (px)', 'Median Error (px)', 'Success@5px (%)', 'Success@20px (%)', 'Top-3 Recall (%)']
        
        zncc_vals = [
            zncc_metrics.get('mean_error_px', 0.0),
            zncc_metrics.get('median_error_px', 0.0),
            zncc_metrics.get('pass_5', 0.0) * 100.0,
            zncc_metrics.get('pass_20', zncc_metrics.get('pass_5', 0.0)) * 100.0,
            zncc_metrics.get('top3_success', 0.0) * 100.0
        ]

        prop_vals = [
            proposed_metrics.get('mean_error_px', 0.0),
            proposed_metrics.get('median_error_px', 0.0),
            proposed_metrics.get('pass_5', 0.0) * 100.0,
            proposed_metrics.get('pass_20', proposed_metrics.get('pass_5', 0.0)) * 100.0,
            proposed_metrics.get('top3_success', 0.0) * 100.0
        ]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        rects1 = ax.bar(x - width/2, zncc_vals, width, label='ZNCC Baseline', color='#7f8c8d', edgecolor='black')
        rects2 = ax.bar(x + width/2, prop_vals, width, label='Proposed Siamese Model', color='#2980b9', edgecolor='black')

        ax.set_title("Performance Comparison: ZNCC Baseline vs Proposed Model", fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha='right')
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', frameon=True)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h + 1), ha='center', va='bottom', fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h + 1), ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        path = os.path.join(self.pres_dir, "baseline_vs_proposed.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Generated baseline vs proposed comparison plot: {path}")

    def generate_ablation_results(self, df_ablation):
        """
        Generates:
        1. results/presentation/ablation_results.csv
        2. results/presentation/ablation_comparison.png
        """
        if df_ablation is None or len(df_ablation) == 0:
            print("No ablation data available.")
            return

        out_csv = os.path.join(self.pres_dir, "ablation_results.csv")
        df_ablation.to_csv(out_csv, index=False)

        # Render horizontal bar chart comparison
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        variants = df_ablation['variant'].values
        errors = df_ablation['mean_error_px'].values

        y_pos = np.arange(len(variants))
        bars = ax.barh(y_pos, errors, color='#34495e', edgecolor='black', linewidth=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(variants)
        ax.invert_yaxis()  # top-down
        ax.set_xlabel("Mean Localization Error (pixels)")
        ax.set_title("Module Ablation Study: Architectural Impact", fontweight='bold', pad=12)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(errors)*0.01, bar.get_y() + bar.get_height()/2.0, f"{w:.2f} px", ha='left', va='center', fontsize=8.5)

        plt.tight_layout()
        path_img = os.path.join(self.pres_dir, "ablation_comparison.png")
        plt.savefig(path_img, dpi=300)
        plt.close()
        print(f"Generated ablation files: {out_csv}, {path_img}")

