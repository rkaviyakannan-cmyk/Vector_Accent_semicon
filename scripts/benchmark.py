"""
benchmark.py

Runtime benchmarking script for FinFET Drift-Sense Localization System.
Measures component timings (ms per image pair) and system hardware specifications.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import platform
import json
import cv2
import torch
import numpy as np

from src.localization import FinFETLocalizer

def get_system_info():
    info = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "device": "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"
    }
    return info

def run_benchmark(num_pairs=10, checkpoint_path="models/checkpoints/best_model.pt", output_dir="results/runtime"):
    os.makedirs(output_dir, exist_ok=True)
    
    sys_info = get_system_info()
    localizer = FinFETLocalizer(model_path=checkpoint_path if os.path.exists(checkpoint_path) else None)

    ref_path = "FINALDATASET/data/references/reference_001.png"
    search_path = "FINALDATASET/data/search_images/search_001.png"

    if not (os.path.exists(ref_path) and os.path.exists(search_path)):
        print("Benchmark target images not found. Skipping benchmark.")
        return

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    # Warmup
    _ = localizer.localize(ref_img, search_img)

    runtimes_ms = []
    for _ in range(num_pairs):
        t0 = time.time()
        _ = localizer.localize(ref_img, search_img)
        dt = (time.time() - t0) * 1000.0
        runtimes_ms.append(dt)

    mean_rt = float(np.mean(runtimes_ms))
    median_rt = float(np.median(runtimes_ms))
    std_rt = float(np.std(runtimes_ms))

    report = {
        "system_info": sys_info,
        "num_trials": num_pairs,
        "mean_runtime_ms": mean_rt,
        "median_runtime_ms": median_rt,
        "std_runtime_ms": std_rt
    }

    report_json = os.path.join(output_dir, "runtime_benchmark.json")
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n========================================")
    print(" FINFET RUNTIME BENCHMARK")
    print("========================================")
    print(f"Device             : {sys_info['device']}")
    print(f"Python Version     : {sys_info['python_version']}")
    print(f"PyTorch Version    : {sys_info['pytorch_version']}")
    print(f"OpenCV Version     : {sys_info['opencv_version']}")
    print(f"----------------------------------------")
    print(f"Average Runtime    : {mean_rt:.2f} ms / pair")
    print(f"Median Runtime     : {median_rt:.2f} ms / pair")
    print(f"Std Deviation      : {std_rt:.2f} ms")
    print("========================================\n")

    return report

if __name__ == "__main__":
    run_benchmark()
