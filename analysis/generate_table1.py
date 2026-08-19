#!/usr/bin/env python3
"""
Generates Table 1 summary statistics for TAROS 2026 paper submission.
"""

import os
import json
import glob
import numpy as np

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ablation_results"))
    variants = ["full", "no_ga", "no_repulsion", "floor"]

    print("=" * 70)
    print("  TABLE 1: TRISAR ABLATION STUDY BENCHMARK SUMMARY (N=30 per cell, ddof=0)")
    print("=" * 70)

    for var in variants:
        folder = os.path.join(base_dir, var)
        files = sorted(glob.glob(os.path.join(folder, "run_*.json")))
        steps, path, energy = [], [], []

        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)
            ep = data.get("episode", {})
            steps.append(ep.get("total_steps", 0))
            path.append(float(ep.get("total_path_length", 0.0)))
            energy.append(float(ep.get("total_energy_consumed", 0.0)))

        print(f"\n--- VARIANT: {var.upper()} ---")
        print(f"  Steps         : {np.mean(steps):6.2f} ± {np.std(steps, ddof=0):5.2f} (Min: {np.min(steps)}, Max: {np.max(steps)})")
        print(f"  Path Length(m): {np.mean(path):6.2f} ± {np.std(path, ddof=0):5.2f} (Min: {np.min(path):.2f}, Max: {np.max(path):.2f})")
        print(f"  Energy (%)    : {np.mean(energy):6.2f} ± {np.std(energy, ddof=0):5.2f} (Min: {np.min(energy):.2f}, Max: {np.max(energy):.2f})")

if __name__ == "__main__":
    main()
