#!/usr/bin/env python3
"""
Prepares master CSV files from raw JSON episode records in ablation_results/.
"""

import os
import json
import glob
import pandas as pd

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ablation_results"))
    variants = ["full", "no_ga", "no_repulsion", "floor"]
    rows = []

    for var in variants:
        folder = os.path.join(base_dir, var)
        files = sorted(glob.glob(os.path.join(folder, "run_*.json")))
        for fpath in files:
            run_num = int(os.path.basename(fpath).replace("run_", "").replace(".json", ""))
            with open(fpath) as f:
                data = json.load(f)
            ep = data.get("episode", {})
            rows.append({
                "variant": var,
                "run": run_num,
                "total_steps": ep.get("total_steps"),
                "total_path_length": round(float(ep.get("total_path_length", 0)), 2),
                "total_energy_consumed (%)": round(float(ep.get("total_energy_consumed", 0)), 2)
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(base_dir, "raw_metrics_all_variants.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Prepare Data] Saved {len(df)} rows to: {out_csv}")

if __name__ == "__main__":
    main()
