"""
Regenerates ablation_summary_table.txt and ablation_summary.csv directly
from the raw per-episode JSON files in ablation_results/.
"""

import json
import os
import glob
import numpy as np

VARIANTS = ["full", "no_ga", "no_repulsion", "floor"]
BASE_DIR = os.path.abspath("ablation_results")

def main():
    all_variant_results = {}

    for var in VARIANTS:
        var_dir = os.path.join(BASE_DIR, var)
        records = []
        for i in range(1, 31):
            fpath = os.path.join(var_dir, f"run_{i:02d}.json")
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r") as f:
                data = json.load(f)
            ep = data.get("episode", {})
            rec = {
                "variant": var,
                "run_id": i,
                "success": ep.get("mission_completed", True),
                "total_steps": ep.get("total_steps", 0),
                "total_reward": round(float(ep.get("total_reward", 0.0)), 2),
                "total_path_length": round(float(ep.get("total_path_length", 0.0)), 2),
                "total_collisions": int(ep.get("total_collisions", 0)),
                "total_energy_consumed": round(float(ep.get("total_energy_consumed", 0.0)), 2),
                "wall_time_seconds": round(float(ep.get("wall_time_seconds", 0.0)), 2)
            }
            records.append(rec)
        all_variant_results[var] = records

    # Write combined CSV
    csv_path = os.path.join(BASE_DIR, "ablation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("variant,run_id,success,total_steps,total_reward,total_path_length,total_collisions,total_energy_consumed,wall_time_seconds\n")
        for var, records in all_variant_results.items():
            for r in records:
                f.write(f"{r['variant']},{r['run_id']},{r['success']},{r['total_steps']},{r['total_reward']},{r['total_path_length']},{r['total_collisions']},{r['total_energy_consumed']},{r['wall_time_seconds']}\n")

    # Generate summary report table
    table_path = os.path.join(BASE_DIR, "ablation_summary_table.txt")
    with open(table_path, "w") as f:
        f.write("=========================================================================================================\n")
        f.write("                          TRISAR MULTI-AGENT ABLATION STUDY BENCHMARK SUMMARY REPORT                     \n")
        f.write("=========================================================================================================\n\n")

        for var, records in all_variant_results.items():
            valid_runs = [r for r in records if r["success"]]
            num_success = len(valid_runs)
            num_failed = len(records) - num_success

            f.write(f"--- VARIANT: {var.upper()} ---\n")
            f.write(f"  Total Runs Requested: {len(records)}\n")
            f.write(f"  Successful Runs     : {num_success} ({num_success / len(records) * 100:.1f}%)\n")
            f.write(f"  Failed/Timed Out    : {num_failed}\n\n")

            if valid_runs:
                steps_arr = [r["total_steps"] for r in valid_runs]
                reward_arr = [r["total_reward"] for r in valid_runs]
                path_arr = [r["total_path_length"] for r in valid_runs]
                collisions_arr = [r["total_collisions"] for r in valid_runs]
                energy_arr = [r["total_energy_consumed"] for r in valid_runs]
                time_arr = [r["wall_time_seconds"] for r in valid_runs]

                f.write(f"  Metric                      Mean         Std Dev      Min          Max\n")
                f.write(f"  -----------------------------------------------------------------------\n")
                f.write(f"  Execution Steps           : {np.mean(steps_arr):<12.2f} {np.std(steps_arr):<12.2f} {np.min(steps_arr):<12.2f} {np.max(steps_arr):<12.2f}\n")
                f.write(f"  Episode Reward            : {np.mean(reward_arr):<12.2f} {np.std(reward_arr):<12.2f} {np.min(reward_arr):<12.2f} {np.max(reward_arr):<12.2f}\n")
                f.write(f"  Total Path Length (m)     : {np.mean(path_arr):<12.2f} {np.std(path_arr):<12.2f} {np.min(path_arr):<12.2f} {np.max(path_arr):<12.2f}\n")
                f.write(f"  Total Collisions          : {np.mean(collisions_arr):<12.2f} {np.std(collisions_arr):<12.2f} {np.min(collisions_arr):<12.2f} {np.max(collisions_arr):<12.2f}\n")
                f.write(f"  Total Energy (%)          : {np.mean(energy_arr):<12.2f} {np.std(energy_arr):<12.2f} {np.min(energy_arr):<12.2f} {np.max(energy_arr):<12.2f}\n")
                f.write(f"  Wall-Clock Time (s)       : {np.mean(time_arr):<12.2f} {np.std(time_arr):<12.2f} {np.min(time_arr):<12.2f} {np.max(time_arr):<12.2f}\n")
            f.write("\n" + "-" * 75 + "\n\n")

    print(f"Summary table successfully regenerated at: {table_path}")

if __name__ == "__main__":
    main()
