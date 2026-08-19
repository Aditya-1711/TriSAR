"""
TriSAR Ablation Study Benchmark Evaluator
Runs 30 episodes sequentially across all 4 system variants:
1. full: Full TriSAR System (GA Allocation + PSO Building Avoidance & Inter-Agent Repulsion)
2. no_ga: Baseline 1 (Greedy / Round-Robin Allocation, PSO Active)
3. no_repulsion: Baseline 2 (GA Allocation Active, PSO Inter-Agent Repulsion Disabled)
4. floor: Baseline 3 (Naive Floor Baseline: No GA, No Inter-Agent Repulsion)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np


VARIANTS = ["full", "no_ga", "no_repulsion", "floor"]


def main():
    parser = argparse.ArgumentParser(description="TriSAR Swarm Ablation Study Evaluator")
    parser.add_argument("--runs", type=int, default=30, help="Number of benchmark runs per variant (default: 30)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per run in seconds (default: 600)")
    args = parser.parse_args()

    base_results_dir = os.path.abspath("ablation_results")
    os.makedirs(base_results_dir, exist_ok=True)
    report_source = os.path.abspath(os.path.join("logs", "gazebo_run_report.json"))

    print("=" * 75)
    print(f"  TriSAR Swarm Ablation Study Evaluator ({args.runs} Runs x {len(VARIANTS)} Variants = {args.runs * len(VARIANTS)} Total Episodes)")
    print("=" * 75)
    print(f" -> Variants: {', '.join(VARIANTS)}")
    print(f" -> Output Directory: {base_results_dir}")
    print("=" * 75 + "\n")

    all_variant_results = {}

    for var in VARIANTS:
        var_dir = os.path.join(base_results_dir, var)
        os.makedirs(var_dir, exist_ok=True)

        print(f"\n" + "=" * 70)
        print(f"  STARTING ABLATION VARIANT: {var.upper()}")
        print("=" * 70)

        run_records = []

        for i in range(1, args.runs + 1):
            print(f"[{var}] Episode [{i:02d}/{args.runs:02d}] Executing...")
            start_t = time.time()
            success = False
            error_msg = None

            try:
                env_vars = dict(os.environ)
                env_vars["BATCH_MODE"] = "1"
                env_vars["TRISAR_VARIANT"] = var

                res = subprocess.run(
                    [sys.executable, os.path.join("trishield_gym", "examples", "run_gazebo.py")],
                    capture_output=True,
                    text=True,
                    env=env_vars,
                    timeout=args.timeout
                )
                elapsed = round(time.time() - start_t, 2)

                if res.returncode == 0 and os.path.exists(report_source):
                    dest_json = os.path.join(var_dir, f"run_{i:02d}.json")
                    shutil.copy(report_source, dest_json)
                    with open(dest_json, "r") as f:
                        data = json.load(f)

                    ep = data.get("episode", {})
                    success = ep.get("mission_completed", True)
                    rec = {
                        "variant": var,
                        "run_id": i,
                        "success": success,
                        "total_steps": ep.get("total_steps", 0),
                        "total_reward": round(float(ep.get("total_reward", 0.0)), 2),
                        "total_path_length": round(float(ep.get("total_path_length", 0.0)), 2),
                        "total_collisions": int(ep.get("total_collisions", 0)),
                        "total_energy_consumed": round(float(ep.get("total_energy_consumed", 0.0)), 2),
                        "wall_time_seconds": round(float(ep.get("wall_time_seconds", elapsed)), 2)
                    }
                    run_records.append(rec)
                    print(f"  -> [{var}] Run {i:02d} Completed: Steps={rec['total_steps']}, Collisions={rec['total_collisions']}, Energy={rec['total_energy_consumed']}%, Time={rec['wall_time_seconds']}s")
                else:
                    error_msg = f"Return Code {res.returncode}"
                    print(f"  -> [{var}] Run {i:02d} Failed: {error_msg}")
                    run_records.append({
                        "variant": var, "run_id": i, "success": False, "total_steps": 0, "total_reward": 0.0,
                        "total_path_length": 0.0, "total_collisions": -1, "total_energy_consumed": 0.0,
                        "wall_time_seconds": elapsed
                    })
            except subprocess.TimeoutExpired:
                elapsed = round(time.time() - start_t, 2)
                print(f"  -> [{var}] Run {i:02d} Timed Out (> {args.timeout}s)")
                run_records.append({
                    "variant": var, "run_id": i, "success": False, "total_steps": 0, "total_reward": 0.0,
                    "total_path_length": 0.0, "total_collisions": -1, "total_energy_consumed": 0.0,
                    "wall_time_seconds": elapsed
                })

        all_variant_results[var] = run_records

    # Write combined CSV
    csv_path = os.path.join(base_results_dir, "ablation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("variant,run_id,success,total_steps,total_reward,total_path_length,total_collisions,total_energy_consumed,wall_time_seconds\n")
        for var, records in all_variant_results.items():
            for r in records:
                f.write(f"{r['variant']},{r['run_id']},{r['success']},{r['total_steps']},{r['total_reward']},{r['total_path_length']},{r['total_collisions']},{r['total_energy_consumed']},{r['wall_time_seconds']}\n")

    # Generate summary report table
    table_path = os.path.join(base_results_dir, "ablation_summary_table.txt")
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

    print("\n" + "=" * 75)
    print(f"  TRISAR ABLATION STUDY COMPLETE!")
    print(f"  Combined Summary CSV : {csv_path}")
    print(f"  Summary Report Table : {table_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
