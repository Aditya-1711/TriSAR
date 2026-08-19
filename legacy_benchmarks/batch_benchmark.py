import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="TriShield Multi-Agent Swarm Batch Benchmark Runner")
    parser.add_argument("--runs", type=int, default=30, help="Number of benchmark runs (default: 30)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per run in seconds (default: 600)")
    args = parser.parse_args()

    results_dir = os.path.abspath("benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    report_source = os.path.abspath(os.path.join("logs", "gazebo_run_report.json"))

    print("=" * 70)
    print(f"  TriShield Swarm Autonomous Benchmark Runner ({args.runs} Episodes)")
    print("=" * 70)
    print(f" -> Scenario: 5 Drones (UAV_1..3 Search/Rescue/3D SLAM, UAV_4..5 Landed Mesh Relays)")
    print(f" -> Target Count: 8 Survivor Targets (Fixed)")
    print(f" -> Output Directory: {results_dir}")
    print("=" * 70 + "\n")

    run_records = []

    for i in range(1, args.runs + 1):
        print(f"[{i:02d}/{args.runs:02d}] Executing Simulation Episode {i}...")
        start_t = time.time()
        success = False
        error_msg = None

        try:
            env_vars = dict(os.environ)
            env_vars["BATCH_MODE"] = "1"
            res = subprocess.run(
                [sys.executable, os.path.join("trishield_gym", "examples", "run_gazebo.py")],
                capture_output=True,
                text=True,
                env=env_vars,
                timeout=args.timeout
            )
            elapsed = round(time.time() - start_t, 2)
            if res.returncode == 0 and os.path.exists(report_source):
                dest_json = os.path.join(results_dir, f"run_{i:02d}.json")
                shutil.copy(report_source, dest_json)
                with open(dest_json, "r") as f:
                    data = json.load(f)
                
                ep = data.get("episode", {})
                success = ep.get("mission_completed", True)
                rec = {
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
                print(f"  -> Run {i:02d} Completed: Steps={rec['total_steps']}, Collisions={rec['total_collisions']}, Time={rec['wall_time_seconds']}s")
            else:
                error_msg = f"Return Code {res.returncode}"
                print(f"  -> Run {i:02d} Failed: {error_msg}")
                run_records.append({
                    "run_id": i, "success": False, "total_steps": 0, "total_reward": 0.0,
                    "total_path_length": 0.0, "total_collisions": -1, "total_energy_consumed": 0.0,
                    "wall_time_seconds": elapsed
                })
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - start_t, 2)
            print(f"  -> Run {i:02d} Timed Out (> {args.timeout}s)")
            run_records.append({
                "run_id": i, "success": False, "total_steps": 0, "total_reward": 0.0,
                "total_path_length": 0.0, "total_collisions": -1, "total_energy_consumed": 0.0,
                "wall_time_seconds": elapsed
            })

    # Write summary.csv
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("run_id,success,total_steps,total_reward,total_path_length,total_collisions,total_energy_consumed,wall_time_seconds\n")
        for r in run_records:
            f.write(f"{r['run_id']},{r['success']},{r['total_steps']},{r['total_reward']},{r['total_path_length']},{r['total_collisions']},{r['total_energy_consumed']},{r['wall_time_seconds']}\n")

    # Generate summary_table.txt
    valid_runs = [r for r in run_records if r["success"]]
    num_success = len(valid_runs)
    num_failed = len(run_records) - num_success

    if valid_runs:
        steps_arr = [r["total_steps"] for r in valid_runs]
        reward_arr = [r["total_reward"] for r in valid_runs]
        path_arr = [r["total_path_length"] for r in valid_runs]
        collisions_arr = [r["total_collisions"] for r in valid_runs]
        energy_arr = [r["total_energy_consumed"] for r in valid_runs]
        time_arr = [r["wall_time_seconds"] for r in valid_runs]

        total_collisions_sum = sum(collisions_arr)

        table_str = "=" * 70 + "\n"
        table_str += "          TRISHIELD SWARM BATCH BENCHMARK SUMMARY REPORT\n"
        table_str += "=" * 70 + "\n"
        table_str += f" Total Runs Executed       : {args.runs}\n"
        table_str += f" Successful Runs (Rescued)  : {num_success} ({num_success / args.runs * 100:.1f}%)\n"
        table_str += f" Failed / Timed Out Runs   : {num_failed}\n"
        table_str += f" Active Drones Fleet Size  : 5 Drones (3 Search/SLAM, 2 Landed Relays)\n"
        table_str += f" Fixed Survivor Targets    : 8 Victims\n"
        table_str += f" Total Recorded Collisions : {total_collisions_sum}\n"
        table_str += "-" * 70 + "\n"
        table_str += f"{'Metric':<25} | {'Mean':<10} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}\n"
        table_str += "-" * 70 + "\n"
        table_str += f"{'Total Steps':<25} | {np.mean(steps_arr):<10.2f} | {np.std(steps_arr):<10.2f} | {np.min(steps_arr):<10.2f} | {np.max(steps_arr):<10.2f}\n"
        table_str += f"{'Total Path Length (m)':<25} | {np.mean(path_arr):<10.2f} | {np.std(path_arr):<10.2f} | {np.min(path_arr):<10.2f} | {np.max(path_arr):<10.2f}\n"
        table_str += f"{'Total Energy (%)':<25} | {np.mean(energy_arr):<10.2f} | {np.std(energy_arr):<10.2f} | {np.min(energy_arr):<10.2f} | {np.max(energy_arr):<10.2f}\n"
        table_str += f"{'Episode Reward':<25} | {np.mean(reward_arr):<10.2f} | {np.std(reward_arr):<10.2f} | {np.min(reward_arr):<10.2f} | {np.max(reward_arr):<10.2f}\n"
        table_str += f"{'Wall Time (s)':<25} | {np.mean(time_arr):<10.2f} | {np.std(time_arr):<10.2f} | {np.min(time_arr):<10.2f} | {np.max(time_arr):<10.2f}\n"
        table_str += "=" * 70 + "\n"
    else:
        table_str = "No runs completed successfully.\n"

    txt_path = os.path.join(results_dir, "summary_table.txt")
    with open(txt_path, "w") as f:
        f.write(table_str)

    print("\n" + table_str)
    print(f"Benchmark Results Saved to:\n - {csv_path}\n - {txt_path}\n")


if __name__ == "__main__":
    main()
