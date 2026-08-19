#!/usr/bin/env python3
"""
Generates Figure 4 3D Trajectory Comparison Plots for TAROS 2026 Paper.
"""

import os
import json
import matplotlib.pyplot as plt

def plot_trajectory_from_json(json_path, output_png_path, title):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(10, 8), dpi=200, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('X (m)', fontsize=11, labelpad=8)
    ax.set_ylabel('Y (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Z (Altitude m)', fontsize=11, labelpad=8)

    color_map = {
        'UAV_1': '#1f77b4', 'UAV_2': '#ff7f0e',
        'UAV_3': '#2ca02c', 'UAV_4': '#d62728', 'UAV_5': '#9467bd'
    }

    for drone_id, path in sorted(data.items()):
        path_arr = list(path)
        if not path_arr:
            continue
        xs = [p[0] for p in path_arr]
        ys = [p[1] for p in path_arr]
        zs = [p[2] for p in path_arr]
        color = color_map.get(drone_id, '#1f77b4')
        ax.plot(xs, ys, zs, color=color, linewidth=2.0, label=drone_id)

    ax.legend(loc='upper right', frameon=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figure 4 Saved] Written to: {output_png_path}")

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_json = os.path.join(repo_root, "logs", "trajectory_data_20260818_232136.json")
    norep_json = os.path.join(repo_root, "logs", "trajectory_data_20260818_232206.json")

    fig_dir = os.path.join(repo_root, "figures")
    plot_trajectory_from_json(full_json, os.path.join(fig_dir, "figure4a_full_trajectory.png"), "Figure 4a: Swarm 3D Trajectory (FULL - Collision-Free)")
    plot_trajectory_from_json(norep_json, os.path.join(fig_dir, "figure4b_norepulsion_trajectory.png"), "Figure 4b: Swarm 3D Trajectory (NO_REPULSION - Collision)")

if __name__ == "__main__":
    main()
