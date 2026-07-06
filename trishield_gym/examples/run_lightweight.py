"""
Example: Run the TriShield simulation with the lightweight backend.

Demonstrates the Gymnasium API with random actions on the default scenario.
Renders a live 3D visualization and exports metrics to CSV.

Usage:
    python trishield_gym/examples/run_lightweight.py
"""

import sys
import os
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trishield_gym import DroneSwarmEnv, MapConfig


def main():
    print("=" * 60)
    print("  TriShield Gym — Lightweight Backend Demo")
    print("=" * 60)

    # Create environment with the default TriShield scenario
    map_config = MapConfig.default_trishield()
    env = DroneSwarmEnv(
        backend="lightweight",
        map_config=map_config,
        action_mode="velocity",
        render_mode="rgb_array",  # Change to 'human' for live window
    )

    print(f"\nDrones: {env.drone_ids}")
    print(f"Action mode: {env.action_mode}")
    print(f"Max steps: {map_config.max_steps}")
    print(f"Threats: {[t['id'] for t in map_config.threats]}")
    print(f"Victims: {[v['id'] for v in map_config.victims]}")

    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial observations keys: {list(obs.keys())}")
    for drone_id in env.drone_ids:
        sv = obs[drone_id]["state_vector"]
        cam = obs[drone_id]["camera"]
        print(f"  {drone_id}: state_vector shape={sv.shape}, camera shape={cam.shape}")

    # Run simulation with random actions
    total_reward = 0.0
    print(f"\nRunning simulation for up to {map_config.max_steps} steps...\n")

    for step in range(map_config.max_steps):
        # Generate random velocity actions for each drone
        actions = {
            drone_id: env.action_space[drone_id].sample()
            for drone_id in env.drone_ids
        }

        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1:4d} | Reward: {reward:+8.2f} | "
                  f"Cumulative: {total_reward:+10.2f} | "
                  f"Collisions: {len(info.get('collisions', []))}")

        if terminated:
            print(f"\n  >>> Mission COMPLETED at step {step + 1}!")
            break
        if truncated:
            print(f"\n  >>> Episode TRUNCATED at step {step + 1} (max steps reached)")
            break

    # Print metrics summary
    summary = env.get_metrics_summary()
    print("\n" + "=" * 60)
    print("  Episode Summary")
    print("=" * 60)
    for key, value in summary["episode"].items():
        print(f"  {key:30s}: {value}")

    print(f"\n  {'Drone':<10} {'Path':>8} {'Collisions':>11} {'Energy':>8} {'Battery':>9} {'Avg Speed':>10}")
    print("  " + "-" * 58)
    for drone_id, data in summary["drones"].items():
        print(f"  {drone_id:<10} {data['path_length']:>8.1f} {data['collisions']:>11} "
              f"{data['energy_consumed']:>8.1f} {data['final_battery']:>8.1f}% {data['average_speed']:>10.2f}")

    # Export metrics to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "metrics_lightweight.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    env.export_metrics_csv(csv_path)
    print(f"\n  Metrics exported to: {os.path.abspath(csv_path)}")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
