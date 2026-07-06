"""
Example: Run the TriShield simulation with the AirSim backend.

Demonstrates that the SAME algorithm code works with a different backend.
The only change is the constructor argument: backend='airsim'.

Prerequisites:
    - AirSim running in Unreal Engine
    - pip install airsim
    - settings.json configured with matching vehicle names (UAV_1, UAV_2, etc.)

Usage:
    python trishield_gym/examples/run_airsim.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trishield_gym import DroneSwarmEnv, MapConfig


def main():
    print("=" * 60)
    print("  TriShield Gym — AirSim Backend Demo")
    print("=" * 60)

    # --- THE ONLY DIFFERENCE: backend='airsim' ---
    map_config = MapConfig.default_trishield()
    env = DroneSwarmEnv(
        backend="airsim",
        map_config=map_config,
        action_mode="velocity",
        render_mode=None,  # AirSim has its own Unreal Engine renderer
    )

    print(f"\nConnected to AirSim!")
    print(f"Drones: {env.drone_ids}")

    # Reset
    obs, info = env.reset()
    print("Environment reset. Starting simulation...\n")

    # Run with random actions
    total_reward = 0.0
    for step in range(200):  # Shorter run for AirSim (slower real-time)
        actions = {
            drone_id: np.array([0.5, 0.0, 0.0], dtype=np.float32)
            for drone_id in env.drone_ids
        }

        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1:4d} | Reward: {reward:+8.2f} | "
                  f"Cumulative: {total_reward:+10.2f}")

        if terminated or truncated:
            print(f"\n  >>> Episode ended at step {step + 1}")
            break

    # Print summary
    summary = env.get_metrics_summary()
    print("\n  Episode Summary:")
    for key, value in summary["episode"].items():
        print(f"    {key}: {value}")

    env.close()
    print("\nDisconnected from AirSim. Done!")


if __name__ == "__main__":
    main()
