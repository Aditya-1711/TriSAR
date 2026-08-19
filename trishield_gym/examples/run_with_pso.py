"""
Example: Bridge existing TriShield PSO/GA algorithms into the Gymnasium env.

This script shows how your EXISTING trishield_core algorithms (PSOOptimizer,
HeterogeneousGA) plug directly into the new Gymnasium framework. The core
algorithm logic is unchanged — only wrapped in the env.step() loop.

Usage:
    python trishield_gym/examples/run_with_pso.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trishield_gym import DroneSwarmEnv, MapConfig
from trishield_core.blackboard import Blackboard
from trishield_core.ga_allocator import HeterogeneousGA
from trishield_core.pso_optimizer import PSOOptimizer
from trishield_core.agent import UAVAgent, UGVAgent


def main():
    print("=" * 60)
    print("  TriShield Gym — PSO/GA Integration Demo")
    print("  (Existing algorithms -> Gymnasium environment)")
    print("=" * 60)

    # Create environment
    map_config = MapConfig.default_trishield()
    env = DroneSwarmEnv(
        backend="airsim",
        map_config=map_config,
        action_mode="velocity",
        render_mode=None,
        enable_cameras=False,  # Massive speedup since PSO doesn't need images
    )

    print("Connecting to AirSim and taking off (this takes a few seconds)...")
    # Reset environment FIRST so targets are randomized before task allocation
    obs, info = env.reset()

    # Initialize the existing trishield_core algorithms
    bb = Blackboard()
    pso = PSOOptimizer(safe_distance=3.0)

    # Create agent objects that mirror the env's drones (for PSO/GA compatibility)
    core_agents = {}
    for dc in map_config.drone_configs:
        if dc["type"] == "uav":
            agent = UAVAgent(dc["id"], dc["spawn"])
        else:
            agent = UGVAgent(dc["id"], dc["spawn"])
        core_agents[dc["id"]] = agent
        bb.broadcast_state(agent)

    # Register threats and victims on the blackboard (using updated positions)
    for threat in env.map_config.threats:
        bb.register_threat(threat["id"], threat["pos"], threat.get("type", "unknown"))
    for victim in env.map_config.victims:
        bb.register_victim(victim["id"], victim["pos"], victim.get("urgency", 5))

    # Run GA task allocation (one-time)
    ga = HeterogeneousGA(bb)
    assignments = ga.allocate()

    print(f"\n--- GA Task Allocation ---")
    for task_id, agent_id in assignments.items():
        print(f"  Task '{task_id}' -> {agent_id}")

        # Apply assignments to core agents
        agent = core_agents[agent_id]
        if task_id in bb.threats:
            agent.assigned_task = bb.threats[task_id]["pos"]
        elif task_id in bb.victims:
            agent.assigned_task = bb.victims[task_id]["pos"]
        agent.mission_status = "ACTIVE"
    print("-" * 30)

    print(f"\nRunning PSO-driven simulation for {map_config.max_steps} steps...\n")
    total_reward = 0.0

    for step in range(map_config.max_steps):
        actions = {}

        for drone_id in env.drone_ids:
            agent = core_agents[drone_id]

            # Sync agent position from environment observations
            state_vec = obs[drone_id]["state_vector"]
            agent.pos = state_vec[:3].astype(float)
            agent.vel = state_vec[3:6].astype(float)
            agent.battery = float(state_vec[6]) * 100.0
            bb.broadcast_state(agent)

            # Compute PSO velocity using existing algorithm
            all_agents_list = list(core_agents.values())
            # Combine restricted_zones and obstacles for collision avoidance
            rz_list = [
                {"pos": rz["pos"], "radius": rz["radius"]}
                for rz in map_config.restricted_zones
            ]
            for obstacle in map_config.obstacles:
                rz_list.append({"pos": obstacle.position, "radius": obstacle.radius})

            new_vel = pso.compute_velocity(
                agent, all_agents_list,
                restricted_zones=rz_list,
                wind_vector=map_config.wind_vector,
            )

            # Normalize velocity to [-1, 1] for the Gymnasium action space
            max_speed = agent.max_speed
            normalized_vel = np.clip(new_vel / max_speed, -1.0, 1.0).astype(np.float32)
            actions[drone_id] = normalized_vel

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if (step + 1) % 50 == 0:
            # Show drone positions
            positions = {did: obs[did]["state_vector"][:3] for did in env.drone_ids}
            print(f"  Step {step + 1:4d} | Reward: {reward:+8.2f} | Total: {total_reward:+10.2f}")
            for did, pos in positions.items():
                bat = obs[did]["state_vector"][6] * 100
                print(f"    {did}: pos=[{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}] bat={bat:.0f}%")

        if terminated:
            print(f"\n  >>> MISSION COMPLETE at step {step + 1}!")
            break
        if truncated:
            print(f"\n  >>> Episode truncated at step {step + 1}")
            break

    # Print metrics
    summary = env.get_metrics_summary()
    print("\n" + "=" * 60)
    print("  Episode Summary (PSO/GA Driven)")
    print("=" * 60)
    for key, value in summary["episode"].items():
        print(f"  {key:30s}: {value}")

    print(f"\n  {'Drone':<10} {'Path':>8} {'Collisions':>11} {'Energy':>8} {'Battery':>9}")
    print("  " + "-" * 48)
    for drone_id, data in summary["drones"].items():
        print(f"  {drone_id:<10} {data['path_length']:>8.1f} {data['collisions']:>11} "
              f"{data['energy_consumed']:>8.1f} {data['final_battery']:>8.1f}%")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export metrics
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", f"metrics_pso_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    env.export_metrics_csv(csv_path)
    print(f"\n  Metrics exported to: {os.path.abspath(csv_path)}")

    # Plot 3D Trajectory
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("PSO Swarm 3D Trajectory")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (Altitude m)")
        
        # Access internal trails safely
        backend = env.unwrapped._backend if hasattr(env, 'unwrapped') else env._backend
        if hasattr(backend, 'trails'):
            for drone_id, trail in backend.trails.items():
                if len(trail) > 0:
                    trail_arr = np.array(trail)
                    ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2], label=drone_id, linewidth=2)
                    ax.scatter(trail_arr[-1, 0], trail_arr[-1, 1], trail_arr[-1, 2], marker='x', s=50) # End point
            
            # Plot Obstacles and Targets from MapConfig
            for zone in map_config.restricted_zones:
                ax.scatter(zone['pos'][0], zone['pos'][1], zone['pos'][2], color='red', marker='o', s=200, alpha=0.3, label='NFZ/Obstacle')
            for task_id, task in ga.allocate().items():
                target_pos = [0,0,0]
                for t in map_config.threats:
                    if t['id'] == task_id: target_pos = t['pos']
                for v in map_config.victims:
                    if v['id'] == task_id: target_pos = v['pos']
                ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='gold', marker='*', s=300, label=f'Target: {task_id}')
                
            # Filter duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            plot_path = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", f"trajectory_plot_{timestamp}.png")
            plt.savefig(plot_path)
            print(f"  Trajectory graph saved to: {os.path.abspath(plot_path)}")
            plt.close()
    except Exception as e:
        print(f"  Could not plot trajectory: {e}")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
