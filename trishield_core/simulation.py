import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os

# Append current directory to path so trishield_core can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trishield_core.agent import UAVAgent
from trishield_core.blackboard import Blackboard
from trishield_core.ga_allocator import HeterogeneousGA
from trishield_core.pso_optimizer import PSOOptimizer

def main():
    bb = Blackboard()
    
    # Initialize drone swarm
    agents = [
        UAVAgent("UAV_1", [0, 0, 10]),
        UAVAgent("UAV_2", [5, -5, 12]),
        UAVAgent("UAV_3", [-5, 5, 8]),
        UAVAgent("UAV_4", [10, 10, 15]),
        UAVAgent("UAV_5", [-10, -10, 6])
    ]
    
    for a in agents:
        bb.broadcast_state(a)
        
    print("Registered Disaster Scenario Tasks:")
    sar_tasks = [
        ("LocateSurvivor", [-15, 15, 0], 10),
        ("DeliverFirstAid", [18, -12, 0], 9),
        ("RooftopSurvivor", [20, 20, 15], 8),
        ("CommsRelay", [-20, -15, 12], 7),
        ("InspectDamage", [10, -18, 5], 6),
    ]
    for task_id, pos, urgency in sar_tasks:
        bb.register_victim(task_id, pos, urgency=urgency)
        print(f"- Task '{task_id}' at {pos} (Urgency: {urgency})")
    
    ga = HeterogeneousGA(bb)
    pso = PSOOptimizer(safe_distance=3.0)
    
    # 1. Run GA Task Allocation
    assignments = ga.allocate()
    print("\n--- GA Task Allocation ---")
    for task_id, agent_id in assignments.items():
        print(f"Task '{task_id}' assigned to -> {agent_id}")
        # Apply the assignment to the actual agent object so PSO has a real target
        for a in agents:
            if a.id == agent_id:
                # Look up the task position from the blackboard
                if task_id in bb.threats:
                    a.assigned_task = bb.threats[task_id]['pos']
                elif task_id in bb.victims:
                    a.assigned_task = bb.victims[task_id]['pos']
                a.mission_status = 'ACTIVE'
                break
    print("--------------------------\n")

    # Setup 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([0, 20])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("TriShield Autonomous Aerial Drone Swarm")
    
    # Initialize scatter objects
    uav_pos = np.array([a.pos for a in agents if isinstance(a, UAVAgent)])
    scatter_uav = ax.scatter(uav_pos[:,0], uav_pos[:,1], uav_pos[:,2], c='blue', marker='^', s=100, label='UAV (Drone)')
    
    for vid, v in bb.victims.items():
        v_pos = v['pos']
        ax.scatter(v_pos[0], v_pos[1], v_pos[2], c='orange', marker='*', s=150, label=f'Task: {vid}')
    
    ax.legend(loc="upper left")
    
    dt = 0.1

    def update(frame):
        for a in agents:
            new_vel = pso.compute_velocity(a, agents)
            a.vel = new_vel
            a.update_position(dt)
            bb.broadcast_state(a)

        # Clear and redraw
        ax.cla()
        ax.set_xlim([-25, 25])
        ax.set_ylim([-25, 25])
        ax.set_zlim([0, 20])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"TriShield Drone Swarm — Frame {frame + 1}/100")

        uav_pos = np.array([a.pos for a in agents if isinstance(a, UAVAgent)])
        ax.scatter(uav_pos[:,0], uav_pos[:,1], uav_pos[:,2], c='blue', marker='^', s=100, label='UAV (Drone)')

        # Redraw victim task markers
        for vid, v in bb.victims.items():
            v_pos = v['pos']
            ax.scatter(v_pos[0], v_pos[1], v_pos[2], c='orange', marker='*', s=150, label=f'Task: {vid}')
        ax.legend(loc='upper left')

    # Generate animation
    anim = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "artifacts"), exist_ok=True)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "trishield_sim.gif"))
    print(f"Rendering Simulation out to {out_path}...")
    try:
        anim.save(out_path, writer='pillow', fps=20)
        print("Rendering successful!")
    except Exception as e:
        print("Error saving animation (ensure Pillow is installed). Error:", e)

if __name__ == "__main__":
    main()
