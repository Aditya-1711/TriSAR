import numpy as np

def compute_swarm_repulsion(agent_pos, agent_id, all_agent_poses, safe_distance=2.5, gain=2.0):
    """Compute exponential potential field inter-agent repulsion vector."""
    repulsion = np.zeros(3)
    for other_id, other_pos in all_agent_poses.items():
        if other_id == agent_id:
            continue
        other_vec = np.array(other_pos, dtype=float)
        dist = np.linalg.norm(agent_pos - other_vec)
        if 0 < dist < safe_distance:
            direction = (agent_pos - other_vec) / dist
            repulsion += direction * (safe_distance - dist) * gain
    return repulsion
