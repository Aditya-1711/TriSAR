import numpy as np

def detect_collisions(drone_poses: dict, collision_radius: float = 2.0) -> list:
    """Detect 3D inter-drone collisions (distance < collision_radius)."""
    collisions = []
    drone_ids = list(drone_poses.keys())
    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            d1_id = drone_ids[i]
            d2_id = drone_ids[j]
            p1 = np.array(drone_poses[d1_id], dtype=float)
            p2 = np.array(drone_poses[d2_id], dtype=float)
            dist = np.linalg.norm(p1 - p2)
            if dist < collision_radius:
                collisions.append((d1_id, d2_id, dist))
    return collisions
