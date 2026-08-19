import os
import numpy as np

class PSOOptimizer:
    """Particle Swarm Optimization 3D Flight Controller with Swarm Repulsion."""

    def __init__(self, w=0.5, c1=0.5, c2=0.5, safe_distance=2.5):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.safe_distance = safe_distance

    def compute_velocity(self, agent, all_agents, restricted_zones=None, wind_vector=None):
        if getattr(agent, 'mission_status', 'ACTIVE') == 'DEAD':
            return np.zeros(3)

        target = agent.assigned_task
        if target is None:
            target = agent.pos.copy()
        else:
            target = np.array(target, dtype=float)

        r1, r2 = np.random.rand(), np.random.rand()
        pbest_force = self.c1 * r1 * (target - agent.pos)
        gbest_force = self.c2 * r2 * (target - agent.pos)

        cognitive_vel = self.w * agent.vel + pbest_force + gbest_force

        # Swarm Repulsion
        repulsion = np.zeros(3)
        variant = os.environ.get("TRISAR_VARIANT", "full")
        if variant not in ["no_repulsion", "floor"]:
            for other in all_agents:
                if other.id == agent.id:
                    continue
                dist = np.linalg.norm(agent.pos - other.pos)
                if 0 < dist < self.safe_distance:
                    direction = (agent.pos - other.pos) / dist
                    repulsion += direction * (self.safe_distance - dist) * 2.0

        final_vel = cognitive_vel + repulsion

        # Speed Limiter
        speed = np.linalg.norm(final_vel)
        max_speed = getattr(agent, 'max_speed', 15.0)
        if speed > max_speed:
            final_vel = (final_vel / speed) * max_speed

        return final_vel
