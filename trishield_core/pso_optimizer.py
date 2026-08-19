import numpy as np

class PSOOptimizer:
    def __init__(self, w=0.5, c1=0.5, c2=0.5, safe_distance=2.5):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.safe_distance = safe_distance
        
    def compute_velocity(self, agent, all_agents, restricted_zones=None, wind_vector=None):
        if agent.mission_status == 'DEAD':
            return np.zeros(3)
            
        target = agent.assigned_task
        if target is None:
            target = agent.pos.copy() # Hover/Stay
        else:
            target = np.array(target)

        r1, r2 = np.random.rand(), np.random.rand()
        pbest_force = self.c1 * r1 * (target - agent.pos)
        gbest_force = self.c2 * r2 * (target - agent.pos)
        
        cognitive_vel = self.w * agent.vel + pbest_force + gbest_force
        
        # Environmental Wind (Handled automatically by AirSim's PID if using AirSim)
        # If using lightweight, we would need integral compensation, but for AirSim we ignore it
        pass

        # Swarm Repulsion (Micro Congestion Preventer)
        repulsion = np.zeros(3)
        import os
        variant = os.environ.get("TRISAR_VARIANT", "full")
        if variant not in ["no_repulsion", "floor"]:
            for other in all_agents:
                if other.id == agent.id: continue
                dist = np.linalg.norm(agent.pos - other.pos)
                if 0 < dist < self.safe_distance:
                    # Exponential repulsion for strict safety limits
                    direction = (agent.pos - other.pos) / dist
                    # Ignore Z-axis repulsion to prevent pushing drones into the ground
                    direction[2] = 0.0 
                    # Cap the repulsion force to avoid violent bouncing/gridlock
                    force_magnitude = min(10.0, ((self.safe_distance / dist)**2) * 2.0)
                    repulsion += direction * force_magnitude
                
        # Restricted Zones / Building Obstacles Repulsion
        if restricted_zones:
            for zone in restricted_zones:
                zone_pos = np.array(zone['pos'])
                zone_radius = zone.get('radius', 12.0)
                dist = np.linalg.norm(agent.pos[:2] - zone_pos[:2])
                if 0 < dist < zone_radius + self.safe_distance and agent.pos[2] < zone_pos[2] + 5.0:
                    direction = np.array([agent.pos[0] - zone_pos[0], agent.pos[1] - zone_pos[1], 0.0])
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 0:
                        direction = direction / norm_dir
                    tangent = np.array([-direction[1], direction[0], 0.0])
                    escape_force = direction * 2.0 + tangent * 1.5
                    # Include upward vertical lift to fly over the building top!
                    escape_force[2] = 3.0
                    force_magnitude = min(15.0, (((zone_radius + self.safe_distance) / max(0.1, dist))**2) * 3.0)
                    repulsion += escape_force * force_magnitude

        total_vel = cognitive_vel + repulsion
        
        # High Altitude Cruise & Smooth Target Descent Logic:
        # Measure 2D horizontal distance to target (ignoring Z height difference)
        dist_2d_to_target = np.linalg.norm(target[:2] - agent.pos[:2])

        if not getattr(agent, 'is_relay_landed', False):
            if dist_2d_to_target > 10.0:
                # Cruise high at 35m altitude over all building tops when far away horizontally
                cruising_z = 35.0
                total_vel[2] = np.clip((cruising_z - agent.pos[2]), -15.0, 15.0)
            else:
                # Direct target Z convergence when within 10m of target sector
                total_vel[2] = np.clip((target[2] - agent.pos[2]), -15.0, 15.0)

        return total_vel
