import numpy as np

class PSOOptimizer:
    def __init__(self, w=0.5, c1=0.8, c2=0.9, safe_distance=3.0):
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
        
        # Environmental Wind
        if wind_vector is not None:
            cognitive_vel += np.array(wind_vector)

        # Swarm Repulsion (Micro Congestion Preventer)
        repulsion = np.zeros(3)
        for other in all_agents:
            if other.id == agent.id: continue
            dist = np.linalg.norm(agent.pos - other.pos)
            if 0 < dist < self.safe_distance:
                # Exponential repulsion for strict safety limits
                direction = (agent.pos - other.pos) / dist
                repulsion += direction * ((self.safe_distance / dist)**3)
                
        # Restricted Zones / Dynamic Obstacles Repulsion
        if restricted_zones:
            for zone in restricted_zones:
                zone_pos = np.array(zone['pos'])
                zone_radius = zone.get('radius', 5.0)
                dist = np.linalg.norm(agent.pos - zone_pos)
                if 0 < dist < zone_radius + self.safe_distance:
                    direction = (agent.pos - zone_pos) / dist
                    # Create a perpendicular tangent vector to push the drone *around* the obstacle
                    # rather than just straight backward (breaking the local minima deadlock)
                    tangent = np.array([-direction[1], direction[0], 0.0])
                    
                    # Combine radial pushback and tangential sweeping force
                    escape_force = direction + (tangent * 1.5)
                    repulsion += escape_force * (((zone_radius + self.safe_distance) / dist)**4)
                
        new_vel = cognitive_vel + repulsion
        return new_vel
