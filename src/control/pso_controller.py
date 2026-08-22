import os
import numpy as np

class PSOOptimizer:
    """
    Population-Based Particle Swarm Optimization 3D Flight Controller.
    
    DESIGN CHOICE EXPLANATION:
    This is a per-drone local trajectory optimizer. Each physical drone maintains its own
    independent population of P candidate velocity vectors at each control step.
    The global-best (g_best) is tracked PER-DRONE across its own P candidates, rather than
    being shared across the fleet. A single fleet-wide g_best is physically meaningless
    when drones are dispatched to different spatial target coordinates.
    """

    def __init__(self, w=0.5, c1=0.5, c2=0.5, safe_distance=2.5, num_candidates=10, num_iterations=5):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.safe_distance = safe_distance
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations
        
        self.drone_states = {}
        self.log_step_counters = {}

    def _evaluate_candidate_fitness_components(self, agent, candidate_vel, dt, target, all_agents, restricted_zones, wind_vector=None):
        if wind_vector is None:
            raise ValueError("wind_vector must be explicitly provided to PSO candidate fitness evaluation")
        raw_wind = np.array(wind_vector, dtype=float)
        wind_term = raw_wind * 0.1
        projected_pos = agent.pos + (candidate_vel + wind_term) * dt
        dist_to_target = float(np.linalg.norm(projected_pos - target))


        repulsion_penalty = 0.0
        variant = os.environ.get("TRISAR_VARIANT", "full")
        if variant not in ["no_repulsion", "floor"]:
            for other in all_agents:
        repulsion_penalty = 0.0
        variant = os.environ.get("TRISAR_VARIANT", "full")
        if variant not in ["no_repulsion", "floor"]:
            for other in all_agents:
                if getattr(other, 'id', None) == agent.id:
                    continue
                dist_to_other = np.linalg.norm(projected_pos - other.pos)
                if dist_to_other < self.safe_distance:
                    repulsion_penalty += 1000.0

            if restricted_zones:
                for zone in restricted_zones:
                    zone_pos = np.array(zone['pos'], dtype=float)
                    zone_radius = zone.get('radius', 12.0)
                    dist_to_zone = np.linalg.norm(projected_pos[:2] - zone_pos[:2])
                    if dist_to_zone < zone_radius + self.safe_distance and projected_pos[2] < zone_pos[2] + 5.0:
                        repulsion_penalty += 1000.0

        return dist_to_target + repulsion_penalty, dist_to_target, repulsion_penalty

    def _evaluate_candidate_fitness(self, agent, candidate_vel, dt, target, all_agents, restricted_zones, wind_vector=None):
        fit, _, _ = self._evaluate_candidate_fitness_components(agent, candidate_vel, dt, target, all_agents, restricted_zones, wind_vector=wind_vector)
        return fit


    def compute_velocity(self, agent, all_agents, restricted_zones=None, wind_vector=None, dt=0.1):
        if getattr(agent, 'mission_status', 'ACTIVE') == 'DEAD':
            return np.zeros(3)

        target = agent.assigned_task
        if target is None or isinstance(target, str):
            target = agent.pos.copy()
        else:
            target = np.array(target, dtype=float)

        agent_id = agent.id
        max_speed = getattr(agent, 'max_speed', 15.0)

        # Fresh candidate population initialization per control step around target-heading vector
        base_dir = target - agent.pos
        base_dist = np.linalg.norm(base_dir)
        if base_dist > 0:
            desired_vel = (base_dir / base_dist) * min(max_speed, base_dist)
        else:
            desired_vel = agent.vel.copy()

        candidates = []
        for _ in range(self.num_candidates):
            noise = np.random.normal(0, 1.0, size=3)
            cand_v = desired_vel + noise
            speed = np.linalg.norm(cand_v)
            if speed > max_speed:
                cand_v = (cand_v / speed) * max_speed
            candidates.append(cand_v)
        candidates = np.array(candidates)

        state = self.drone_states.get(agent_id, {})
        state['current_target'] = target.copy()
        self.drone_states[agent_id] = state

        p_best_vel = candidates.copy()
        p_best_fit = np.array([
            self._evaluate_candidate_fitness(agent, v, dt, target, all_agents, restricted_zones, wind_vector=wind_vector)
            for v in candidates
        ])

        g_best_idx = np.argmin(p_best_fit)
        g_best_vel = p_best_vel[g_best_idx].copy()
        g_best_fit = p_best_fit[g_best_idx]

        for _ in range(self.num_iterations):
            for i in range(self.num_candidates):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = self.c1 * r1 * (p_best_vel[i] - candidates[i])
                social = self.c2 * r2 * (g_best_vel - candidates[i])
                candidates[i] = self.w * candidates[i] + cognitive + social

                speed = np.linalg.norm(candidates[i])
                if speed > max_speed:
                    candidates[i] = (candidates[i] / speed) * max_speed

                fit = self._evaluate_candidate_fitness(agent, candidates[i], dt, target, all_agents, restricted_zones, wind_vector=wind_vector)

                if fit < p_best_fit[i]:
                    p_best_fit[i] = fit
                    p_best_vel[i] = candidates[i].copy()

                if fit < g_best_fit:
                    g_best_fit = fit
                    g_best_vel = candidates[i].copy()

        best_cand_idx = np.argmin(p_best_fit)
        selected_vel = candidates[best_cand_idx].copy()

        repulsion = np.zeros(3)
        variant = os.environ.get("TRISAR_VARIANT", "full")
        if variant not in ["no_repulsion", "floor"]:
            for other in all_agents:
                if getattr(other, 'id', None) == agent.id:
                    continue
                dist = np.linalg.norm(agent.pos - other.pos)
                if 0 < dist < self.safe_distance:
                    direction = (agent.pos - other.pos) / dist
                    direction[2] = 0.0
                    force_magnitude = min(10.0, ((self.safe_distance / dist) ** 2) * 2.0)
                    repulsion += direction * force_magnitude

            if restricted_zones:
                for zone in restricted_zones:
                    zone_pos = np.array(zone['pos'], dtype=float)
                    zone_radius = zone.get('radius', 12.0)
                    dist = np.linalg.norm(agent.pos[:2] - zone_pos[:2])
                    if 0 < dist < zone_radius + self.safe_distance and agent.pos[2] < zone_pos[2] + 5.0:
                        direction = np.array([agent.pos[0] - zone_pos[0], agent.pos[1] - zone_pos[1], 0.0], dtype=float)
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 0:
                            direction = direction / norm_dir
                        tangent = np.array([-direction[1], direction[0], 0.0], dtype=float)
                        escape_force = direction * 2.0 + tangent * 1.5
                        escape_force[2] = 3.0
                        force_magnitude = min(15.0, (((zone_radius + self.safe_distance) / max(0.1, dist)) ** 2) * 3.0)
                        repulsion += escape_force * force_magnitude

        final_vel = selected_vel + repulsion

        speed = np.linalg.norm(final_vel)
        if speed > max_speed:
            final_vel = (final_vel / speed) * max_speed

        return final_vel



