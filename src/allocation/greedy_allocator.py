import numpy as np

class GreedyAllocator:
    """Single-pass nearest-neighbor greedy allocator for multi-UAV swarm task assignment."""

    def __init__(self, blackboard):
        self.bb = blackboard

    def allocate(self) -> dict:
        assignments = {}
        unassigned_victims = list(self.bb.victims.keys())
        assigned_agents = set()

        for victim_id in unassigned_victims:
            v = self.bb.victims[victim_id]
            v_pos = np.array(v['pos'], dtype=float)
            best_agent = None
            best_dist = float('inf')

            for agent_id, agent in self.bb.agent_states.items():
                if agent_id in assigned_agents:
                    continue
                a_pos = np.array(agent.pos, dtype=float)
                dist = float(np.linalg.norm(a_pos - v_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_agent = agent

            if best_agent is not None:
                assignments[victim_id] = best_agent.id
                assigned_agents.add(best_agent.id)

        return assignments
