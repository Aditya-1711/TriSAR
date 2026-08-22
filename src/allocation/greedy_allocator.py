import numpy as np

class GreedyAllocator:
    """Multi-round nearest-neighbor greedy allocator for multi-UAV swarm task assignment."""

    def __init__(self, blackboard):
        self.bb = blackboard

    def allocate(self) -> dict:
        agent_queues = {agent_id: [] for agent_id in self.bb.agent_states.keys()}
        unassigned_victims = list(self.bb.victims.keys())
        assigned_agents_in_round = set()

        for victim_id in unassigned_victims:
            if len(assigned_agents_in_round) >= len(self.bb.agent_states):
                assigned_agents_in_round.clear()

            v = self.bb.victims[victim_id]
            v_pos = np.array(v['pos'], dtype=float)
            best_agent = None
            best_dist = float('inf')

            for agent_id, agent in self.bb.agent_states.items():
                if agent_id in assigned_agents_in_round:
                    continue
                a_pos = np.array(agent.pos, dtype=float)
                dist = float(np.linalg.norm(a_pos - v_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_agent = agent

            if best_agent is not None:
                agent_queues[best_agent.id].append(victim_id)
                assigned_agents_in_round.add(best_agent.id)

        return agent_queues

