import numpy as np

class HeterogeneousGA:
    def __init__(self, blackboard):
        self.bb = blackboard
        
    def calculate_fitness(self, agent, task_pos, task_type, urgency=1.0):
        """
        Hardware-aware fitness function. Lower score is better.
        Returns float('inf') if physically impossible or agent is dead.
        """
        if agent.mission_status in ['DEAD', 'RTB']:
            return float('inf')
            
        distance = np.linalg.norm(np.array(agent.pos) - np.array(task_pos))
        eta = distance / agent.max_speed
        
        # Physical / Hardware constraints
        agent_type = agent.__class__.__name__
        if task_type == 'aerial_intercept':
            if agent_type == 'UGVAgent':
                return float('inf')
            fitness = eta * 1.0 
        elif task_type == 'ground_rescue':
            if agent_type == 'UAVAgent':
                return eta * 2.0 
            fitness = eta * 0.5 
        else:
            fitness = eta
            
        battery_penalty = 1.0 + (100.0 - agent.battery) / 50.0
        
        # Urgency lowers the fitness score (making it more attractive)
        urgency_factor = max(0.1, 11 - urgency) / 10.0 # 10 is high urgency -> 0.1 factor
        return fitness * battery_penalty * urgency_factor

    def allocate(self):
        """
        Dynamic Mission Priorities allocation mapping capabilities to highest urgency.
        """
        assignments = {}
        agents_available = list(self.bb.agent_states.values())
        
        all_tasks = []
        for tid, t in self.bb.threats.items():
            all_tasks.append((tid, t['pos'], 'aerial_intercept' if t['pos'][2] > 2 else 'ground_threat', t.get('urgency', 5)))
        for vid, v in self.bb.victims.items():
            all_tasks.append((vid, v['pos'], 'ground_rescue', v.get('urgency', 8)))
            
        # Dependancy / Urgency Sorting
        all_tasks.sort(key=lambda x: x[3], reverse=True)
            
        for task_id, task_pos, task_type, urgency in all_tasks:
            best_agent = None
            best_fitness = float('inf')
            
            for agent in agents_available:
                if agent.id in assignments.values(): 
                    continue # Single assignment enforcement
                
                fitness = self.calculate_fitness(agent, task_pos, task_type, urgency)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_agent = agent
            
            if best_agent is not None:
                assignments[task_id] = best_agent.id
                
        return assignments
