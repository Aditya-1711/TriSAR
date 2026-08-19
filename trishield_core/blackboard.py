class Blackboard:
    def __init__(self):
        self.threats = {}       # id -> {'pos': [x,y,z], 'type': 'rooftop_rescue'}
        self.victims = {}       # id -> {'pos': [x,y,z], 'type': 'human'}
        self.agent_states = {}  # agent_id -> Agent object reference
        
    def broadcast_state(self, agent):
        self.agent_states[agent.id] = agent
        
    def register_threat(self, threat_id, pose, threat_type):
        self.threats[threat_id] = {'pos': pose, 'type': threat_type}
        
    def register_victim(self, victim_id, pose, urgency):
        self.victims[victim_id] = {'pos': pose, 'urgency': urgency}

    def clear(self):
        self.threats.clear()
        self.victims.clear()
