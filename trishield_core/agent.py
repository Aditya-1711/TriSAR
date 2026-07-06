import numpy as np

class Agent:
    def __init__(self, agent_id, initial_pos, max_speed, drain_rate=0.2):
        self.id = agent_id
        self.pos = np.array(initial_pos, dtype=float)
        self.base_location = np.array([0.0, 0.0, 0.0], dtype=float) # Centralized Mothership Return-to-base location
        self.vel = np.zeros(3, dtype=float)
        self.max_speed = max_speed
        
        self.assigned_task = None
        self.mission_status = 'IDLE' # 'IDLE', 'ACTIVE', 'RTB', 'DEAD'
        
        # Energy Awareness
        self.battery = 100.0
        self.energy_drain_rate = drain_rate 
        
        self.max_acceleration = 5.0 # Hardware actuator limits

    def update_position(self, dt):
        if self.battery <= 0:
            self.mission_status = 'DEAD'
            self.vel = np.zeros(3, dtype=float)
            self.apply_kinematic_constraints()
            self.pos += self.vel * dt
            return

        # Base Station Recharging Logic
        dist_to_base = np.linalg.norm(self.pos - self.base_location)
        if self.mission_status == 'RTB' and dist_to_base < 3.0:
            self.vel = np.zeros(3, dtype=float)
            self.battery = min(100.0, self.battery + 15.0 * dt) # Recharge fast
            if self.battery >= 100.0:
                self.mission_status = 'IDLE' # Deploy again!
            return

        # Battery Simulation based on effort (velocity magnitude)
        speed = np.linalg.norm(self.vel)
        effort_drain = speed * 0.02 * dt
        base_drain = self.energy_drain_rate * dt
        self.battery -= (base_drain + effort_drain)

        # Trigger RTB
        if self.battery < 20.0 and self.mission_status != 'RTB':
            self.mission_status = 'RTB'
            self.assigned_task = self.base_location.copy()

        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        
        # Accelerations constraints (rough approximation bounds)
        self.apply_kinematic_constraints()
        self.pos += self.vel * dt

    def apply_kinematic_constraints(self):
        pass

class UAVAgent(Agent):
    def __init__(self, agent_id, initial_pos):
        super().__init__(agent_id, initial_pos, max_speed=15.0, drain_rate=0.3) # Adjusted drain for flying
        self.payloads = ["RGB", "Thermal", "NetGun"]

    def apply_kinematic_constraints(self):
        if self.mission_status == 'DEAD':
            # Plummet
            self.vel[2] = -9.8
            if self.pos[2] < 0:
                self.pos[2] = 0.0
                self.vel[2] = 0.0
            return
            
        if self.pos[2] < 1.0:
            self.pos[2] = 1.0
            if self.vel[2] < 0:
                self.vel[2] = 0.0

class UGVAgent(Agent):
    def __init__(self, agent_id, initial_pos):
        super().__init__(agent_id, initial_pos, max_speed=5.0, drain_rate=0.05) # Low drain for rolling
        self.payloads = ["LiDAR", "HeavyMedkit"]

    def apply_kinematic_constraints(self):
        # UGVs strictly bound to the 2D plane
        self.pos[2] = 0.0
        self.vel[2] = 0.0
