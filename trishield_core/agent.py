import numpy as np

# Energy Model Constants (Chapter 4.5)
P_HOVER = 0.28    # %/s, baseline hover power
C_DRAG = 0.00007  # %/s per (m/s)^3, parasitic aerodynamic drag coefficient
C_MASS = 0.05     # %/s per (m/s^2), acceleration/inertial penalty

class Agent:
    def __init__(self, agent_id, initial_pos, max_speed, drain_rate=0.2):
        self.id = agent_id
        self.pos = np.array(initial_pos, dtype=float)
        self.base_location = np.array([0.0, 0.0, 0.0], dtype=float) # Centralized Mothership Return-to-base location
        self.vel = np.zeros(3, dtype=float)
        self.prev_vel = np.zeros(3, dtype=float)
        self.max_speed = max_speed
        
        self.assigned_task = None
        self.mission_status = 'IDLE' # 'IDLE', 'ACTIVE', 'RTB', 'DEAD'
        
        # Energy Awareness
        self.battery = 100.0
        self.energy_drain_rate = drain_rate 
        
        self.max_acceleration = 5.0 # Hardware actuator limits

        self.charging_stations = [
            np.array([20.0, -15.0, 1.25], dtype=float),
            np.array([-25.0, -20.0, 1.25], dtype=float)
        ]
        self.is_relay_landed = False
        self.task_type = 'rescue'

    def update_position(self, dt):
        if self.battery <= 0:
            self.mission_status = 'DEAD'
            self.vel = np.zeros(3, dtype=float)
            self.prev_vel = self.vel.copy()
            self.apply_kinematic_constraints()
            self.pos += self.vel * dt
            return

        # Mesh Relay Power-Saving Rooftop Landing Logic
        if self.is_relay_landed:
            self.vel = np.zeros(3, dtype=float)
            self.battery -= (self.energy_drain_rate * 0.05 * dt)  # 95% power saving while relaying!
            self.prev_vel = self.vel.copy()
            return

        # Charging Station Recharging Logic
        nearest_station = min(self.charging_stations, key=lambda pad: np.linalg.norm(self.pos - pad))
        dist_to_station = np.linalg.norm(self.pos - nearest_station)
        if self.mission_status in ['RTC', 'RTB'] and dist_to_station < 2.5:
            self.vel = np.zeros(3, dtype=float)
            self.battery = min(100.0, self.battery + 20.0 * dt)  # Fast recharge station
            if self.battery >= 98.0:
                self.mission_status = 'IDLE'  # Fully charged, re-deploy to mission!
            self.prev_vel = self.vel.copy()
            return

        # Cubic-drag flight energy model (Chapter 4.5)
        speed = np.linalg.norm(self.vel)
        acceleration = np.linalg.norm(self.vel - self.prev_vel) / dt if dt > 0 else 0.0

        drain = (P_HOVER + C_DRAG * (speed ** 3) + C_MASS * acceleration) * dt
        self.battery -= drain

        # Trigger Return-To-Charge (RTC) at 25% battery
        if self.battery < 25.0 and self.mission_status not in ['RTC', 'RTB']:
            self.mission_status = 'RTC'
            self.assigned_task = nearest_station.copy()

        self.prev_vel = self.vel.copy()

        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        
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


