"""
Lightweight NumPy-based simulation backend.

This backend reuses the existing trishield_core classes (Agent, Blackboard,
PSOOptimizer) for physics simulation. It runs entirely in-process with no
external dependencies beyond NumPy, making it ideal for fast prototyping,
unit testing, and algorithm development.

Camera images are synthetically rendered as simple top-down depth maps so
the observation space is consistent with the AirSim backend.
"""

import numpy as np
import sys
import os

# Ensure trishield_core is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trishield_core.agent import UAVAgent, UGVAgent
from trishield_core.blackboard import Blackboard
from trishield_gym.backends.base import SimBackend, DroneState, CollisionEvent, MissionStatus
from trishield_gym.map_config import MapConfig


class LightweightBackend(SimBackend):
    """Fast NumPy physics backend built on top of trishield_core.

    Simulates drone movement with velocity/acceleration constraints,
    battery drain, collision detection, and synthetic camera rendering.
    """

    def __init__(self):
        self.agents: dict[str, UAVAgent | UGVAgent] = {}
        self.blackboard: Blackboard = Blackboard()
        self.map_config: MapConfig | None = None
        self.trails: dict[str, list[np.ndarray]] = {}
        self._collision_buffer: list[CollisionEvent] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, map_config: MapConfig) -> dict[str, DroneState]:
        """Initialize world from map config, spawn drones, register threats/victims."""
        self.map_config = map_config
        self.agents.clear()
        self.trails.clear()
        self._collision_buffer.clear()
        self._step_count = 0
        self.blackboard = Blackboard()

        # Spawn drones using trishield_core agent classes
        for dc in map_config.drone_configs:
            spawn = dc["spawn"]
            if dc["type"] == "uav":
                agent = UAVAgent(dc["id"], spawn)
            else:
                agent = UGVAgent(dc["id"], spawn)
            self.agents[dc["id"]] = agent
            self.blackboard.broadcast_state(agent)
            self.trails[dc["id"]] = [np.array(spawn, dtype=float)]

        # Register threats and victims on the blackboard
        for threat in map_config.threats:
            self.blackboard.register_threat(
                threat["id"], threat["pos"], threat.get("type", "unknown")
            )
        for victim in map_config.victims:
            self.blackboard.register_victim(
                victim["id"], victim["pos"], victim.get("urgency", 5)
            )

        return self.get_observations()

    def close(self) -> None:
        """Release resources (minimal for lightweight backend)."""
        self.agents.clear()
        self.trails.clear()

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #

    def apply_actions(self, actions: dict[str, np.ndarray],
                      action_mode: str = "velocity") -> dict[str, DroneState]:
        """Apply velocity or waypoint actions and advance physics by dt.

        Args:
            actions: Dict of drone_id → action vector.
            action_mode: 'velocity' → [vx, vy, vz] normalized [-1, 1]
                         'waypoint' → [x, y, z] absolute target position
        """
        dt = self.map_config.dt if self.map_config else 0.1
        self._collision_buffer.clear()

        for drone_id, action in actions.items():
            agent = self.agents.get(drone_id)
            if agent is None or agent.mission_status == "DEAD":
                continue

            action = np.array(action, dtype=float)

            if action_mode == "velocity":
                # Scale normalized [-1, 1] action to agent's max speed
                agent.vel = action * agent.max_speed
            elif action_mode == "waypoint":
                # Compute velocity toward waypoint
                direction = action - agent.pos
                dist = np.linalg.norm(direction)
                if dist > 0.5:
                    agent.vel = (direction / dist) * min(agent.max_speed, dist / dt)
                else:
                    agent.vel = np.zeros(3)
            else:
                raise ValueError(f"Unknown action_mode '{action_mode}'")

            # Apply wind effect
            if self.map_config and hasattr(self.map_config, "wind_vector"):
                agent.vel += np.array(self.map_config.wind_vector)

            agent.update_position(dt)
            self.blackboard.broadcast_state(agent)
            self.trails[drone_id].append(agent.pos.copy())

        # Collision detection after all moves
        self._detect_collisions()
        self._step_count += 1

        return self.get_observations()

    # ------------------------------------------------------------------ #
    #  Observations
    # ------------------------------------------------------------------ #

    def get_observations(self) -> dict[str, DroneState]:
        """Return current state of all drones as DroneState objects."""
        states = {}
        for drone_id, agent in self.agents.items():
            # Map trishield_core string status to MissionStatus enum
            status_map = {
                "IDLE": MissionStatus.IDLE,
                "ACTIVE": MissionStatus.ACTIVE,
                "RTB": MissionStatus.RTB,
                "DEAD": MissionStatus.DEAD,
            }

            camera_img = self._render_synthetic_camera(agent)

            states[drone_id] = DroneState(
                drone_id=drone_id,
                drone_type="uav" if isinstance(agent, UAVAgent) else "ugv",
                position=agent.pos.copy(),
                velocity=agent.vel.copy(),
                battery=float(agent.battery),
                mission_status=status_map.get(agent.mission_status, MissionStatus.IDLE),
                assigned_task=np.array(agent.assigned_task) if agent.assigned_task is not None else None,
                camera_image=camera_img,
            )
        return states

    def _render_synthetic_camera(self, agent) -> np.ndarray:
        """Generate a simple synthetic top-down camera image for the drone.

        Creates an 84×84 RGB image where:
        - Other drones appear as colored circles
        - Obstacles appear as red regions
        - Threats/victims appear as markers
        This provides consistent observation shape with AirSim's real camera.
        """
        res = self.map_config.camera_resolution if self.map_config else (84, 84)
        w, h = res
        img = np.zeros((h, w, 3), dtype=np.uint8)

        if self.map_config is None:
            return img

        bounds = self.map_config.bounds
        cam_range = max(bounds[0], bounds[1]) * 2  # View range around agent

        def world_to_pixel(world_pos):
            """Convert world XY relative to agent into pixel coordinates."""
            rel = world_pos[:2] - agent.pos[:2]
            px = int((rel[0] / cam_range + 0.5) * w)
            py = int((-rel[1] / cam_range + 0.5) * h)
            return px, py

        def draw_circle(img, cx, cy, radius, color):
            """Draw a filled circle on the image."""
            for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                        img[y, x] = color

        # Draw obstacles (red)
        for obs in self.map_config.obstacles:
            px, py = world_to_pixel(np.array(obs.position))
            pixel_radius = max(2, int(obs.radius / cam_range * w))
            draw_circle(img, px, py, pixel_radius, [180, 40, 40])

        # Draw threats (bright red X markers)
        for threat in self.map_config.threats:
            px, py = world_to_pixel(np.array(threat["pos"]))
            draw_circle(img, px, py, 3, [255, 0, 0])

        # Draw victims (orange)
        for victim in self.map_config.victims:
            px, py = world_to_pixel(np.array(victim["pos"]))
            draw_circle(img, px, py, 3, [255, 165, 0])

        # Draw other drones (blue for UAV, green for UGV)
        for did, other in self.agents.items():
            if did == agent.id:
                continue
            px, py = world_to_pixel(other.pos)
            color = [50, 100, 255] if isinstance(other, UAVAgent) else [50, 200, 50]
            draw_circle(img, px, py, 2, color)

        # Draw self (white center dot)
        cx, cy = w // 2, h // 2
        draw_circle(img, cx, cy, 2, [255, 255, 255])

        return img

    # ------------------------------------------------------------------ #
    #  Collisions
    # ------------------------------------------------------------------ #

    def check_collisions(self) -> list[CollisionEvent]:
        """Return collisions detected during the last step."""
        return list(self._collision_buffer)

    def _detect_collisions(self):
        """Check drone-drone and drone-obstacle collisions."""
        agents_list = list(self.agents.values())

        # Drone-drone collisions
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                a, b = agents_list[i], agents_list[j]
                dist = np.linalg.norm(a.pos - b.pos)
                collision_threshold = 2.0  # meters
                if dist < collision_threshold:
                    self._collision_buffer.append(CollisionEvent(
                        drone_id=a.id,
                        collided_with=b.id,
                        collision_type="drone-drone",
                        position=((a.pos + b.pos) / 2).copy(),
                        penetration_depth=collision_threshold - dist,
                    ))

        # Drone-obstacle collisions
        if self.map_config:
            for agent in agents_list:
                for obs in self.map_config.obstacles:
                    obs_pos = np.array(obs.position)
                    dist = np.linalg.norm(agent.pos - obs_pos)
                    if dist < obs.radius:
                        self._collision_buffer.append(CollisionEvent(
                            drone_id=agent.id,
                            collided_with=obs.name,
                            collision_type="drone-obstacle",
                            position=agent.pos.copy(),
                            penetration_depth=obs.radius - dist,
                        ))

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def get_render_data(self) -> dict:
        """Return all data needed for the Matplotlib renderer."""
        states = self.get_observations()
        return {
            "drone_states": states,
            "obstacles": self.map_config.obstacles if self.map_config else [],
            "threats": self.map_config.threats if self.map_config else [],
            "victims": self.map_config.victims if self.map_config else [],
            "bounds": self.map_config.bounds if self.map_config else (25, 25, 20),
            "trails": {did: list(trail) for did, trail in self.trails.items()},
            "step": self._step_count,
        }
