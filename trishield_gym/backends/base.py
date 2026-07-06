"""
Abstract base class for simulation backends and shared data structures.

All backends (lightweight, AirSim, etc.) implement the SimBackend interface,
allowing the DroneSwarmEnv to swap physics engines without changing algorithm code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
from typing import Optional


class MissionStatus(IntEnum):
    """Drone mission status codes, matching trishield_core.agent conventions."""
    IDLE = 0
    ACTIVE = 1
    RTB = 2      # Return to base
    DEAD = 3


@dataclass
class DroneState:
    """Snapshot of a single drone's state at a point in time.

    This is the universal data exchange format between backends and the
    environment. Backends populate these; the environment reads them.
    """
    drone_id: str
    drone_type: str                                # 'uav' or 'ugv'
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    battery: float = 100.0
    mission_status: MissionStatus = MissionStatus.IDLE
    assigned_task: Optional[np.ndarray] = None     # Target position or None
    camera_image: Optional[np.ndarray] = None      # RGB image (H, W, 3) uint8

    def status_onehot(self) -> np.ndarray:
        """Return mission status as a one-hot vector of length 4."""
        onehot = np.zeros(4, dtype=np.float32)
        onehot[int(self.mission_status)] = 1.0
        return onehot


@dataclass
class CollisionEvent:
    """Record of a collision detected during a simulation step."""
    drone_id: str
    collided_with: str          # Other drone ID or obstacle name
    collision_type: str         # 'drone-drone' or 'drone-obstacle'
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    penetration_depth: float = 0.0


class SimBackend(ABC):
    """Abstract interface that all simulation backends must implement.

    The DroneSwarmEnv delegates all physics, state queries, and rendering
    to a concrete SimBackend. This enables swapping between a fast NumPy
    simulator and AirSim without changing the algorithm.

    Lifecycle:
        1. __init__() — construct with config
        2. reset(map_config) — initialize/reset the world
        3. apply_actions(actions) — advance physics by one step
        4. get_observations() — read drone states
        5. check_collisions() — detect collisions this step
        6. get_render_data() — get visualization data
        7. close() — clean up resources
    """

    @abstractmethod
    def reset(self, map_config) -> dict[str, DroneState]:
        """Reset the simulation world and return initial drone states.

        Args:
            map_config: A MapConfig defining the scenario (spawns, obstacles, etc.)

        Returns:
            Dictionary mapping drone_id → DroneState for all drones.
        """
        ...

    @abstractmethod
    def apply_actions(self, actions: dict[str, np.ndarray],
                      action_mode: str = "velocity") -> dict[str, DroneState]:
        """Apply actions to all drones and advance simulation by one timestep.

        Args:
            actions: Dict mapping drone_id → action array.
                - velocity mode: np.ndarray shape (3,) → [vx, vy, vz] in [-1, 1]
                - waypoint mode: np.ndarray shape (3,) → [x, y, z] absolute target
            action_mode: Either 'velocity' or 'waypoint'.

        Returns:
            Updated dictionary of drone_id → DroneState.
        """
        ...

    @abstractmethod
    def get_observations(self) -> dict[str, DroneState]:
        """Return current drone states without advancing simulation.

        Returns:
            Dictionary mapping drone_id → DroneState.
        """
        ...

    @abstractmethod
    def check_collisions(self) -> list[CollisionEvent]:
        """Check for collisions that occurred during the last step.

        Returns:
            List of CollisionEvent objects (empty if no collisions).
        """
        ...

    @abstractmethod
    def get_render_data(self) -> dict:
        """Return data needed to render the current frame.

        Returns:
            Dict with keys: 'drone_states', 'obstacles', 'threats',
            'victims', 'bounds', 'trails'.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by the backend."""
        ...
