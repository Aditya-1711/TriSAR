"""
Map configuration and scenario definitions.

Maps define the physical world: bounds, obstacles, drone spawn positions,
threats, victims, and simulation parameters. They can be loaded from YAML
files or constructed programmatically.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import yaml


@dataclass
class Obstacle:
    """A physical obstacle in the simulation world.

    Attributes:
        position: [x, y, z] center of the obstacle.
        radius: Collision radius in meters.
        obstacle_type: Visual/behavior hint — 'sphere', 'no_fly_zone', 'building'.
        name: Optional human-readable name.
    """
    position: list[float]
    radius: float
    obstacle_type: str = "sphere"
    name: str = "obstacle"


@dataclass
class MapConfig:
    """Complete scenario configuration for a simulation episode.

    This defines everything the environment needs to set up a world:
    where drones spawn, what obstacles exist, where threats/victims are,
    and simulation parameters like timestep and max duration.

    Can be loaded from YAML files or built with factory methods.
    """
    name: str = "default"
    bounds: tuple[float, float, float] = (50.0, 50.0, 30.0)  # Half-extents

    # Drone spawn configurations
    # Each entry: {'id': 'UAV_1', 'type': 'uav', 'spawn': [x, y, z]}
    drone_configs: list[dict] = field(default_factory=list)

    # Environment objects
    obstacles: list[Obstacle] = field(default_factory=list)
    threats: list[dict] = field(default_factory=list)    # {'id': ..., 'pos': ..., 'type': ...}
    victims: list[dict] = field(default_factory=list)    # {'id': ..., 'pos': ..., 'urgency': ...}
    restricted_zones: list[dict] = field(default_factory=list)  # {'pos': ..., 'radius': ...}
    wind_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Simulation parameters
    max_steps: int = 500
    dt: float = 0.1
    camera_resolution: tuple[int, int] = (84, 84)  # (width, height) for camera obs

    @classmethod
    def from_yaml(cls, path: str) -> "MapConfig":
        """Load a map configuration from a YAML file.

        Args:
            path: Path to the YAML file. Can be absolute or relative to
                  the maps/ directory inside trishield_gym.

        Returns:
            A populated MapConfig instance.
        """
        # Try absolute path first, then relative to built-in maps directory
        if not os.path.isabs(path) or not os.path.exists(path):
            maps_dir = os.path.join(os.path.dirname(__file__), "maps")
            candidate = os.path.join(maps_dir, path)
            if os.path.exists(candidate):
                path = candidate

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        obstacles = [
            Obstacle(
                position=o["position"],
                radius=o["radius"],
                obstacle_type=o.get("obstacle_type", "sphere"),
                name=o.get("name", "obstacle"),
            )
            for o in data.get("obstacles", [])
        ]

        return cls(
            name=data.get("name", "unnamed"),
            bounds=tuple(data.get("bounds", [50.0, 50.0, 30.0])),
            drone_configs=data.get("drone_configs", []),
            obstacles=obstacles,
            threats=data.get("threats", []),
            victims=data.get("victims", []),
            restricted_zones=data.get("restricted_zones", []),
            wind_vector=data.get("wind_vector", [0.0, 0.0, 0.0]),
            max_steps=data.get("max_steps", 500),
            dt=data.get("dt", 0.1),
            camera_resolution=tuple(data.get("camera_resolution", [84, 84])),
        )

    @classmethod
    def default_trishield(cls) -> "MapConfig":
        """Factory for the standard TriShield scenario.

        Matches the original simulation.py setup: 3 UAVs, 2 UGVs,
        1 rogue drone threat, 1 survivor victim.
        """
        return cls(
            name="trishield_default",
            bounds=(25.0, 25.0, 20.0),
            drone_configs=[
                {"id": "UAV_1", "type": "uav", "spawn": [0.0, 0.0, 10.0]},
                {"id": "UAV_2", "type": "uav", "spawn": [5.0, -5.0, 12.0]},
                {"id": "UAV_3", "type": "uav", "spawn": [-5.0, 5.0, 8.0]},
                {"id": "UAV_4", "type": "uav", "spawn": [10.0, 10.0, 10.0]},
                {"id": "UAV_5", "type": "uav", "spawn": [-10.0, -10.0, 10.0]},
            ],
            obstacles=[
                Obstacle([0.0, 10.0, 5.0], radius=5.0, obstacle_type="no_fly_zone",
                         name="Central NFZ"),
                Obstacle([-10.0, -5.0, 0.0], radius=8.0, obstacle_type="no_fly_zone",
                         name="Southern NFZ"),
            ],
            threats=[
                {"id": "DefencePerimeter", "pos": [18.0, -18.0, 8.0], "type": "chokepoint"},
            ],
            victims=[
                {"id": "LocateSurvivor", "pos": [-15.0, 18.0, 0.0], "urgency": 10},
                {"id": "DeliverFirstAid", "pos": [-14.0, 17.0, 1.0], "urgency": 8},
                {"id": "MapEnvironment", "pos": [0.0, 15.0, 18.0], "urgency": 5},
            ],
            wind_vector=[0.5, 0.2, 0.0],
            max_steps=500,
            dt=0.1,
        )

    @property
    def drone_ids(self) -> list[str]:
        """Return ordered list of drone IDs."""
        return [d["id"] for d in self.drone_configs]

    @property
    def num_drones(self) -> int:
        """Return total number of drones."""
        return len(self.drone_configs)

    @property
    def num_threats(self) -> int:
        """Return total number of threats."""
        return len(self.threats)

    @property
    def num_victims(self) -> int:
        """Return total number of victims."""
        return len(self.victims)
