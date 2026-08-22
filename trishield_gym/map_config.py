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
    charging_stations: list[dict] = field(default_factory=list) # [{'id': ..., 'pos': ...}]
    wind_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    weather_params: dict[str, float] = field(default_factory=dict)  # e.g., {'Rain': 0.0, 'Snow': 0.0, 'Fog': 0.0}

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

        charging_stations = data.get("charging_stations", [])
        
        return cls(
            name=data.get("name", "unnamed"),
            bounds=tuple(data.get("bounds", [75.0, 75.0, 40.0])),
            drone_configs=data.get("drone_configs", []),
            obstacles=obstacles,
            threats=data.get("threats", []),
            victims=data.get("victims", []),
            restricted_zones=data.get("restricted_zones", []),
            charging_stations=charging_stations,
            wind_vector=data.get("wind_vector", [0.0, 0.0, 0.0]),
            weather_params=data.get("weather_params", {}),
            max_steps=data.get("max_steps", 500),
            dt=data.get("dt", 0.1),
            camera_resolution=tuple(data.get("camera_resolution", [84, 84])),
        )

    def randomize_targets(self, rng=None):
        """Randomize positions of victims while constraining them strictly to open street corridors or rooftop elevations."""
        import numpy as np
        import math
        if rng is None:
            rng = np.random.default_rng()

        # Constrain drone circular spawn helipad at exact world center (X = 0.0, Y = 0.0)
        cx = 0.0
        cy = 0.0
        base_z = 10.0
        radius = 4.0

        num_drones = len(self.drone_configs)
        for i, dc in enumerate(self.drone_configs):
            angle = (2.0 * math.pi * i) / max(1, num_drones)
            dx = round(cx + radius * math.cos(angle), 2)
            dy = round(cy + radius * math.sin(angle), 2)
            dz = round(base_z + (i * 0.5), 2)
            dc["spawn"] = [dx, dy, dz]

        # Constrain survivor search targets across open street avenues & accessible rooftop elevations
        street_points = [
            [25.0, 25.0, 33.0],   # Office Tower 1 Rooftop (33m elevation)
            [-25.0, 25.0, 33.0],  # Office Tower 2 Rooftop (33m elevation)
            [25.0, -25.0, 33.0],  # Office Tower 3 Rooftop (33m elevation)
            [-25.0, -25.0, 33.0], # Office Tower 4 Rooftop (33m elevation)
            [0.0, 35.0, 3.0],     # Open Main Avenue North
            [35.0, 0.0, 3.0],     # Open Main Avenue East
            [-35.0, 0.0, 3.0],    # Open Main Avenue West
            [0.0, -35.0, 3.0],    # Open Main Avenue South
        ]

        for i, victim in enumerate(self.victims):
            sp = street_points[i % len(street_points)]
            # Add small random jitter within open street bounds (+/- 2.0m)
            rx = round(float(sp[0] + rng.uniform(-2.0, 2.0) if sp[2] < 10 else sp[0]), 2)
            ry = round(float(sp[1] + rng.uniform(-2.0, 2.0) if sp[2] < 10 else sp[1]), 2)
            rz = sp[2]
            victim['pos'] = [rx, ry, rz]

        self.threats = []

    @classmethod
    def default_trishield(cls) -> "MapConfig":
        """Factory for the standard TriShield scenario.

        Matches the original simulation.py setup: 3 UAVs, 2 UGVs,
        1 rooftop survivor rescue, 1 survivor victim.
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
            threats=[],
            victims=[
                {"id": "LocateSurvivor", "pos": [-15.0, 18.0, 2.0], "urgency": 10},
                {"id": "DeliverFirstAid", "pos": [-10.0, 18.0, 2.0], "urgency": 8},
                {"id": "MapEnvironment", "pos": [0.0, 15.0, 18.0], "urgency": 5},
                {"id": "EstablishCommsRelay", "pos": [15.0, 0.0, 10.0], "urgency": 7},
                {"id": "InspectDamage", "pos": [5.0, -15.0, 8.0], "urgency": 6},
            ],
            wind_vector=[0.5, 0.2, 0.0],
            weather_params={"Rain": 0.0, "Snow": 0.0, "Fog": 0.0},
            max_steps=500,
            dt=0.1,
        )

    def randomize(self, seed: Optional[int] = None):
        """Randomize targets, wind, and weather for the scenario.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        import random
        if seed is not None:
            random.seed(seed)
            
        bx, by, bz = self.bounds
        
        def get_valid_pos():
            while True:
                pos = [
                    random.uniform(-bx * 0.75, bx * 0.75),
                    random.uniform(-by * 0.75, by * 0.75),
                    random.uniform(0.0, bz * 0.8)
                ]
                # Check against obstacles (with a 1.0m padding to ensure it's reachable)
                valid = True
                for obs in self.obstacles:
                    dist = ((pos[0] - obs.position[0])**2 + (pos[1] - obs.position[1])**2)**0.5
                    if dist <= obs.radius + 1.0:
                        valid = False
                        break
                if valid:
                    return pos

        # Jitter threats
        for threat in self.threats:
            threat["pos"] = get_valid_pos()
            
        # Jitter victims
        for victim in self.victims:
            victim["pos"] = get_valid_pos()
            
        # Randomize wind vector (X, Y components, minimal Z)
        self.wind_vector = [
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(-0.2, 0.2)
        ]
        
        # Pick a single dominant weather type or "Clear"
        weather_choice = random.choice(["Rain", "Snow", "Fog", "Clear"])
        
        self.weather_params = {
            "Rain": 0.0,
            "Snow": 0.0,
            "Fog": 0.0,
        }
        
        if weather_choice == "Rain":
            self.weather_params["Rain"] = random.uniform(0.2, 0.8)
        elif weather_choice == "Snow":
            self.weather_params["Snow"] = random.uniform(0.1, 0.5)
        elif weather_choice == "Fog":
            self.weather_params["Fog"] = random.uniform(0.1, 0.4)

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
