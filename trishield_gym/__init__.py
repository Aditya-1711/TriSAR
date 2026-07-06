"""
TriShield Gymnasium Drone Simulation Framework
===============================================

A modular, backend-agnostic Gymnasium environment for multi-drone coordination.
Supports pluggable simulation backends (lightweight NumPy physics or Microsoft AirSim),
configurable maps, reward functions, and comprehensive metrics logging.

Quick Start:
    from trishield_gym import DroneSwarmEnv, MapConfig

    env = DroneSwarmEnv(backend='lightweight', map_config=MapConfig.default_trishield())
    obs, info = env.reset()
    actions = {drone_id: env.action_space[drone_id].sample() for drone_id in env.drone_ids}
    obs, reward, terminated, truncated, info = env.step(actions)
"""

from trishield_gym.env import DroneSwarmEnv
from trishield_gym.map_config import MapConfig, Obstacle

__all__ = ["DroneSwarmEnv", "MapConfig", "Obstacle"]
__version__ = "1.0.0"
