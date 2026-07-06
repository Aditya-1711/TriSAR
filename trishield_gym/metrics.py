"""
Metrics logging for drone simulation episodes.

Tracks per-drone and per-episode metrics including path length, collisions,
energy consumption, mission completion, and timing. Supports CSV export
for post-analysis.
"""

import csv
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from trishield_gym.backends.base import DroneState, CollisionEvent


@dataclass
class DroneMetrics:
    """Accumulated metrics for a single drone across an episode."""
    drone_id: str
    total_path_length: float = 0.0
    total_collisions: int = 0
    collision_events: list[str] = field(default_factory=list)
    initial_battery: float = 100.0
    final_battery: float = 100.0
    energy_consumed: float = 0.0
    average_speed: float = 0.0
    max_speed: float = 0.0
    tasks_completed: int = 0
    _speed_samples: list[float] = field(default_factory=list)
    _last_position: Optional[np.ndarray] = None

    def update(self, state: DroneState, collisions: list[CollisionEvent]):
        """Update metrics with new state and collision data."""
        # Path length (cumulative Euclidean distance)
        if self._last_position is not None:
            step_dist = np.linalg.norm(state.position - self._last_position)
            self.total_path_length += step_dist
        self._last_position = state.position.copy()

        # Speed tracking
        speed = np.linalg.norm(state.velocity)
        self._speed_samples.append(speed)
        self.max_speed = max(self.max_speed, speed)

        # Battery / energy
        self.final_battery = state.battery
        self.energy_consumed = self.initial_battery - state.battery

        # Collisions
        drone_collisions = [c for c in collisions if c.drone_id == self.drone_id]
        self.total_collisions += len(drone_collisions)
        for c in drone_collisions:
            self.collision_events.append(
                f"{c.collision_type}: {c.collided_with}"
            )

    def finalize(self):
        """Compute final aggregate metrics."""
        if self._speed_samples:
            self.average_speed = float(np.mean(self._speed_samples))


class MetricsLogger:
    """Episode-level metrics logger for the drone simulation.

    Tracks per-drone metrics and provides episode summaries and CSV export.

    Usage:
        logger = MetricsLogger()
        logger.start_episode(drone_ids)
        # ... simulation loop ...
        logger.log_step(states, collisions)
        # ... end of episode ...
        summary = logger.get_summary()
        logger.export_csv("results.csv")
    """

    def __init__(self):
        self.drone_metrics: dict[str, DroneMetrics] = {}
        self.episode_start_time: float = 0.0
        self.total_steps: int = 0
        self.total_reward: float = 0.0
        self.mission_completed: bool = False
        self._episode_history: list[dict] = []

    def start_episode(self, drone_ids: list[str]):
        """Initialize tracking for a new episode.

        Args:
            drone_ids: List of drone IDs to track.
        """
        self.drone_metrics = {
            did: DroneMetrics(drone_id=did) for did in drone_ids
        }
        self.episode_start_time = time.time()
        self.total_steps = 0
        self.total_reward = 0.0
        self.mission_completed = False

    def log_step(
        self,
        states: dict[str, DroneState],
        collisions: list[CollisionEvent],
        reward: float = 0.0,
    ):
        """Log metrics for a single simulation step.

        Args:
            states: Current drone states.
            collisions: Collisions detected this step.
            reward: Reward received this step.
        """
        self.total_steps += 1
        self.total_reward += reward

        for drone_id, state in states.items():
            if drone_id in self.drone_metrics:
                self.drone_metrics[drone_id].update(state, collisions)

    def get_summary(self) -> dict:
        """Return a summary of the episode metrics.

        Returns:
            Dictionary with episode-level and per-drone metrics.
        """
        elapsed = time.time() - self.episode_start_time

        # Finalize per-drone metrics
        for dm in self.drone_metrics.values():
            dm.finalize()

        drone_summaries = {}
        for did, dm in self.drone_metrics.items():
            drone_summaries[did] = {
                "path_length": round(dm.total_path_length, 2),
                "collisions": dm.total_collisions,
                "energy_consumed": round(dm.energy_consumed, 2),
                "final_battery": round(dm.final_battery, 2),
                "average_speed": round(dm.average_speed, 2),
                "max_speed": round(dm.max_speed, 2),
                "tasks_completed": dm.tasks_completed,
            }

        total_collisions = sum(dm.total_collisions for dm in self.drone_metrics.values())
        total_path = sum(dm.total_path_length for dm in self.drone_metrics.values())
        total_energy = sum(dm.energy_consumed for dm in self.drone_metrics.values())

        return {
            "episode": {
                "total_steps": self.total_steps,
                "total_reward": round(self.total_reward, 2),
                "total_path_length": round(total_path, 2),
                "total_collisions": total_collisions,
                "total_energy_consumed": round(total_energy, 2),
                "mission_completed": self.mission_completed,
                "wall_time_seconds": round(elapsed, 2),
            },
            "drones": drone_summaries,
        }

    def export_csv(self, path: str):
        """Export per-drone metrics to a CSV file.

        Args:
            path: Output file path.
        """
        summary = self.get_summary()

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "drone_id", "path_length", "collisions", "energy_consumed",
                "final_battery", "average_speed", "max_speed", "tasks_completed",
            ])
            for did, data in summary["drones"].items():
                writer.writerow([
                    did,
                    data["path_length"],
                    data["collisions"],
                    data["energy_consumed"],
                    data["final_battery"],
                    data["average_speed"],
                    data["max_speed"],
                    data["tasks_completed"],
                ])

        # Append episode summary as a separate section
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["Episode Summary"])
            for key, value in summary["episode"].items():
                writer.writerow([key, value])

    def reset(self):
        """Clear all metrics for reuse."""
        self.drone_metrics.clear()
        self.total_steps = 0
        self.total_reward = 0.0
        self.mission_completed = False
