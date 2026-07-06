"""
Pluggable reward functions for the drone simulation environment.

Reward functions are decoupled from the environment so they can be swapped,
combined, or tuned without modifying the environment code. All reward
functions inherit from RewardFunction and implement compute().
"""

from abc import ABC, abstractmethod
import numpy as np

from trishield_gym.backends.base import DroneState, CollisionEvent


class RewardFunction(ABC):
    """Abstract base class for reward computation.

    Subclasses implement compute() to return a scalar team reward based
    on the current simulation state and events.
    """

    @abstractmethod
    def compute(
        self,
        states: dict[str, DroneState],
        actions: dict[str, np.ndarray],
        collisions: list[CollisionEvent],
        info: dict,
    ) -> float:
        """Compute the team reward for the current step.

        Args:
            states: Current drone states keyed by drone_id.
            actions: Actions taken this step keyed by drone_id.
            collisions: Collisions that occurred this step.
            info: Additional context (threats, victims, step count, etc.)

        Returns:
            Scalar reward value.
        """
        ...


class MissionReward(RewardFunction):
    """Reward function for threat-neutralization and victim-rescue missions.

    Rewards drones for:
        - Moving closer to assigned targets (proximity bonus)
        - Reaching targets within threshold (completion bonus)
        - Maintaining battery (efficiency bonus)

    Penalizes:
        - Collisions (safety penalty)
        - Excessive energy use (waste penalty)
        - Time taken (urgency pressure)
    """

    def __init__(
        self,
        proximity_weight: float = 1.0,
        completion_bonus: float = 50.0,
        collision_penalty: float = -20.0,
        energy_penalty_weight: float = 0.1,
        time_penalty: float = -0.1,
        completion_distance: float = 3.0,
    ):
        self.proximity_weight = proximity_weight
        self.completion_bonus = completion_bonus
        self.collision_penalty = collision_penalty
        self.energy_penalty_weight = energy_penalty_weight
        self.time_penalty = time_penalty
        self.completion_distance = completion_distance

    def compute(self, states, actions, collisions, info) -> float:
        reward = 0.0

        threats = info.get("threats", [])
        victims = info.get("victims", [])
        all_targets = [(t["id"], np.array(t["pos"])) for t in threats] + \
                      [(v["id"], np.array(v["pos"])) for v in victims]

        # --- Proximity reward ---
        for drone_id, state in states.items():
            if state.assigned_task is not None:
                dist = np.linalg.norm(state.position - state.assigned_task)
                # Inverse distance reward (closer = better)
                reward += self.proximity_weight / (1.0 + dist)

                # Completion bonus
                if dist < self.completion_distance:
                    reward += self.completion_bonus

        # --- Collision penalty ---
        reward += len(collisions) * self.collision_penalty

        # --- Energy penalty ---
        for drone_id, state in states.items():
            energy_used = 100.0 - state.battery
            reward -= energy_used * self.energy_penalty_weight

        # --- Time pressure ---
        reward += self.time_penalty

        return reward


class ExplorationReward(RewardFunction):
    """Reward function for area coverage and search-and-rescue missions.

    Tracks which grid cells have been visited and rewards new coverage.
    Penalizes revisiting already-explored areas.
    """

    def __init__(
        self,
        grid_resolution: float = 5.0,
        new_cell_bonus: float = 2.0,
        revisit_penalty: float = -0.5,
        collision_penalty: float = -15.0,
    ):
        self.grid_resolution = grid_resolution
        self.new_cell_bonus = new_cell_bonus
        self.revisit_penalty = revisit_penalty
        self.collision_penalty = collision_penalty
        self.visited_cells: set[tuple[int, int, int]] = set()

    def _pos_to_cell(self, position: np.ndarray) -> tuple[int, int, int]:
        """Convert a world position to a grid cell index."""
        return tuple((position / self.grid_resolution).astype(int))

    def compute(self, states, actions, collisions, info) -> float:
        reward = 0.0

        for drone_id, state in states.items():
            cell = self._pos_to_cell(state.position)
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                reward += self.new_cell_bonus
            else:
                reward += self.revisit_penalty

        reward += len(collisions) * self.collision_penalty
        return reward

    def reset(self):
        """Clear visited cells for a new episode."""
        self.visited_cells.clear()


class CompositeReward(RewardFunction):
    """Weighted combination of multiple reward functions.

    Example:
        reward_fn = CompositeReward([
            (MissionReward(), 1.0),
            (ExplorationReward(), 0.5),
        ])
    """

    def __init__(self, components: list[tuple[RewardFunction, float]]):
        """
        Args:
            components: List of (RewardFunction, weight) tuples.
        """
        self.components = components

    def compute(self, states, actions, collisions, info) -> float:
        total = 0.0
        for reward_fn, weight in self.components:
            total += weight * reward_fn.compute(states, actions, collisions, info)
        return total
