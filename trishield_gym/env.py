"""
DroneSwarmEnv — Core Gymnasium environment for multi-drone coordination.

This is the single entry point for all algorithm code. It wraps a pluggable
SimBackend (lightweight or AirSim) behind the standard Gymnasium API:
reset(), step(), render(), close().

Supports:
    - Multiple drones (UAVs + UGVs) with heterogeneous capabilities
    - Two action modes: continuous velocity commands or discrete waypoints
    - Raw camera image observations + state vectors
    - Pluggable reward functions and metrics logging
    - Configurable maps via YAML or programmatic MapConfig

Usage:
    env = DroneSwarmEnv(backend='lightweight', map_config=MapConfig.default_trishield())
    obs, info = env.reset()
    actions = {drone_id: np.array([0.5, 0.3, 0.0]) for drone_id in env.drone_ids}
    obs, reward, terminated, truncated, info = env.step(actions)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

from trishield_gym.backends import create_backend, SimBackend, DroneState
from trishield_gym.map_config import MapConfig
from trishield_gym.rewards import RewardFunction, MissionReward
from trishield_gym.metrics import MetricsLogger
from trishield_gym.rendering import SimRenderer


class DroneSwarmEnv(gym.Env):
    """Multi-drone Gymnasium environment with pluggable simulation backends.

    The environment manages N drones in a 3D world. At each step, the
    algorithm provides an action for each drone, and the environment
    returns observations, rewards, and termination signals through the
    standard Gymnasium API.

    Args:
        backend: Name of the simulation backend ('lightweight' or 'airsim').
        map_config: Scenario configuration (drones, obstacles, threats, etc.)
        reward_fn: Optional custom reward function. Defaults to MissionReward.
        action_mode: 'velocity' (continuous [-1,1]^3) or 'waypoint' (absolute [x,y,z]).
        render_mode: 'human' (live window), 'rgb_array' (numpy), or None.
        backend_kwargs: Additional kwargs passed to the backend constructor.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        backend: str = "lightweight",
        map_config: Optional[MapConfig] = None,
        reward_fn: Optional[RewardFunction] = None,
        action_mode: str = "velocity",
        render_mode: Optional[str] = None,
        **backend_kwargs,
    ):
        super().__init__()

        # Configuration
        self.map_config = map_config or MapConfig.default_trishield()
        self.action_mode = action_mode
        self.render_mode = render_mode

        # Create backend
        self._backend: SimBackend = create_backend(backend, **backend_kwargs)

        # Reward function (default: mission-oriented)
        self.reward_fn = reward_fn or MissionReward()

        # Metrics logger
        self.logger = MetricsLogger()

        # Renderer (lazy-initialized)
        self._renderer: Optional[SimRenderer] = None

        # Drone info
        self.drone_ids = self.map_config.drone_ids
        self.num_drones = self.map_config.num_drones

        # Step tracking
        self._current_step = 0
        self._current_states: dict[str, DroneState] = {}

        # ---- Define spaces ----
        self._define_spaces()

    def _define_spaces(self):
        """Define action and observation spaces for all drones.

        Observation per drone (Dict):
            'state_vector': Box — [x, y, z, vx, vy, vz, battery, status_onehot(4),
                                    assigned_task(3), threats_rel(N*3), victims_rel(N*3)]
            'camera': Box — (H, W, 3) uint8 raw camera image

        Action per drone:
            velocity mode: Box(-1, 1, shape=(3,)) — normalized [vx, vy, vz]
            waypoint mode: Box(low_bound, high_bound, shape=(3,)) — [x, y, z]
        """
        bounds = self.map_config.bounds
        n_threats = max(1, self.map_config.num_threats)
        n_victims = max(1, self.map_config.num_victims)
        cam_w, cam_h = self.map_config.camera_resolution

        # State vector length: 3(pos) + 3(vel) + 1(battery) + 4(status) + 3(task) + N*3(threats) + N*3(victims)
        state_dim = 3 + 3 + 1 + 4 + 3 + n_threats * 3 + n_victims * 3

        # Per-drone observation space
        single_obs_space = spaces.Dict({
            "state_vector": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim,), dtype=np.float32,
            ),
            "camera": spaces.Box(
                low=0, high=255,
                shape=(cam_h, cam_w, 3), dtype=np.uint8,
            ),
        })

        # Per-drone action space
        if self.action_mode == "velocity":
            single_action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
            )
        elif self.action_mode == "waypoint":
            single_action_space = spaces.Box(
                low=np.array([-bounds[0], -bounds[1], 0.0]),
                high=np.array([bounds[0], bounds[1], bounds[2]]),
                shape=(3,), dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown action_mode '{self.action_mode}'")

        # Multi-drone spaces (Dict keyed by drone_id)
        self.observation_space = spaces.Dict({
            did: single_obs_space for did in self.drone_ids
        })
        self.action_space = spaces.Dict({
            did: single_action_space for did in self.drone_ids
        })

    # ------------------------------------------------------------------ #
    #  Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Optional dict with reset configuration.

        Returns:
            (observations, info) tuple.
        """
        super().reset(seed=seed)
        self._current_step = 0

        # Reset backend
        self._current_states = self._backend.reset(self.map_config)

        # Reset metrics
        self.logger.start_episode(self.drone_ids)

        # Reset exploration reward if applicable
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()

        obs = self._build_observations(self._current_states)
        info = self._build_info()

        return obs, info

    def step(self, actions: dict[str, np.ndarray]):
        """Execute one simulation step with the given actions.

        Args:
            actions: Dict mapping drone_id → action array.
                velocity mode: np.array([vx, vy, vz]) in [-1, 1]
                waypoint mode: np.array([x, y, z]) absolute position

        Returns:
            (observations, reward, terminated, truncated, info)
        """
        self._current_step += 1

        # Apply actions through the backend
        self._current_states = self._backend.apply_actions(
            actions, action_mode=self.action_mode
        )

        # Check collisions
        collisions = self._backend.check_collisions()

        # Compute reward
        info = self._build_info()
        reward = self.reward_fn.compute(
            self._current_states, actions, collisions, info
        )

        # Log metrics
        self.logger.log_step(self._current_states, collisions, reward)

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._current_step >= self.map_config.max_steps

        # Build observations
        obs = self._build_observations(self._current_states)

        # Enrich info
        info["step"] = self._current_step
        info["collisions"] = [
            {"drone": c.drone_id, "with": c.collided_with, "type": c.collision_type}
            for c in collisions
        ]
        info["reward"] = reward

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current state of the simulation.

        Returns:
            numpy array (H, W, 3) if render_mode='rgb_array', else None.
        """
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = SimRenderer(render_mode=self.render_mode)

        render_data = self._backend.get_render_data()
        return self._renderer.render_frame(render_data)

    def close(self):
        """Clean up backend and renderer resources."""
        self._backend.close()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    #  Public Helpers
    # ------------------------------------------------------------------ #

    def get_metrics_summary(self) -> dict:
        """Return a summary of episode metrics.

        Returns:
            Dict with episode-level and per-drone metrics.
        """
        return self.logger.get_summary()

    def export_metrics_csv(self, path: str):
        """Export episode metrics to CSV.

        Args:
            path: Output CSV file path.
        """
        self.logger.export_csv(path)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _build_observations(self, states: dict[str, DroneState]) -> dict:
        """Convert DroneState objects into Gymnasium observation format.

        For each drone, builds:
            - 'state_vector': flat numpy array of floats
            - 'camera': RGB image array (H, W, 3) uint8
        """
        obs = {}
        n_threats = max(1, self.map_config.num_threats)
        n_victims = max(1, self.map_config.num_victims)

        for drone_id, state in states.items():
            # Build state vector
            parts = [
                state.position.astype(np.float32),                      # 3
                state.velocity.astype(np.float32),                      # 3
                np.array([state.battery / 100.0], dtype=np.float32),    # 1
                state.status_onehot(),                                  # 4
            ]

            # Assigned task (3 values, zeros if none)
            if state.assigned_task is not None:
                parts.append(state.assigned_task.astype(np.float32))
            else:
                parts.append(np.zeros(3, dtype=np.float32))

            # Relative threat positions (padded/truncated to n_threats)
            for i in range(n_threats):
                if i < len(self.map_config.threats):
                    threat_pos = np.array(self.map_config.threats[i]["pos"], dtype=np.float32)
                    rel = threat_pos - state.position.astype(np.float32)
                    parts.append(rel)
                else:
                    parts.append(np.zeros(3, dtype=np.float32))

            # Relative victim positions (padded/truncated to n_victims)
            for i in range(n_victims):
                if i < len(self.map_config.victims):
                    victim_pos = np.array(self.map_config.victims[i]["pos"], dtype=np.float32)
                    rel = victim_pos - state.position.astype(np.float32)
                    parts.append(rel)
                else:
                    parts.append(np.zeros(3, dtype=np.float32))

            state_vector = np.concatenate(parts)

            # Camera image
            camera = state.camera_image if state.camera_image is not None else \
                np.zeros((*self.map_config.camera_resolution[::-1], 3), dtype=np.uint8)

            obs[drone_id] = {
                "state_vector": state_vector,
                "camera": camera,
            }

        return obs

    def _build_info(self) -> dict:
        """Build the info dict with map and mission context."""
        return {
            "threats": self.map_config.threats,
            "victims": self.map_config.victims,
            "obstacles": [
                {"position": o.position, "radius": o.radius, "name": o.name}
                for o in self.map_config.obstacles
            ],
            "map_name": self.map_config.name,
            "max_steps": self.map_config.max_steps,
        }

    def _check_terminated(self) -> bool:
        """Check if the episode should terminate (all missions complete or all drones dead)."""
        # All drones dead → terminate
        all_dead = all(
            s.mission_status.name == "DEAD"
            for s in self._current_states.values()
        )
        if all_dead:
            return True

        # All targets reached → terminate
        completion_dist = 3.0
        all_threats_handled = True
        all_victims_rescued = True

        for threat in self.map_config.threats:
            threat_pos = np.array(threat["pos"])
            reached = any(
                np.linalg.norm(s.position - threat_pos) < completion_dist
                for s in self._current_states.values()
                if s.drone_type == "uav"
            )
            if not reached:
                all_threats_handled = False

        for victim in self.map_config.victims:
            victim_pos = np.array(victim["pos"])
            reached = any(
                np.linalg.norm(s.position - victim_pos) < completion_dist
                for s in self._current_states.values()
            )
            if not reached:
                all_victims_rescued = False

        if all_threats_handled and all_victims_rescued:
            self.logger.mission_completed = True
            return True

        return False
