"""
Matplotlib-based 3D renderer for the drone simulation.

Provides both interactive (human) and headless (rgb_array) rendering modes.
Renders drones, obstacles, threats, victims, flight trails, and battery labels
in a 3D scatter plot.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from trishield_gym.backends.base import DroneState, MissionStatus


class SimRenderer:
    """3D Matplotlib renderer for the drone swarm simulation.

    Supports two render modes:
        - 'human': Opens an interactive Matplotlib window.
        - 'rgb_array': Returns frames as numpy arrays (for video recording).
    """

    def __init__(self, render_mode: str = "rgb_array"):
        """
        Args:
            render_mode: Either 'human' or 'rgb_array'.
        """
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self._initialized = False

        if render_mode == "rgb_array":
            matplotlib.use("Agg")

    def _init_figure(self, bounds: tuple):
        """Create the Matplotlib figure and axes on first render."""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._initialized = True

    def render_frame(self, render_data: dict) -> np.ndarray | None:
        """Render a single frame from simulation data.

        Args:
            render_data: Dict from backend.get_render_data() containing:
                - drone_states, obstacles, threats, victims, bounds, trails, step

        Returns:
            RGB numpy array (H, W, 3) if render_mode='rgb_array', else None.
        """
        bounds = render_data.get("bounds", (25, 25, 20))

        if not self._initialized:
            self._init_figure(bounds)

        self.ax.cla()
        self.ax.set_xlim([-bounds[0], bounds[0]])
        self.ax.set_ylim([-bounds[1], bounds[1]])
        self.ax.set_zlim([0, bounds[2]])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        step = render_data.get("step", 0)
        self.ax.set_title(f"TriShield Swarm — Step {step}")

        # -- Draw obstacles --
        for obs in render_data.get("obstacles", []):
            pos = obs.position if hasattr(obs, "position") else obs.get("position", [0, 0, 0])
            radius = obs.radius if hasattr(obs, "radius") else obs.get("radius", 5.0)
            self._draw_sphere_wireframe(pos, radius, color="red", alpha=0.15)

        # -- Draw threats --
        for threat in render_data.get("threats", []):
            pos = threat.get("pos", threat.get("position", [0, 0, 0]))
            self.ax.scatter(
                pos[0], pos[1], pos[2],
                c="red", marker="x", s=120, linewidths=3,
                label=f"Threat: {threat.get('id', '?')}",
            )

        # -- Draw victims --
        for victim in render_data.get("victims", []):
            pos = victim.get("pos", victim.get("position", [0, 0, 0]))
            self.ax.scatter(
                pos[0], pos[1], pos[2],
                c="orange", marker="*", s=180,
                label=f"Victim: {victim.get('id', '?')}",
            )

        # -- Draw trails --
        trails = render_data.get("trails", {})
        for drone_id, trail in trails.items():
            if len(trail) < 2:
                continue
            trail_arr = np.array(trail[-50:])  # Last 50 positions for performance
            is_uav = "UAV" in drone_id.upper()
            color = "#4488ff" if is_uav else "#44cc44"
            self.ax.plot(
                trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2],
                color=color, alpha=0.3, linewidth=1,
            )

        # -- Draw drones --
        drone_states: dict[str, DroneState] = render_data.get("drone_states", {})
        uav_positions, ugv_positions = [], []
        for drone_id, state in drone_states.items():
            pos = state.position
            is_uav = state.drone_type == "uav"

            if is_uav:
                uav_positions.append(pos)
            else:
                ugv_positions.append(pos)

            # Battery label
            battery_text = f"{state.battery:.0f}%"
            status_str = state.mission_status.name
            color = "white"
            if state.mission_status == MissionStatus.DEAD:
                color = "gray"
            elif state.mission_status == MissionStatus.RTB:
                color = "orange"

            self.ax.text(
                pos[0], pos[1], pos[2] + 1.5,
                f"{drone_id}\n{battery_text} [{status_str}]",
                fontsize=6, ha="center", color=color,
            )

        # Plot drone markers
        if uav_positions:
            uav_arr = np.array(uav_positions)
            self.ax.scatter(
                uav_arr[:, 0], uav_arr[:, 1], uav_arr[:, 2],
                c="dodgerblue", marker="^", s=100, label="UAV",
                edgecolors="white", linewidths=0.5,
            )
        if ugv_positions:
            ugv_arr = np.array(ugv_positions)
            self.ax.scatter(
                ugv_arr[:, 0], ugv_arr[:, 1], ugv_arr[:, 2],
                c="limegreen", marker="s", s=100, label="UGV",
                edgecolors="white", linewidths=0.5,
            )

        # Legend (deduplicated)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=7)

        if self.render_mode == "human":
            plt.pause(0.01)
            return None
        else:
            # Render to numpy array
            self.fig.canvas.draw()
            width, height = self.fig.canvas.get_width_height()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(height, width, 3)
            return img

    def _draw_sphere_wireframe(self, center, radius, color="red", alpha=0.2):
        """Draw a wireframe sphere at the given position."""
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 8)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)

    def close(self):
        """Close the Matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self._initialized = False
