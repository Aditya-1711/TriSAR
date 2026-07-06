"""
Microsoft AirSim simulation backend.

Connects to a running AirSim instance (Unreal Engine) to provide realistic
multi-rotor physics, camera rendering, and collision detection. This backend
implements the same SimBackend interface as the lightweight backend, allowing
seamless algorithm transfer.

Requirements:
    - AirSim simulator running (Unreal Engine with AirSim plugin)
    - `pip install airsim` (optional dependency)
    - Multi-vehicle settings configured in AirSim's settings.json
"""

import numpy as np
import time
import warnings

from trishield_gym.backends.base import SimBackend, DroneState, CollisionEvent, MissionStatus
from trishield_gym.map_config import MapConfig

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False


class AirSimBackend(SimBackend):
    """AirSim-connected backend for realistic drone simulation.

    Connects to a running AirSim instance and controls multiple drones
    through the Python API. Provides real camera images, physics-based
    flight dynamics, and hardware-accurate collision detection.

    Note:
        This backend requires AirSim to be running before the environment
        is created. Install with: pip install airsim
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 41451):
        """Initialize AirSim connection parameters.

        Args:
            ip: IP address of the AirSim instance.
            port: Port number for the AirSim API.
        """
        if not AIRSIM_AVAILABLE:
            raise ImportError(
                "AirSim is not installed. Install with: pip install airsim\n"
                "You also need a running AirSim instance (Unreal Engine).\n"
                "For development without AirSim, use backend='lightweight'."
            )

        self.ip = ip
        self.port = port
        self.client: airsim.MultirotorClient | None = None
        self.vehicle_names: list[str] = []
        self.map_config: MapConfig | None = None
        self.trails: dict[str, list[np.ndarray]] = {}
        self._collision_buffer: list[CollisionEvent] = []
        self._step_count: int = 0
        self._batteries: dict[str, float] = {}  # Simulated battery (AirSim lacks this)
        self._max_speeds: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, map_config: MapConfig) -> dict[str, DroneState]:
        """Connect to AirSim, reset the simulation, and arm all vehicles.

        The drone vehicle names in AirSim's settings.json must match the
        drone IDs in the MapConfig (e.g., 'UAV_1', 'UAV_2', etc.).
        """
        self.map_config = map_config
        self._collision_buffer.clear()
        self._step_count = 0

        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
        print("  [DEBUG] Confirming connection...")
        self.client.confirmConnection()
        print("  [DEBUG] Resetting AirSim...")
        self.client.reset()
        time.sleep(1.0)  # Allow reset to complete

        self.vehicle_names = [dc["id"] for dc in map_config.drone_configs]
        self.trails = {name: [] for name in self.vehicle_names}
        self._batteries = {}
        self._max_speeds = {}

        futures = []
        for dc in map_config.drone_configs:
            name = dc["id"]
            self._batteries[name] = 100.0
            self._max_speeds[name] = 15.0 if dc["type"] == "uav" else 5.0

            print(f"  [DEBUG] Enabling API for {name}...")
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)

            # Take off UAVs
            if dc["type"] == "uav":
                print(f"  [DEBUG] Taking off {name}...")
                futures.append(self.client.takeoffAsync(timeout_sec=10, vehicle_name=name))

        print("  [DEBUG] Waiting for takeoffs to complete...")
        for f in futures:
            try:
                f.join()
            except Exception as e:
                print(f"  [DEBUG] Warning: Takeoff join failed: {e}")
        
        print("  [DEBUG] Fetching initial observations...")
        return self.get_observations()

    def close(self) -> None:
        """Disarm all vehicles and disconnect from AirSim."""
        if self.client is not None:
            for name in self.vehicle_names:
                try:
                    self.client.armDisarm(False, vehicle_name=name)
                    self.client.enableApiControl(False, vehicle_name=name)
                except Exception:
                    pass
            self.client = None

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #

    def apply_actions(self, actions: dict[str, np.ndarray],
                      action_mode: str = "velocity") -> dict[str, DroneState]:
        """Send velocity or waypoint commands to AirSim vehicles.

        Args:
            actions: Dict of drone_id → action vector.
            action_mode: 'velocity' → [vx, vy, vz] normalized [-1, 1]
                         'waypoint' → [x, y, z] absolute target position
        """
        self._collision_buffer.clear()
        dt = self.map_config.dt if self.map_config else 0.1

        futures = []
        for drone_id, action in actions.items():
            if self._batteries.get(drone_id, 0) <= 0:
                continue

            action = np.array(action, dtype=float)
            max_speed = self._max_speeds.get(drone_id, 15.0)

            if action_mode == "velocity":
                # Scale from normalized [-1, 1] to m/s
                vx, vy, vz = action * max_speed
                future = self.client.moveByVelocityAsync(
                    float(vx), float(vy), float(-vz),  # Invert Z for AirSim NED frame
                    duration=dt,
                    vehicle_name=drone_id,
                )
                futures.append(future)

            elif action_mode == "waypoint":
                x, y, z = action
                future = self.client.moveToPositionAsync(
                    float(x), float(y), float(-z),  # AirSim uses NED (z is down)
                    velocity=max_speed,
                    vehicle_name=drone_id,
                )
                futures.append(future)

            # Simulate battery drain
            speed = np.linalg.norm(action) * max_speed if action_mode == "velocity" else max_speed
            drain_rate = 0.3 if "UAV" in drone_id else 0.05
            effort_drain = speed * 0.02 * dt
            base_drain = drain_rate * dt
            self._batteries[drone_id] -= (base_drain + effort_drain)
            self._batteries[drone_id] = max(0.0, self._batteries[drone_id])

        # Wait for all movement commands to complete
        time.sleep(dt)

        # Update trails and check collisions
        states = self.get_observations()
        for drone_id, state in states.items():
            self.trails[drone_id].append(state.position.copy())

        self._detect_collisions()
        self._step_count += 1

        return states

    # ------------------------------------------------------------------ #
    #  Observations
    # ------------------------------------------------------------------ #

    def get_observations(self) -> dict[str, DroneState]:
        """Read telemetry and camera images from all AirSim vehicles."""
        states = {}

        for dc in self.map_config.drone_configs:
            drone_id = dc["id"]
            drone_type = dc["type"]

            try:
                # Get multirotor state (position, velocity)
                ms = self.client.getMultirotorState(vehicle_name=drone_id)
                pos = ms.kinematics_estimated.position
                vel = ms.kinematics_estimated.linear_velocity

                # AirSim uses NED coordinates — convert z to positive-up
                position = np.array([pos.x_val, pos.y_val, -pos.z_val])
                velocity = np.array([vel.x_val, vel.y_val, -vel.z_val])

                # Capture camera image
                camera_img = self._get_camera_image(drone_id)

                # Determine mission status from battery
                battery = self._batteries.get(drone_id, 100.0)
                if battery <= 0:
                    status = MissionStatus.DEAD
                elif battery < 20.0:
                    status = MissionStatus.RTB
                else:
                    status = MissionStatus.ACTIVE

                states[drone_id] = DroneState(
                    drone_id=drone_id,
                    drone_type=drone_type,
                    position=position,
                    velocity=velocity,
                    battery=battery,
                    mission_status=status,
                    assigned_task=None,
                    camera_image=camera_img,
                )

            except Exception as e:
                warnings.warn(f"Failed to read state for {drone_id}: {e}")
                states[drone_id] = DroneState(
                    drone_id=drone_id,
                    drone_type=drone_type,
                    mission_status=MissionStatus.DEAD,
                )

        return states

    def _get_camera_image(self, drone_id: str) -> np.ndarray:
        """Capture an RGB camera image from AirSim for the specified vehicle.

        Returns:
            RGB image as numpy array of shape (H, W, 3) uint8.
        """
        res = self.map_config.camera_resolution if self.map_config else (84, 84)
        w, h = res

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    "0",                            # Camera name
                    airsim.ImageType.Scene,          # RGB scene
                    False,                           # Not float
                    False,                           # Not compressed
                )
            ], vehicle_name=drone_id)

            if responses and responses[0].width > 0:
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img = img_1d.reshape(responses[0].height, responses[0].width, 3)

                # Resize to target resolution if needed
                if img.shape[:2] != (h, w):
                    from PIL import Image
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((w, h), Image.BILINEAR)
                    img = np.array(pil_img)

                return img

        except Exception:
            pass

        # Fallback: return blank image
        return np.zeros((h, w, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    #  Collisions
    # ------------------------------------------------------------------ #

    def check_collisions(self) -> list[CollisionEvent]:
        """Return collisions detected during the last step."""
        return list(self._collision_buffer)

    def _detect_collisions(self):
        """Query AirSim's built-in collision detection for each vehicle."""
        for drone_id in self.vehicle_names:
            try:
                col_info = self.client.simGetCollisionInfo(vehicle_name=drone_id)
                if col_info.has_collided:
                    self._collision_buffer.append(CollisionEvent(
                        drone_id=drone_id,
                        collided_with=col_info.object_name or "unknown",
                        collision_type="drone-obstacle",
                        position=np.array([
                            col_info.position.x_val,
                            col_info.position.y_val,
                            -col_info.position.z_val,
                        ]),
                        penetration_depth=col_info.penetration_depth,
                    ))
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def get_render_data(self) -> dict:
        """Return data for the Matplotlib renderer.

        Note: With AirSim, you typically use its built-in Unreal Engine
        rendering. This method provides data for the framework's own
        simple renderer as a fallback.
        """
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
