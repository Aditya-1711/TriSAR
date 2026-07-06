"""
Simulation backends for the TriShield Gymnasium environment.

Available backends:
    - 'lightweight': Fast NumPy-based physics simulator (default)
    - 'airsim': Microsoft AirSim connector for realistic validation
"""

from trishield_gym.backends.base import SimBackend, DroneState, CollisionEvent


def create_backend(backend_name: str, **kwargs) -> SimBackend:
    """Factory function to create a simulation backend by name.

    Args:
        backend_name: Either 'lightweight' or 'airsim'.
        **kwargs: Backend-specific configuration.

    Returns:
        An initialized SimBackend instance.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if backend_name == "lightweight":
        from trishield_gym.backends.lightweight import LightweightBackend
        return LightweightBackend(**kwargs)
    elif backend_name == "airsim":
        from trishield_gym.backends.airsim_backend import AirSimBackend
        return AirSimBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available backends: 'lightweight', 'airsim'"
        )


__all__ = ["SimBackend", "DroneState", "CollisionEvent", "create_backend"]
