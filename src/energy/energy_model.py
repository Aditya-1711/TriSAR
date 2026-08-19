import numpy as np

class BatteryEnergyModel:
    """Cubic drag aerodynamics battery energy consumption model for multi-UAV swarm."""

    def __init__(self, battery_capacity_wh=100.0, hover_power_w=150.0, drag_coeff=0.05):
        self.capacity_wh = battery_capacity_wh
        self.hover_power_w = hover_power_w
        self.drag_coeff = drag_coeff

    def compute_power_w(self, speed_mps: float) -> float:
        """Compute aerodynamic power consumption (Watts) incorporating cubic drag."""
        return float(self.hover_power_w + self.drag_coeff * (speed_mps ** 3))

    def compute_step_energy_consumed_pct(self, speed_mps: float, dt_seconds: float) -> float:
        """Compute step energy consumed as percentage points of total battery capacity."""
        power_w = self.compute_power_w(speed_mps)
        energy_wh = power_w * (dt_seconds / 3600.0)
        pct_consumed = (energy_wh / self.capacity_wh) * 100.0
        return float(pct_consumed)
