"""
Import verification test suite for TriSAR multi-UAV framework.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_imports():
    import src.allocation.ga_allocator
    import src.allocation.greedy_allocator
    import src.control.pso_controller
    import src.control.repulsion
    import src.energy.energy_model
    import src.evaluation.collision_detection
    print("All core framework modules imported successfully!")

if __name__ == "__main__":
    test_imports()
