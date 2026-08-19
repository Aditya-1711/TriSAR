#!/usr/bin/env python3
"""
Single episode simulation runner wrapper for TriSAR Autonomous Drone Swarm.
Executes 3D flight steps, real-time SLAM occupancy grid mapping, GA task allocation,
3D PSO flight control, and exports JSON performance metrics & SLAM PNG map.
"""

import os
import sys

# Ensure repository root is on PYTHONPATH
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from trishield_gym.examples.run_gazebo import main

if __name__ == "__main__":
    main()
