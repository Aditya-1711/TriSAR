# 🚀 How to Run TriSAR Swarm Framework

This guide covers all methods to run and evaluate the **TriSAR** multi-UAV swarm framework — including local simulation, 3D Gazebo rendering, ROS 2 multi-agent deployment, containerized Docker execution, and paper statistical reproduction.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Option A — Quickstart Simulation (WSL / Linux / macOS)](#2-option-a--quickstart-simulation-wsl--linux--macos)
3. [Option B — Quickstart Simulation (Windows PowerShell)](#3-option-b--quickstart-simulation-windows-powershell)
4. [Option C — 3D Gazebo Simulator Rendering (WSL 2)](#4-option-c--3d-gazebo-simulator-rendering-wsl-2)
5. [Option D — ROS 2 Multi-Agent Swarm Launch](#5-option-d--ros-2-multi-agent-swarm-launch)
6. [Option E — Fully Containerized Execution (Docker)](#6-option-e--fully-containerized-execution-docker)
7. [Option F — Paper Statistical Analysis & Plot Reproduction](#7-option-f--paper-statistical-analysis--plot-reproduction)
8. [Troubleshooting & Gotchas](#8-troubleshooting--gotchas)

---

## 1. Prerequisites

| Component | Minimum Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Local simulation & statistical evaluation |
| **Gazebo Sim** | Harmonic / Garden / Fortress | 3D physical disaster environment rendering |
| **ROS 2** | Humble / Iron / Jazzy | Native multi-agent node communication |
| **Docker** | 24.0+ | Zero-dependency containerized execution |

---

## 2. Option A — Quickstart Simulation (WSL / Linux / macOS)

### Step 1: Navigate to repository root
```bash
cd "/mnt/d/major project"
```

### Step 2: Install Python dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 3: Run single Search & Rescue episode
```bash
PYTHONPATH=. python3 experiments/run_episode.py
```

---

## 3. Option B — Quickstart Simulation (Windows PowerShell)

```powershell
cd "D:\major project"
$env:PYTHONPATH='.'
python experiments/run_episode.py
```

---

## 4. Option C — 3D Gazebo Simulator Rendering (WSL 2)

### Terminal 1: Launch Gazebo 3D Disaster Metropolis World
```bash
# Fix WSL runtime permission if needed
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR

# Launch 3D Gazebo environment
gz sim simulation/worlds/city_earthquake_fire.sdf
```
*(In Gazebo GUI, click the **Play ▶** button at the bottom-left corner)*

### Terminal 2: Run Autonomous Swarm Controller
```bash
cd "/mnt/d/major project"
PYTHONPATH=. python3 experiments/run_episode.py
```
*Note: `experiments/run_episode.py` automatically launches the background UDP Pose Bridge (`wsl_gz_pose_bridge.py`) on port 9876 to stream real-time drone movements into Gazebo.*

---

## 5. Option D — ROS 2 Multi-Agent Swarm Launch

```bash
cd "/mnt/d/major project"
source /opt/ros/humble/setup.bash

# Launch ROS 2 swarm nodes & Gazebo pose bridge
ros2 launch simulation/launch/gazebo_swarm.launch.py
```

---

## 6. Option E — Fully Containerized Execution (Docker)

```bash
# Build Docker image
docker build -t trisar .

# Run test episode (outputs SLAM map PNG to ./logs)
docker run --rm -v $(pwd)/logs:/app/logs trisar

# Run statistical ANOVA analysis inside Docker
docker run --rm trisar python analysis/statistical_analysis.py
```

---

## 7. Option F — Paper Statistical Analysis & Plot Reproduction

### 1. Reproduce Table 1 (Ablation Benchmark Summary Statistics)
```bash
PYTHONPATH=. python3 analysis/generate_table1.py
```

### 2. Reproduce Two-Way ANOVA & Fisher's Exact Significance Tests
```bash
PYTHONPATH=. python3 analysis/statistical_analysis.py
```

### 3. Reproduce Figure 4 (3D Swarm Trajectory Comparison Plots)
```bash
PYTHONPATH=. python3 analysis/generate_figure4.py
```

---

## 8. Troubleshooting & Gotchas

* **`ModuleNotFoundError: No module named 'gymnasium'`**
  * Run `pip install -r requirements.txt --break-system-packages` (or use a virtual environment).
* **`QStandardPaths: error creating runtime directory` in WSL**
  * Run `export XDG_RUNTIME_DIR=/tmp/runtime-$USER && mkdir -p $XDG_RUNTIME_DIR` in WSL.
* **Drones stationary in Gazebo GUI**
  * Ensure the orange **Play (▶)** button in the bottom-left corner of Gazebo Sim is clicked.
