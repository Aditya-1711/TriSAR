# 🚀 How to Run TriShield

This guide covers every way to run the TriShield swarm system — from a quick local simulation on your laptop to a full ROS 2 multi-agent deployment.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Option A — Run Locally (PowerShell / Windows)](#2-option-a--run-locally-powershell--windows)
3. [Option B — Run Locally (Linux / macOS)](#3-option-b--run-locally-linux--macos)
4. [Option C — Run Gymnasium Framework (Local)](#4-option-c--run-gymnasium-framework-local)
5. [Option D — Run with Docker (Any OS)](#5-option-d--run-with-docker-any-os)
6. [Option E — Full ROS 2 Swarm (Linux / Docker)](#6-option-e--full-ros-2-swarm-linux--docker)
7. [Viewing the Output](#7-viewing-the-output)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement            | Needed For          | Install Link                                      |
| ---------------------- | ------------------- | ------------------------------------------------- |
| **Python 3.10+**       | Local simulation    | [python.org](https://www.python.org/downloads/)   |
| **pip**                | Python packages     | Bundled with Python                                |
| **Docker Desktop**     | Docker methods      | [docker.com](https://www.docker.com/get-started/) |
| **Docker Compose v2**  | Multi-container run | Bundled with Docker Desktop                        |
| **ROS 2 Humble**       | Native ROS 2 only   | [ROS 2 Docs](https://docs.ros.org/en/humble/Installation.html) |
| **Git** *(optional)*   | Cloning the repo    | [git-scm.com](https://git-scm.com/)               |

---

## 2. Option A — Run Locally (PowerShell / Windows)

### Step 1: Open PowerShell and navigate to the project

```powershell
cd "D:\major project"
```

### Step 2: Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> [!NOTE]
> If you get an error about script execution policy, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Run the simulation

```powershell
python trishield_core/simulation.py
```

### What happens

- The Genetic Algorithm allocates tasks (threat interception + victim rescue) to agents.
- The PSO optimizer runs 100 frames of swarm movement.
- A GIF is saved to `artifacts/trishield_sim.gif`.

---

## 3. Option B — Run Locally (Linux / macOS)

### Step 1: Navigate to the project

```bash
cd /path/to/major-project
```

### Step 2: Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the simulation

```bash
python trishield_core/simulation.py
```

The output GIF will be saved to `artifacts/trishield_sim.gif`.

---

## 4. Option C — Run Gymnasium Framework (Local)

The new modular drone simulation framework allows running algorithms within a standard Gymnasium environment. Ensure you have activated your virtual environment and installed the dependencies (including `gymnasium`).

### Run with Lightweight Backend
Provides a fast, headless-capable simulation using NumPy.
```powershell
python trishield_gym/examples/run_lightweight.py
```
This runs the simulation with random actions and generates a CSV metrics report in the `artifacts/` folder.

### Run with PSO/GA Integration
Demonstrates the environment using the existing TriShield PSO and GA algorithms.
```powershell
python trishield_gym/examples/run_with_pso.py
```
This will run the full logic ending with a mission complete or fail, and log metrics.

### Run with AirSim Backend (Optional)
Connects the exact same Gymnasium environment to a running Microsoft AirSim (Unreal Engine) instance for realistic physics.
```powershell
python trishield_gym/examples/run_airsim.py
```
*(Requires AirSim to be installed and running: `pip install airsim`)*

---

## 5. Option D — Run with Docker (Any OS)

This is the **easiest** method — no Python install needed. Works on Windows, macOS, and Linux.

### Run the simulation only

```powershell
# Build and run the simulation container
docker compose up trishield-sim --build
```

The GIF output is volume-mapped, so it appears automatically in your local `artifacts/` folder.

### Run both simulation + ROS 2 nodes

```powershell
docker compose up --build
```

This boots:

| Container               | What it does                                  |
| ----------------------- | --------------------------------------------- |
| `trishield_simulation`  | Runs the core Python simulation → outputs GIF |
| `trishield_ros2_nodes`  | Launches the full ROS 2 swarm (5 UAVs + 3 UGVs) with Foxglove bridge on port `8765` |

### Stop everything

```powershell
docker compose down
```

---

## 6. Option E — Full ROS 2 Swarm (Linux / Docker)

> [!IMPORTANT]
> ROS 2 Humble requires **Ubuntu 22.04**. Use Docker on other operating systems.

### Native (Ubuntu 22.04)

```bash
# 1. Source ROS 2
source /opt/ros/humble/setup.bash

# 2. Navigate to a colcon workspace
mkdir -p ~/trishield_ws/src
cp -r /path/to/major-project/trishield_ros2 ~/trishield_ws/src/
cp -r /path/to/major-project/trishield_core ~/trishield_ws/src/

# 3. Build
cd ~/trishield_ws
colcon build --packages-select trishield_ros2

# 4. Source the build
source install/setup.bash

# 5. Launch the swarm
ros2 launch trishield_ros2 swarm_launch.py
```

### What gets launched

| Node                 | Count | Role                                             |
| -------------------- | ----- | ------------------------------------------------ |
| `foxglove_bridge`    | 1     | WebSocket bridge for 3D visualization on port 8765 |
| `global_blackboard`  | 1     | Shared distributed blackboard for inter-agent comms |
| `uav_X_controller`   | 5     | Aerial drone agents (UAV)                        |
| `ugv_X_controller`   | 3     | Ground robot agents (UGV)                        |

### Connect Foxglove Studio for 3D visualization

1. Download [Foxglove Studio](https://foxglove.dev/download).
2. Open it and connect to: `ws://localhost:8765`
3. You will see live ROS 2 topic data from all 8 agents in real time.

---

## 7. Viewing the Output

### Simulation GIF

After running the simulation (any method), open the generated animation:

**PowerShell:**
```powershell
Start-Process "artifacts\trishield_sim.gif"
```

**Linux / macOS:**
```bash
xdg-open artifacts/trishield_sim.gif    # Linux
open artifacts/trishield_sim.gif        # macOS
```

### ROS 2 Topics (if running the swarm)

```bash
# List all active topics
ros2 topic list

# Echo a specific agent's position
ros2 topic echo /uav_1_controller/agent_state
```

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

You haven't installed the dependencies. Run:

```powershell
pip install -r requirements.txt
```

### PowerShell says "cannot be loaded because running scripts is disabled"

Run this once to allow script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Docker build fails on `Dockerfile.ros2`

Make sure Docker Desktop is running and has at least **4 GB RAM** allocated. The ROS 2 base image is large (~1.5 GB).

### GIF is not generated / `Error saving animation`

Ensure `pillow` is installed:

```powershell
pip install pillow>=10.0.0
```

### `colcon build` fails with "package not found"

Make sure both `trishield_ros2/` and `trishield_core/` are inside the `src/` directory of your colcon workspace.

---

> [!TIP]
> **Quickest way to see it working:** Just run `docker compose up trishield-sim --build` — zero setup required beyond having Docker installed.
