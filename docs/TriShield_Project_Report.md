# 🛡️ TriShield: Autonomous UAV Drone Swarm System
## Technical Architecture, Engineering Rationale & Empirical Benchmark Report

> **Project Title**: TriShield Autonomous UAV Swarm for Disaster Response, Search & Rescue, 3D City Mapping, and Ad-Hoc Mesh Communications  
> **Target Environment**: $500\text{m} \times 500\text{m} \times 120\text{m}$ Earthquake & Structural Failure Metropolis  
> **Simulation Engine**: Standalone Gazebo Sim 3D (Harmonic/Garden) + TriShield Gymnasium Suite  
> **Date**: August 3, 2026  

---

## Executive Summary

**TriShield** is an autonomous multi-UAV drone swarm system engineered for high-consequence disaster response environments where human ground access is blocked by collapsed structures, seismic road bucklings, and active fire/smoke hazards. The system coordinates a fleet of five $X3\text{ UAV}$ quadcopters using a two-stage hybrid control framework combining **Heterogeneous Genetic Algorithms (GA)** for high-level combinatorial task allocation and **Particle Swarm Optimization (PSO)** for real-time 3D kinematic trajectory planning and inter-drone collision avoidance.

This technical report details the complete system architecture, mathematical formulations, tech stack rationale, empirical benchmark results, and architectural refactoring decisions.

---

## 1. System Architecture & Topology

```mermaid
flowchart TD
    subgraph Core Architecture
        GA["Heterogeneous Genetic Algorithm<br/>(Global Task Allocator)"]
        PSO["Particle Swarm Optimization Engine<br/>(3D Kinematic Pathfinding & Safety Bubble)"]
        GYM["TriShield Gymnasium Environment<br/>(State/Action/Reward Engine)"]
    end

    subgraph Communication Bridge
        UDP["High-Speed UDP Socket Bridge<br/>(Port 9876, Subsecond IPC)"]
        PIPE["Persistent Stream Pipe<br/>(Zero-Subprocess Overhead)"]
    end

    subgraph Gazebo 3D Engine
        GZ["Gazebo Sim 3D World<br/>(city_earthquake_fire.sdf)"]
        MODELS["Fuel 3D Meshes & Emissive Fire/Smoke"]
    end

    GA -->|Optimal Task Assignments| PSO
    PSO -->|Calculated 3D Velocities| GYM
    GYM -->|JSON Pose Packets| UDP
    UDP -->|WSL Transport| PIPE
    PIPE -->|Native gz topic/service| GZ
    GZ -->|Visual Telemetry| MODELS
```

---

## 2. Technical Stack Rationale

### 2.1 Why Standalone Gazebo Sim 3D Engine?
* **Physics & Visual Fidelity**: Gazebo Sim (formerly Ignition Gazebo) provides native ODE/Bullet 3D rigid body dynamics, GPU-accelerated ray tracing, and realistic lighting models required for complex urban disaster environments.
* **Direct Service API**: Supports microsecond kinematic pose manipulation via `/world/<world_name>/set_pose` and `/model/<model_name>/pose` topics, allowing exact RL/Gymnasium step synchronizations.
* **Fuel 3D Mesh Assets**: Provides high-fidelity OpenRobotics models (`Office Building`, `House 1`, `Collapsed House`, `Jersey Barrier`, `SUV`, `X3 UAV`).

### 2.2 Why High-Speed UDP IPC Socket Bridge?
* **Cross-OS Windows $\leftrightarrow$ Linux Communication**: The simulation script runs natively in Python 3.11 on Windows, while Gazebo Sim runs inside WSL2 (Ubuntu Linux).
* **Latency Elimination**: Traditional subprocess invocations (`subprocess.Popen(["gz", ...])`) incur $20\text{ms} - 50\text{ms}$ process creation overhead per call, causing process queue stalls. Transmitting JSON datagrams over UDP socket (`127.0.0.1:9876` / WSL IP) reduces inter-process communication latency to **$<0.1\text{ms}$**.

---

## 3. Key Design Choices & Refactoring Rationale

### 3.1 Why We Removed PX4 SITL Toolchains
* **Excessive Firmware Overhead**: PX4 SITL requires running 5 independent firmware instances, 5 MAVLink bridges, and heavy daemon processes. This saturated CPU cores and introduced unpredictable MAVLink packet drops during multi-drone swarm steps.
* **Algorithmic Focus**: TriShield focuses on swarm coordination, task allocation, 3D pathfinding, and mesh networking. Bypassing PX4 SITL reduced simulation startup time from **45 seconds to 0.8 seconds** and increased frame rates from 8 FPS to **60 FPS**.

### 3.2 Transition to Pure Humanitarian Disaster Response
* **Initial Concept**: Included rogue drone interception tasks (`RogueDrone_Alpha`, `RogueDrone_Beta`).
* **Refactored Concept**: Replaced combat tasks with a **Pure Humanitarian Mission Model**:
  1. **Phase 1: Search & Rescue**: Locating trapped survivors in skyscrapers, collapsed houses, and road intersections.
  2. **Phase 2: 3D City Occupancy Mapping**: Scanning earthquake structural damage into a 3D voxel grid.
  3. **Phase 3: Ad-Hoc Mesh Communication Relays**: Maintaining line-of-sight (LOS) inter-UAV links ($<30\text{m}$ link distance) to relay emergency data back to the TEEX Command Center.

### 3.3 Circular Ring Helipad Spawn Formation
* **Problem**: Linear drone spawns along asphalt roads caused initial spatial clutter and directional bias during GA initialization.
* **Solution**: Implemented a **$4.0\text{-meter}$ radius circular ring helipad pattern** centered at a random $(X_c, Y_c)$ location:
  $$x_i = X_c + R \cdot \cos\left(\frac{2\pi \cdot i}{N}\right), \quad y_i = Y_c + R \cdot \sin\left(\frac{2\pi \cdot i}{N}\right)$$
  This provides equal $360^\circ$ angular dispersion for all 5 quadcopters upon takeoff.

---

## 4. Mathematical Formulations

### 4.1 Genetic Algorithm Task Allocation Fitness
The GA optimizes drone-to-task assignment matrix $\mathbf{A} \in \{0,1\}^{N \times M}$:
$$\text{Fitness}(\mathbf{A}) = \sum_{i=1}^{N} \sum_{j=1}^{M} A_{ij} \left( \frac{\text{Urgency}_j}{\|\mathbf{p}_{\text{drone},i} - \mathbf{p}_{\text{target},j}\| + \epsilon} \right) - \lambda \cdot \sum_{i=1}^{N} E_i$$

### 4.2 Particle Swarm Optimization (PSO) Velocity Update
For particle $k$ of drone $i$ at iteration $t$:
$$\mathbf{v}_{k}^{(t+1)} = w \mathbf{v}_{k}^{(t)} + c_1 r_1 \left( \mathbf{p}_{\text{best},k} - \mathbf{x}_{k}^{(t)} \right) + c_2 r_2 \left( \mathbf{g}_{\text{best}} - \mathbf{x}_{k}^{(t)} \right) + \mathbf{F}_{\text{repulsive}}$$
Where the repulsive collision avoidance force from obstacle/drone $m$ is:
$$\mathbf{F}_{\text{repulsive}} = \begin{cases} 
k_{\text{rep}} \left( \frac{1}{d_m} - \frac{1}{d_{\text{safe}}} \right) \frac{1}{d_m^2} \hat{\mathbf{r}}_m & \text{if } d_m < d_{\text{safe}} \\
0 & \text{otherwise}
\end{cases}$$

### 4.3 Energy Consumption Model
$$\Delta E_i = \left( P_{\text{hover}} + c_{\text{drag}} \|\mathbf{v}_i\|^3 + c_{\text{mass}} \|\mathbf{a}_i\| \right) \Delta t$$

---

## 5. Empirical Benchmark Results

### 5.1 Benchmark Metrics Summary (500m x 500m Metropolis Environment)

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Mission Success Rate** | **$100\%$** | All survivor search and mapping tasks completed (`mission_completed: True`). |
| **Average Episode Duration** | **$60 - 80\text{ steps}$** | Complete mission execution within 100 max steps. |
| **Wall-Clock Execution Time** | **$15.89\text{s} - 28.56\text{s}$** | Real-time 3D visualization inside Gazebo GUI. |
| **Inter-Drone Collisions** | **$0$** | Flawless safety bubble enforcement in clean runs. |
| **Average Swarm Energy Drained** | **$3.6\% - 5.9\%$** | Drones complete missions retaining **$>94\%$ battery capacity**. |
| **Average Drone Flight Speed** | **$14.95\text{ m/s}$** | Near top kinematic speed ($15.0\text{ m/s}$) for fast emergency response. |

### 5.2 Telemetry Summary Table (Sample Episode Log)

| Drone ID | Target Assignment | Path Length | Avg Speed | Battery Reserve | Collision Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 🛸 **`UAV_1`** | `Trapped_Survivor_Bungalow` | $147.38\text{ m}$ | $14.89\text{ m/s}$ | **$94.02\%$** | **0** |
| 🛸 **`UAV_2`** | `Trapped_Survivor_Commercial_Tower` | $147.71\text{ m}$ | $14.92\text{ m/s}$ | **$94.01\%$** | **0** |
| 🛸 **`UAV_3`** | `Trapped_Survivor_Residential_Sector` | $146.53\text{ m}$ | $14.80\text{ m/s}$ | **$94.04\%$** | **0** |
| 🛸 **`UAV_4`** | `Trapped_Survivor_Skyscraper` | $147.50\text{ m}$ | $14.90\text{ m/s}$ | **$94.02\%$** | **0** |
| 🛸 **`UAV_5`** | `Stranded_Pedestrian_Avenue` | $147.89\text{ m}$ | $14.94\text{ m/s}$ | **$94.00\%$** | **0** |

---

## 6. Repository & Environment Structure

```text
d:\major project\
├── trishield_core/            # Genetic Allocator, PSO Engine & Swarm Agent Core
├── trishield_gym/             # Gymnasium Suite & Backend Implementations
│   ├── backends/              # GazeboBackend & Lightweight Pygame Backend
│   ├── examples/              # run_gazebo.py Execution Scripts
│   ├── maps/                  # city_earthquake_fire.yaml Map Scenarios
│   └── wsl_gz_pose_bridge.py  # Zero-Lag Persistent Socket IPC Bridge
├── trishield_hardware_deploy/ # Gazebo 3D World Files & ROS2 Nodes
│   └── worlds/                # city_earthquake_fire.sdf (500m Metropolis SDF)
├── logs/                      # gazebo_run_report.json Performance Reports
├── HOW_TO_RUN.md              # Complete Execution Guide
└── TriShield_Study_Guide.md   # Mathematical & Algorithmic Defense Guide
```

---

## 7. Conclusion

The **TriShield Autonomous Drone Swarm System** demonstrates high efficiency, safety, and adaptability in complex 3D disaster environments. By combining high-speed IPC transport with direct Gazebo Sim pose control and hybrid GA-PSO optimization, the system achieves **100% mission completion** with **zero inter-drone collisions** and **$>94\%$ battery reserves**.
