# TriShield: Multi-Domain Autonomous Swarm System
**Comprehensive Study Guide & Architecture Documentation**
TriShield is a conceptually verified, containerized architecture that bridges pure mathematical theories (Particle Swarm Optimization & Genetic Algorithms) into an industry-grade distributed network (ROS 2 Humble). This project proves that a swarm of physically distinct robotic agents (Aerial UAVs and Ground UGVs) can dynamically allocate tasks and navigate environments without a central command server.

---

## Phase 1: The Core AI Mathematics (`trishield_core`)

To ensure the logic is highly testable and framework-agnostic, the raw mathematics and physics of the swarm were written in pure Python.

### 1. Universal Hardware Abstraction (`agent.py`)
Rather than writing separate codebases for drones and ground rovers, we created a single `Agent` class wrapper.
- **Why?** It enforces *Kinematic Constraints*. The `UAVAgent` allows full 3D translation ($X, Y, Z$). However, the `UGVAgent` strictly intercepts the velocity update and forcefully writes `pos[2] = 0.0`. This ensures ground robots can never "accidentally fly" while testing algorithms.

### 2. Heterogeneous Task Allocation (`ga_allocator.py`)
This script uses a **Genetic Algorithm** to decide which agent handles which threat (e.g., stopping a rogue drone vs. saving a trapped survivor).
- **The Logic:** It calculates the distance/ETA to the target, but highly mutates the "fitness score" based on hardware.
- **Emergent Behavior Example:** It typically applies a $2.0 \times$ massive penalty if a UAV attempts a ground rescue (since UAVs can't carry heavy medkits). However, during our simulation, the UGVs spawned so far away that their speed penalty outweighed the UAV hardware penalty. The swarm dynamically broke its own hardware preference purely to save the survivor faster using the drone!

### 3. Flight Kinematics (`pso_optimizer.py`)
This script uses **Particle Swarm Optimization** to physically route the drones.
- **Macro Avoidance (Attraction):** Instead of calculating an exact path trace, the PSO applies an attractive "gravity" vector (`cognitive_vel`) constantly pulling the drone towards its assigned task coordinates.
- **Micro Avoidance (Repulsion):** During flight, the algorithm iterates over all peer drones. If a peer enters a 3.0-meter radius, an inverse vector is violently added to the drone's velocity, effortlessly diverting flight paths around each other to prevent mid-air collisions.

---

## Phase 2: Distributed ROS 2 Integration (`trishield_ros2`)

A Python simulation uses a single `for-loop` to process data sequentially. Physical robots cannot do this. You must deploy independent code to each drone's internal processor. We achieved this distributed setup using **ROS 2**.

### 1. The Global Radar (`tri_blackboard_node.py`)
This node simulates a network array (like Ground Radar or shared AI perception). Instead of telling drones *where to move*, it simply blasts a JSON payload of Anomalies (Threats/Survivors) blindly onto the `/trishield/global_blackboard` topic at $1$Hz.

### 2. The Universal Node (`tri_agent_node.py`)
This single script is deployed onto every physical chassis in the fleet. It is completely independent and autonomous.
- **Input:** It listens to the Blackboard topic. Once it discovers the layout, it injects the data into the local `PSOOptimizer` logic.
- **Processing:** It computes its theoretical physics entirely locally at $10$Hz.
- **Output:** It publishes pure, standardized geometry (`geometry_msgs/Twist` or `visualization_msgs/Marker`) back out. No drone knows the direct intention of any other drone; they simply exist and react.

### 3. Fleet Orchestration (`swarm_launch.py`)
Rather than opening $5$ terminal windows and running `ros2 run` for each robot, the `launch` descriptor spins up the Blackboard, `UAV_1`, `UAV_2`, and `UGV_1` completely simultaneously and threads their output into a unified terminal feed.

---

## Phase 3: Dockerized Web Telemetry (Foxglove)

ROS 2 requires heavy, native Linux libraries. Building and running a 3D GPU simulator (like Gazebo Harmonic) inside a WSL2 Docker container natively to a Windows host machine is disastrous and locks up systems. We engineered a massive shortcut using **Foxglove Studio**.

### 1. Native Dockerizing (`Dockerfile.ros2`)
We used the official `ros:humble-ros-base` image. The `Dockerfile` securely mounts both the `trishield_core` math module and the `trishield_ros2` network module, and runs `colcon build` to seamlessly compile the Python into internal Linux binaries.

### 2. WebSocket Bridging (`ros-humble-foxglove-bridge`)
Instead of a heavy GPU GUI, the `tri_agent_node.py` actively intercepts the physical X, Y, Z calculations from the internal `math_agent.pos` and translates them into rigid ROS 2 `Marker` objects (Blue Spheres, Green Cubes).
These markers are shot over a local networking WebSocket on port `8765`.

### 3. Rendering The Flow
By simply opening `studio.foxglove.dev` in a standard web browser (like Chrome/Edge) and binding to `ws://localhost:8765`, the browser natively receives the high-speed telemetry and constructs the 3D map seamlessly without any heavy application downloads!

---

## Summary of the Full Node Data Pipeline
1. `docker compose` spins up the Linux Container.
2. `swarm_launch.py` boots 4 separate Nodes and 1 WebBridge.
3. `global_blackboard` broadcasts Target locations to the network.
4. `tri_agent` nodes intercept the target using the `HeterogeneousGA`.
5. `tri_agent` calculates movement physics at 10Hz using the `PSOOptimizer`.
6. `tri_agent` publishes `Marker` objects tracking its location.
7. Foxglove WebSockets intercept the Markers and render the 3D grid live.
