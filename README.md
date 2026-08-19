# TriSAR: Evaluating GA Task Allocation and PSO-Based Reactive Collision Avoidance in Multi-UAV Disaster Response

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Conference: TAROS 2026](https://img.shields.io/badge/Conference-TAROS%202026-brightgreen.svg)](https://www.taros2026.org)

Official code repository and benchmark evaluation suite for the paper:  
**"TriSAR: Evaluating GA Task Allocation and PSO-Based Reactive Collision Avoidance in Multi-UAV Disaster Response"**  
Submitted to *Proceedings of Towards Autonomous Robotic Systems (TAROS 2026)*.

---

## 📌 Abstract

Unmanned Aerial Vehicle (UAV) swarms offer transformative capabilities for disaster response, search-and-rescue, and emergency aerial mapping in GPS-denied, post-earthquake urban environments. However, scaling multi-drone operations requires solving NP-hard combinatorial task allocation while simultaneously maintaining real-time 3D flight safety under tight inter-agent proximity.

**TriSAR** introduces a decoupled hierarchical framework integrating a **Genetic Algorithm (GA)** for multi-objective target allocation with a **Particle Swarm Optimization (PSO)** controller incorporating an exponential potential-field inter-agent repulsion mechanism. Across a balanced factorial ablation study ($N = 120$ matched simulation episodes, $n = 30$ per condition), we demonstrate that active inter-agent repulsion provides critical flight safety, reducing collision rate from **53.3% ($16/30$) to 6.7% ($2/30$)** ($p = 0.000145$, relative risk $8.00\times$), while preserving target completion efficiency and battery endurance.

---

## 🏗️ System Architecture

TriSAR decouples global target distribution from local 3D trajectory generation:

```text
                                  +-----------------------------+
                                  |   Blackboard Shared State   |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |  GA Task Allocator          |
                                  |  (Combinatorial Search)     |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |  3D PSO Flight Controller   |
                                  |  + Inter-Drone Repulsion    |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |  ROS 2 / Gazebo Sim Backend |
                                  |  + Voxel SLAM & Mesh Relay  |
                                  +-----------------------------+
```

1. **Global Task Allocation (GA):** Evaluates multi-drone target assignments maximizing rescue urgency while minimizing total swarm travel distance.
2. **Local 3D Navigation (PSO):** Computes continuous velocity commands $v_i \in \mathbb{R}^3$ for each drone with dynamic velocity clamping ($15\text{ m/s}$).
3. **Reactive Collision Avoidance (Repulsion):** Applies an artificial potential field force when inter-drone distance falls below the safe boundary ($d_{\text{safe}} = 2.5\text{ m}$).
4. **Voxel Mapping & Mesh Relay:** Maintains a real-time 3D SLAM occupancy grid while dedicating high-altitude stationary drones as ad-hoc wireless communication relays.

---

## 📂 Repository Structure

```text
TriSAR/
├── README.md                          # Academic overview & paper reproduction guide
├── LICENSE                            # MIT License (2026 Aditya Anil Kapile)
├── CITATION.cff                       # Citation metadata (Kapile, Kennedy, Machado 2026)
├── requirements.txt                   # Standard Python dependencies
├── environment.yml                    # Conda environment specification file
├── Dockerfile                         # Container build configuration
├── configs/                           # Experimental condition configs
│   ├── full.yaml                      # GA Allocator + PSO + Repulsion (Proposed)
│   ├── no_ga.yaml                     # Greedy Allocator + PSO + Repulsion
│   ├── no_repulsion.yaml              # GA Allocator + PSO - Repulsion
│   └── floor.yaml                     # Greedy Allocator + PSO - Repulsion
├── simulation/                        # Gazebo world models and ROS 2 launch files
│   ├── worlds/                        # Gazebo 3D urban environment SDF models
│   ├── models/                        # Custom CAD/SDF model assets
│   └── launch/                        # ROS 2 & Gazebo launch manifests
├── src/                               # Core Python framework packages
│   ├── allocation/                    # GA & Greedy task allocators (ga_allocator.py, greedy_allocator.py)
│   ├── control/                       # 3D PSO controller & repulsion (pso_controller.py, repulsion.py)
│   ├── energy/                        # Aerodynamic drag battery model (energy_model.py)
│   └── evaluation/                    # Metrics logger & collision detector (metrics.py, collision_detection.py)
├── experiments/                       # Execution entry points
│   ├── run_episode.py                 # Single episode execution wrapper
│   ├── run_ablation.py                # 30-episode automated ablation study runner
│   └── run_all_experiments.sh         # Master shell execution script
├── data/                              # Data dictionary & processed CSVs
│   ├── raw/                           # Raw episode datasets
│   └── processed/                     # Summarized CSVs & data_dictionary.md
├── analysis/                          # Statistical analysis & figure generation
│   ├── prepare_data.py                # Raw JSON to CSV aggregator
│   ├── statistical_analysis.py        # Two-Way ANOVA & Fisher's exact test
│   ├── generate_table1.py             # Table 1 summary statistics reporter
│   └── generate_figure4.py            # Figure 4 3D trajectory plot generator
├── figures/                           # High-resolution paper figures (Figure 4a/b PNGs)
├── legacy_benchmarks/                 # Raw ablation JSON dataset (commit bd73c88) & benchmark scripts
├── tests/                             # Test suite directory (test_imports.py)
└── docs/                              # Project specifications & study guide documentation
```

---

## 🛠️ Installation & Setup

### Option A: Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/Aditya-1711/TriShield.git
cd TriShield

# Create and activate Conda environment
conda env create -f environment.yml
conda activate trisar
```

### Option B: Using Pip

```bash
pip install -r requirements.txt
```

### Option C: Using Docker (Fully Containerized)

Build and run the complete simulation environment in a zero-dependency Docker container:

```bash
# Build Docker container image
docker build -t trisar .

# Run container (Executes test episode and outputs logs & SLAM map)
docker run --rm -v $(pwd)/logs:/app/logs trisar

# Run statistical ANOVA analysis inside container
docker run --rm trisar python analysis/statistical_analysis.py
```

---

## 🚀 Quickstart & Reproduction Commands

### 1. Single Demonstration Run

Run a single 3D search-and-rescue episode under the proposed `FULL` condition:

```bash
PYTHONPATH=. python experiments/run_episode.py
```

### 2. Reproduce Table 1 (Ablation Benchmark Summary)

Generate summary statistics across all $120$ matched canonical ablation runs ($N=30$ per condition):

```bash
PYTHONPATH=. python analysis/generate_table1.py
```

### 3. Reproduce Statistical Analysis (Two-Way ANOVA & Fisher's Test)

Run two-way ANOVA for continuous metrics (steps, path length, energy) and Fisher's exact test for collision outcome safety:

```bash
PYTHONPATH=. python analysis/statistical_analysis.py
```

### 4. Reproduce Figure 4 (3D Flight Trajectories)

Generate high-resolution 3D swarm flight trajectory comparison plots:

```bash
PYTHONPATH=. python analysis/generate_figure4.py
```

---

## 📊 Experimental Results

### Table 1: Ablation Study Benchmark Summary ($N = 120$ total, $n = 30$ per cell, Mean $\pm$ Population SD)

| Experimental Variant | Allocation Mechanism | Repulsion Mechanism | Success Rate | Total Steps | Path Length (m) | Battery Consumed (%) | Collision Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FULL** *(Proposed)* | Genetic Algorithm | Active ($d_{\text{safe}}=2.5\text{m}$) | **100.0%** | $67.03 \pm 13.61$ | $437.51 \pm 74.90$ | $55.68 \pm 7.57$ | **6.7%** ($2/30$) |
| **NO_GA** | Greedy (Nearest) | Active ($d_{\text{safe}}=2.5\text{m}$) | **100.0%** | $62.23 \pm 16.67$ | $423.01 \pm 68.73$ | $53.58 \pm 6.77$ | **0.0%** ($0/30$) |
| **NO_REPULSION** | Genetic Algorithm | Disabled | **100.0%** | $61.83 \pm 12.15$ | $416.86 \pm 64.43$ | $52.70 \pm 6.83$ | **53.3%** ($16/30$) |
| **FLOOR** | Greedy (Nearest) | Disabled | **100.0%** | $67.67 \pm 22.41$ | $438.70 \pm 86.19$ | $55.34 \pm 8.65$ | **0.0%** ($0/30$) |

### Statistical Significance Highlights

* **Collision Avoidance Protective Effect:** Fisher's exact test demonstrates a highly significant reduction in collision incidence when potential-field repulsion is active under GA allocation (**$6.7\%$ vs $53.3\%$, $p = 0.000145$**, Relative Risk = $8.00\times$).
* **Task Allocation Timing:** High-precision timing confirms that GA allocation completes in **$61.19 \pm 4.96\text{ ms}$** (16–19 generations), well within real-time mission planning constraints.

---

## 📜 Citation

If you use TriSAR or find this codebase useful in your research, please cite our TAROS 2026 paper:

```bibtex
@inproceedings{kapile2026trisar,
  title     = {TriSAR: Evaluating GA Task Allocation and PSO-Based Reactive Collision Avoidance in Multi-UAV Disaster Response},
  author    = {Kapile, Aditya Anil and Kennedy, Isibor and Machado, Pedro},
  booktitle = {Proceedings of Towards Autonomous Robotic Systems (TAROS 2026)},
  year      = {2026}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
