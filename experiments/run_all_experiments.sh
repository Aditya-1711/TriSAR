#!/usr/bin/env bash
# Shell execution script for running single episodes, ablation benchmarks, and statistical analysis

set -e

echo "=========================================================="
echo "  TriSAR Multi-UAV Swarm — Master Experiment Runner"
echo "=========================================================="

# 1. Run single demonstration episode under 'full' variant
echo "[1/3] Running single test episode (full variant)..."
export TRISAR_VARIANT="full"
python3 experiments/run_episode.py

# 2. Run statistical factorial ANOVA analysis
echo "[2/3] Executing statistical ANOVA & Fisher's exact test..."
python3 analysis/statistical_analysis.py

echo "=========================================================="
echo "  Experiments finished successfully!"
echo "=========================================================="
