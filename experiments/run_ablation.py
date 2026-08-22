#!/usr/bin/env python3
"""
Automated 30-episode batch ablation study runner for TriSAR.
Executes batch benchmarks across all four experimental conditions:
1. full (GA Allocator + PSO + Repulsion)
2. no_ga (Greedy Allocator + PSO + Repulsion)
3. no_repulsion (GA Allocator + PSO - Repulsion)
4. floor (Greedy Allocator + PSO - Repulsion)
"""

import os
import sys
import subprocess

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def main():
    print("=" * 70)
    print("  TriSAR Automated 30-Episode Ablation Benchmark Runner")
    print("=" * 70)

    cmd = [sys.executable, os.path.join(repo_root, "legacy_benchmarks", "ablation_study.py")]
    res = subprocess.run(cmd, cwd=repo_root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
