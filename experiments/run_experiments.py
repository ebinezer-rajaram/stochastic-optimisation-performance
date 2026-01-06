"""
Comprehensive grid search experiment runner for PSO and GA algorithms.

This script runs a full grid search over algorithm hyperparameters using
paired random seeds for fair comparison. Results are saved as both detailed
JSON runlogs and a summary CSV.

Usage:
    python experiments/run_experiments.py
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from itertools import product
from datetime import datetime

from src.objective import cross_in_tray
from src.pso import pso_optimize
from src.ga import ga_optimize
from src.logging_utils import save_runlog_json, RunLog


# ============================================================================
# CONFIGURATION
# ============================================================================

# Known global minima for Cross-in-Tray function
# Source: Standard Cross-in-Tray test function (symmetry at ±1.34941)
# These are the four symmetric global minima in [-15,15]²
KNOWN_MINIMA = [
    np.array([1.34941, 1.34941]),
    np.array([-1.34941, 1.34941]),
    np.array([1.34941, -1.34941]),
    np.array([-1.34941, -1.34941])
]

# Compute f_star by evaluating objective at known minima (robust approach)
# This ensures we use the exact minimum value from our objective function
F_STAR = min(cross_in_tray(m) for m in KNOWN_MINIMA)

# Target tolerance (configurable, default 1e-3)
EPS = 1e-3
TARGET_F = F_STAR + EPS

# Problem bounds
BOUNDS = (-15.0, 15.0)

# Evaluation budget
BUDGET = 1000

# PSO grid parameters
PSO_GRID = {
    "swarm_size": [5, 10, 15, 20, 30, 40],
    "w": [0.3, 0.4, 0.5, 0.6, 0.7, 0.9],
    "c1": [1.5],  # Fixed as per specification
    "c2": [1.5]   # Fixed as per specification
}

# GA grid parameters
GA_GRID = {
    "pop_size": [10, 20, 30, 50, 75, 100],
    "mut_sigma": [0.05, 0.1, 0.3, 0.5, 0.7, 1.0],
    "mut_prob": [0.2],  # Fixed as per specification
    "cx_prob": [0.9],   # Fixed as per specification
    "elitism": [1]      # Fixed as per specification
}

# Results directory structure
RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_FILE = RESULTS_DIR / "summary.csv"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_seeds(seed_file: str = "experiments/seeds.json") -> List[int]:
    """Load random seeds from JSON file."""
    with open(seed_file, 'r') as f:
        data = json.load(f)
    return data["seeds"]


def create_pso_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Generate all PSO configurations from grid."""
    configs = []
    for swarm_size, w, c1, c2 in product(
        PSO_GRID["swarm_size"],
        PSO_GRID["w"],
        PSO_GRID["c1"],
        PSO_GRID["c2"]
    ):
        config = {
            "swarm_size": swarm_size,
            "w": w,
            "c1": c1,
            "c2": c2
        }
        # Create a unique config ID
        config_id = f"swarm{swarm_size}_w{w}_c1{c1}_c2{c2}"
        configs.append((config_id, config))
    return configs


def create_ga_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Generate all GA configurations from grid."""
    configs = []
    for pop_size, mut_sigma, mut_prob, cx_prob, elitism in product(
        GA_GRID["pop_size"],
        GA_GRID["mut_sigma"],
        GA_GRID["mut_prob"],
        GA_GRID["cx_prob"],
        GA_GRID["elitism"]
    ):
        config = {
            "pop_size": pop_size,
            "mut_sigma": mut_sigma,
            "mut_prob": mut_prob,
            "cx_prob": cx_prob,
            "elitism": elitism
        }
        # Create a unique config ID
        config_id = f"pop{pop_size}_sigma{mut_sigma}_mp{mut_prob}_cp{cx_prob}_e{elitism}"
        configs.append((config_id, config))
    return configs


def run_pso_experiment(
    seed: int,
    config: Dict[str, Any],
    target_f: float,
    minima: List[np.ndarray]
) -> RunLog:
    """Run a single PSO experiment with given configuration."""
    return pso_optimize(
        objective_fn=cross_in_tray,
        bounds=BOUNDS,
        seed=seed,
        budget=BUDGET,
        swarm_size=config["swarm_size"],
        w=config["w"],
        c1=config["c1"],
        c2=config["c2"],
        target_f=target_f,
        minima=minima
    )


def run_ga_experiment(
    seed: int,
    config: Dict[str, Any],
    target_f: float,
    minima: List[np.ndarray]
) -> RunLog:
    """Run a single GA experiment with given configuration."""
    return ga_optimize(
        objective_fn=cross_in_tray,
        bounds=BOUNDS,
        seed=seed,
        budget=BUDGET,
        pop_size=config["pop_size"],
        cx_prob=config["cx_prob"],
        mut_prob=config["mut_prob"],
        mut_sigma=config["mut_sigma"],
        tournament_k=2,  # Default from GA implementation
        elitism=config["elitism"],
        target_f=target_f,
        minima=minima
    )


def save_results(
    runlog: RunLog,
    algo: str,
    config_id: str,
    seed: int
) -> None:
    """Save runlog to JSON file in appropriate directory structure."""
    # Create directory: results/raw/{algo}/{config_id}/
    output_dir = RAW_DIR / algo / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as: seed_{seed}.json
    output_file = output_dir / f"seed_{seed}.json"
    save_runlog_json(str(output_file), runlog)


def append_to_summary(
    summary_rows: List[Dict[str, Any]],
    runlog: RunLog,
    algo: str,
    config_id: str,
    seed: int,
    f_star: float,
    target_f: float
) -> None:
    """Append run results to summary list."""
    # Robust success criteria: either reached target during run OR final value is within tolerance
    success = (runlog.evals_to_target is not None) or (runlog.final_best_f <= target_f)

    row = {
        "algo": algo,
        "config": config_id,
        "seed": seed,
        "f_star": f_star,
        "target_f": target_f,
        "final_best_f": runlog.final_best_f,
        "evals_to_target": runlog.evals_to_target if runlog.evals_to_target is not None else -1,
        "basin_id": runlog.basin_id,
        "success": success,
        "final_best_x1": runlog.final_best_x[0],
        "final_best_x2": runlog.final_best_x[1]
    }
    summary_rows.append(row)


def save_summary_csv(summary_rows: List[Dict[str, Any]]) -> None:
    """Save summary results to CSV file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define CSV headers (including f_star and target_f)
    fieldnames = [
        "algo", "config", "seed", "f_star", "target_f", "final_best_f",
        "evals_to_target", "basin_id", "success", "final_best_x1", "final_best_x2"
    ]

    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary saved to: {SUMMARY_FILE}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    """Main experiment execution function."""
    print("=" * 80)
    print("STOCHASTIC OPTIMIZATION PERFORMANCE EXPERIMENTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load seeds
    print("Loading seeds...")
    seeds = load_seeds()
    print(f"  Loaded {len(seeds)} seeds: {seeds}")
    print()

    # Generate configurations
    print("Generating configurations...")
    pso_configs = create_pso_configs()
    ga_configs = create_ga_configs()
    print(f"  PSO configurations: {len(pso_configs)}")
    print(f"  GA configurations: {len(ga_configs)}")
    print()

    # Calculate total runs
    total_pso_runs = len(pso_configs) * len(seeds)
    total_ga_runs = len(ga_configs) * len(seeds)
    total_runs = total_pso_runs + total_ga_runs
    print(f"Total runs to execute: {total_runs}")
    print(f"  PSO: {total_pso_runs} runs")
    print(f"  GA: {total_ga_runs} runs")
    print()

    # Storage for summary
    summary_rows = []

    # Run counters
    run_count = 0
    pso_success_count = 0
    ga_success_count = 0

    # ========================================================================
    # RUN PSO EXPERIMENTS
    # ========================================================================
    print("-" * 80)
    print("RUNNING PSO EXPERIMENTS")
    print("-" * 80)
    print(f"f_star = {F_STAR:.8f}")
    print(f"target_f = {TARGET_F:.8f} (f_star + {EPS})")
    print()

    for config_idx, (config_id, config) in enumerate(pso_configs, 1):
        print(f"\nPSO Config {config_idx}/{len(pso_configs)}: {config_id}")
        print(f"  Parameters: {config}")

        for seed_idx, seed in enumerate(seeds, 1):
            run_count += 1

            # Run experiment
            runlog = run_pso_experiment(seed, config, TARGET_F, KNOWN_MINIMA)

            # Save results
            save_results(runlog, "PSO", config_id, seed)
            append_to_summary(summary_rows, runlog, "PSO", config_id, seed, F_STAR, TARGET_F)

            # Track success with robust criteria
            success = (runlog.evals_to_target is not None) or (runlog.final_best_f <= TARGET_F)
            if success:
                pso_success_count += 1
                if runlog.evals_to_target is not None:
                    status = f"SUCCESS (evals={runlog.evals_to_target})"
                else:
                    status = f"SUCCESS (final_best_f={runlog.final_best_f:.6f} <= target)"
            else:
                delta = runlog.final_best_f - F_STAR
                status = f"FAILED (best_f={runlog.final_best_f:.6f}, delta={delta:.6f})"

            # Progress update
            print(f"    Seed {seed_idx}/{len(seeds)} (seed={seed}): {status} | "
                  f"Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")

    # ========================================================================
    # RUN GA EXPERIMENTS
    # ========================================================================
    print()
    print("-" * 80)
    print("RUNNING GA EXPERIMENTS")
    print("-" * 80)
    print(f"f_star = {F_STAR:.8f}")
    print(f"target_f = {TARGET_F:.8f} (f_star + {EPS})")
    print()

    for config_idx, (config_id, config) in enumerate(ga_configs, 1):
        print(f"\nGA Config {config_idx}/{len(ga_configs)}: {config_id}")
        print(f"  Parameters: {config}")

        for seed_idx, seed in enumerate(seeds, 1):
            run_count += 1

            # Run experiment
            runlog = run_ga_experiment(seed, config, TARGET_F, KNOWN_MINIMA)

            # Save results
            save_results(runlog, "GA", config_id, seed)
            append_to_summary(summary_rows, runlog, "GA", config_id, seed, F_STAR, TARGET_F)

            # Track success with robust criteria
            success = (runlog.evals_to_target is not None) or (runlog.final_best_f <= TARGET_F)
            if success:
                ga_success_count += 1
                if runlog.evals_to_target is not None:
                    status = f"SUCCESS (evals={runlog.evals_to_target})"
                else:
                    status = f"SUCCESS (final_best_f={runlog.final_best_f:.6f} <= target)"
            else:
                delta = runlog.final_best_f - F_STAR
                status = f"FAILED (best_f={runlog.final_best_f:.6f}, delta={delta:.6f})"

            # Progress update
            print(f"    Seed {seed_idx}/{len(seeds)} (seed={seed}): {status} | "
                  f"Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")

    # ========================================================================
    # SAVE SUMMARY AND FINAL REPORT
    # ========================================================================
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    save_summary_csv(summary_rows)

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total runs completed: {run_count}")
    print()
    print(f"PSO Results:")
    print(f"  Total runs: {total_pso_runs}")
    print(f"  Successful: {pso_success_count} ({100*pso_success_count/total_pso_runs:.1f}%)")
    print(f"  Failed: {total_pso_runs - pso_success_count}")
    print()
    print(f"GA Results:")
    print(f"  Total runs: {total_ga_runs}")
    print(f"  Successful: {ga_success_count} ({100*ga_success_count/total_ga_runs:.1f}%)")
    print(f"  Failed: {total_ga_runs - ga_success_count}")
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
