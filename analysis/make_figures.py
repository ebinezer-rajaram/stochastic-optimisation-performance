"""
Analysis script for generating figures from experiment results.

Loads RunLog JSON files and creates visualizations for algorithm comparison.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from logging_utils import load_runlog_json


def load_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """
    Load all RunLog JSON files from results directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of dictionaries containing run logs
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found!")
        return []

    logs = []
    for json_file in results_path.glob("*.json"):
        log = load_runlog_json(str(json_file))
        logs.append(log)

    return logs


def plot_convergence_curves(logs: List[Dict[str, Any]], output_file: str = "figures/convergence.png"):
    """
    Plot convergence curves showing best function value over evaluations.

    Args:
        logs: List of run log dictionaries
        output_file: Output file path for the figure
    """
    if not logs:
        print("No logs to plot!")
        return

    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Group by algorithm
    algo_logs = {}
    for log in logs:
        algo_name = log["algo_name"]
        if algo_name not in algo_logs:
            algo_logs[algo_name] = []
        algo_logs[algo_name].append(log)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, algo_runs in algo_logs.items():
        # Extract convergence curves
        curves = [run["best_f_by_eval"] for run in algo_runs]

        # Convert to array and compute statistics
        max_len = max(len(c) for c in curves)
        curves_padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        curves_array = np.array(curves_padded)

        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)
        evals = np.arange(len(mean_curve))

        # Plot mean with confidence interval
        ax.plot(evals, mean_curve, label=f"{algo_name} (n={len(curves)})", linewidth=2)
        ax.fill_between(evals, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    ax.set_xlabel("Function Evaluations", fontsize=12)
    ax.set_ylabel("Best Function Value", fontsize=12)
    ax.set_title("Convergence Curves", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[SAVED] Convergence plot to {output_file}")
    plt.close()


def plot_basin_distribution(logs: List[Dict[str, Any]], output_file: str = "figures/basin_distribution.png"):
    """
    Plot distribution of final solutions across basins of attraction.

    Args:
        logs: List of run log dictionaries
        output_file: Output file path for the figure
    """
    if not logs:
        print("No logs to plot!")
        return

    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Group by algorithm
    algo_logs = {}
    for log in logs:
        algo_name = log["algo_name"]
        if algo_name not in algo_logs:
            algo_logs[algo_name] = []
        algo_logs[algo_name].append(log)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    algos = list(algo_logs.keys())
    x_pos = np.arange(len(algos))
    width = 0.2

    # Get all unique basin IDs
    all_basins = set()
    for log in logs:
        all_basins.add(log["basin_id"])
    basins = sorted(list(all_basins))

    # Plot bars for each basin
    for i, basin_id in enumerate(basins):
        counts = []
        for algo in algos:
            basin_count = sum(1 for log in algo_logs[algo] if log["basin_id"] == basin_id)
            counts.append(basin_count)

        ax.bar(x_pos + i * width, counts, width, label=f"Basin {basin_id}")

    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Number of Runs", fontsize=12)
    ax.set_title("Distribution of Final Solutions Across Basins", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width * (len(basins) - 1) / 2)
    ax.set_xticklabels(algos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[SAVED] Basin distribution plot to {output_file}")
    plt.close()


def generate_summary_table(logs: List[Dict[str, Any]], output_file: str = "figures/summary.csv"):
    """
    Generate summary statistics table for all algorithms.

    Args:
        logs: List of run log dictionaries
        output_file: Output file path for the CSV table
    """
    if not logs:
        print("No logs to summarize!")
        return

    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Group by algorithm
    algo_logs = {}
    for log in logs:
        algo_name = log["algo_name"]
        if algo_name not in algo_logs:
            algo_logs[algo_name] = []
        algo_logs[algo_name].append(log)

    # Compute statistics
    summary_data = []
    for algo_name, algo_runs in algo_logs.items():
        final_values = [run["final_best_f"] for run in algo_runs]

        summary_data.append({
            "Algorithm": algo_name,
            "Runs": len(algo_runs),
            "Mean Best": np.mean(final_values),
            "Std Best": np.std(final_values),
            "Min Best": np.min(final_values),
            "Max Best": np.max(final_values)
        })

    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False, float_format="%.6f")
    print(f"[SAVED] Summary table to {output_file}")
    print("\nSummary Statistics:")
    print(df.to_string(index=False))


def main():
    """Load results and generate all figures."""
    print("Loading experiment results...")
    logs = load_all_results("results")

    if not logs:
        print("No results found! Run experiments first.")
        return

    print(f"Loaded {len(logs)} experiment runs")

    # Generate figures
    print("\nGenerating figures...")
    plot_convergence_curves(logs)
    plot_basin_distribution(logs)
    generate_summary_table(logs)

    print("\n[DONE] Analysis complete!")


if __name__ == "__main__":
    main()
