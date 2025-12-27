"""
Comprehensive analysis script for generating figures from experiment results.

Reads results/raw JSON files and results/summary.csv to produce:
1) Convergence plot: median best_f_by_eval with IQR band for best 1-2 configs per algorithm
2) ECDF plot of evals_to_target for each algorithm (best config)
3) Heatmaps (3x3) for each algorithm:
   - success rate (fraction with evals_to_target not None)
   - median final_best_f
   - median evals_to_target (on successful runs)
4) Basin coverage bar chart for best configs
5) Trajectory plots on a contour map for 3 representative runs per algorithm
   (success fast, stuck-then-success, fail)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
from logging_utils import load_runlog_json
from objective import cross_in_tray

# Constants
RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_FILE = RESULTS_DIR / "summary.csv"
FIGURES_DIR = RESULTS_DIR / "figures"

# Known global minima for basin identification
KNOWN_MINIMA = [
    np.array([1.3491, 1.3491]),
    np.array([-1.3491, 1.3491]),
    np.array([1.3491, -1.3491]),
    np.array([-1.3491, -1.3491])
]

F_STAR = -20.6261


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_all_raw_results() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Load all raw JSON results from results/raw directory.

    Returns:
        Nested dict: {algo: {config_id: [runlogs...]}}
    """
    if not RAW_DIR.exists():
        print(f"Error: Raw results directory '{RAW_DIR}' not found!")
        return {}

    all_results = {}

    # Iterate through algorithm directories (PSO, GA)
    for algo_dir in RAW_DIR.iterdir():
        if not algo_dir.is_dir():
            continue

        algo_name = algo_dir.name
        all_results[algo_name] = {}

        # Iterate through config directories
        for config_dir in algo_dir.iterdir():
            if not config_dir.is_dir():
                continue

            config_id = config_dir.name
            all_results[algo_name][config_id] = []

            # Load all seed runs for this config
            for json_file in config_dir.glob("seed_*.json"):
                runlog = load_runlog_json(str(json_file))
                all_results[algo_name][config_id].append(runlog)

    return all_results


def load_summary_csv() -> pd.DataFrame:
    """Load the summary CSV file."""
    if not SUMMARY_FILE.exists():
        print(f"Error: Summary file '{SUMMARY_FILE}' not found!")
        return pd.DataFrame()

    return pd.read_csv(SUMMARY_FILE)


def get_best_configs(df: pd.DataFrame, n_configs: int = 2) -> Dict[str, List[str]]:
    """
    Identify the best 1-2 configurations per algorithm based on success rate
    and median evals_to_target.

    Args:
        df: Summary DataFrame
        n_configs: Number of best configs to return per algorithm

    Returns:
        Dict mapping algo -> list of best config_ids
    """
    best_configs = {}

    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo]

        # Group by config
        config_stats = algo_data.groupby('config').agg({
            'success': 'mean',  # success rate
            'evals_to_target': lambda x: x[x > 0].median() if (x > 0).any() else float('inf')
        }).reset_index()

        config_stats.columns = ['config', 'success_rate', 'median_evals']

        # Sort by success rate (descending), then by median evals (ascending)
        config_stats = config_stats.sort_values(
            by=['success_rate', 'median_evals'],
            ascending=[False, True]
        )

        # Take top n configs
        best_configs[algo] = config_stats.head(n_configs)['config'].tolist()

    return best_configs


# ============================================================================
# FIGURE 1: CONVERGENCE PLOT WITH MEDIAN AND IQR
# ============================================================================

def plot_convergence(raw_results: Dict[str, Dict[str, List[Dict]]],
                    best_configs: Dict[str, List[str]],
                    output_file: str = "convergence.png"):
    """
    Plot convergence curves showing median best_f_by_eval with IQR bands
    for the best 1-2 configurations per algorithm.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, configs in best_configs.items():
        for config_id in configs:
            if config_id not in raw_results[algo]:
                continue

            runs = raw_results[algo][config_id]

            # Extract convergence curves
            curves = [run["best_f_by_eval"] for run in runs]

            # Pad to same length
            max_len = max(len(c) for c in curves)
            curves_padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
            curves_array = np.array(curves_padded)

            # Compute median and IQR
            median_curve = np.median(curves_array, axis=0)
            q25 = np.percentile(curves_array, 25, axis=0)
            q75 = np.percentile(curves_array, 75, axis=0)
            evals = np.arange(1, len(median_curve) + 1)

            # Plot median line
            label = f"{algo}:{config_id[:15]}" if len(configs) > 1 else algo
            line = ax.plot(evals, median_curve, label=label, linewidth=2, alpha=0.9)
            color = line[0].get_color()

            # Plot IQR band
            ax.fill_between(evals, q25, q75, alpha=0.25, color=color)

    ax.axhline(y=F_STAR, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Target: {F_STAR:.4f}')

    ax.set_xlabel("Function Evaluations", fontsize=12, fontweight='bold')
    ax.set_ylabel("Best Function Value", fontsize=12, fontweight='bold')
    ax.set_title("Convergence Curves (Median with IQR)", fontsize=14, fontweight="bold")
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)

    plt.tight_layout()
    output_path = FIGURES_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Convergence plot to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 2: ECDF PLOT OF EVALS_TO_TARGET
# ============================================================================

def plot_ecdf(df: pd.DataFrame, best_configs: Dict[str, List[str]],
              output_file: str = "ecdf_evals_to_target.png"):
    """
    Plot ECDF of evals_to_target for the best configuration of each algorithm.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, configs in best_configs.items():
        # Use the best config (first in list)
        best_config = configs[0]

        # Filter data for this algo and config
        data = df[(df['algo'] == algo) & (df['config'] == best_config)]

        # Get successful runs only (evals_to_target > 0)
        evals = data[data['evals_to_target'] > 0]['evals_to_target'].values

        if len(evals) == 0:
            print(f"Warning: No successful runs for {algo}:{best_config}")
            continue

        # Compute ECDF
        evals_sorted = np.sort(evals)
        ecdf = np.arange(1, len(evals_sorted) + 1) / len(evals_sorted)

        # Plot ECDF
        ax.plot(evals_sorted, ecdf, marker='o', markersize=4, linewidth=2,
                label=f"{algo} (n={len(evals)})", alpha=0.9)

    ax.set_xlabel("Evaluations to Target", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight='bold')
    ax.set_title("ECDF of Evaluations to Target (Best Config per Algorithm)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = FIGURES_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] ECDF plot to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 3: HEATMAPS (3x3) FOR EACH ALGORITHM
# ============================================================================

def plot_heatmaps(df: pd.DataFrame, algo: str, grid_params: Dict[str, List],
                 output_file: str):
    """
    Create 3x3 heatmaps for a single algorithm showing:
    1) Success rate
    2) Median final_best_f
    3) Median evals_to_target (successful runs only)

    Args:
        df: Summary DataFrame
        algo: Algorithm name (PSO or GA)
        grid_params: Dict mapping parameter names to their grid values
        output_file: Output filename
    """
    # Determine which parameters vary (for PSO: swarm_size and w; for GA: pop_size and mut_sigma)
    if algo == "PSO":
        param1_name, param2_name = "swarm_size", "w"
        param1_values = grid_params["swarm_size"]
        param2_values = grid_params["w"]

        # Parse config to extract parameters
        def parse_config(config_str):
            parts = config_str.split('_')
            swarm_size = int(parts[0].replace('swarm', ''))
            w = float(parts[1].replace('w', ''))
            return swarm_size, w

    else:  # GA
        param1_name, param2_name = "pop_size", "mut_sigma"
        param1_values = grid_params["pop_size"]
        param2_values = grid_params["mut_sigma"]

        # Parse config to extract parameters
        def parse_config(config_str):
            parts = config_str.split('_')
            pop_size = int(parts[0].replace('pop', ''))
            mut_sigma = float(parts[1].replace('sigma', ''))
            return pop_size, mut_sigma

    # Filter data for this algorithm
    algo_data = df[df['algo'] == algo].copy()

    # Parse configs
    algo_data['param1'], algo_data['param2'] = zip(*algo_data['config'].apply(parse_config))

    # Create 3x3 matrices
    n_param1 = len(param1_values)
    n_param2 = len(param2_values)

    success_matrix = np.zeros((n_param2, n_param1))
    final_f_matrix = np.zeros((n_param2, n_param1))
    evals_matrix = np.zeros((n_param2, n_param1))

    for i, p2_val in enumerate(param2_values):
        for j, p1_val in enumerate(param1_values):
            subset = algo_data[(algo_data['param1'] == p1_val) &
                              (algo_data['param2'] == p2_val)]

            if len(subset) > 0:
                # Success rate
                success_matrix[i, j] = subset['success'].mean()

                # Median final_best_f
                final_f_matrix[i, j] = subset['final_best_f'].median()

                # Median evals_to_target (successful runs only)
                successful = subset[subset['evals_to_target'] > 0]
                if len(successful) > 0:
                    evals_matrix[i, j] = successful['evals_to_target'].median()
                else:
                    evals_matrix[i, j] = np.nan

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Heatmap 1: Success rate
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(success_matrix, cmap='RdYlGn', aspect='auto',
                     vmin=0, vmax=1, origin='lower')
    ax1.set_xticks(range(n_param1))
    ax1.set_yticks(range(n_param2))
    ax1.set_xticklabels(param1_values)
    ax1.set_yticklabels(param2_values)
    ax1.set_xlabel(param1_name, fontweight='bold')
    ax1.set_ylabel(param2_name, fontweight='bold')
    ax1.set_title("Success Rate", fontweight='bold')

    # Add text annotations
    for i in range(n_param2):
        for j in range(n_param1):
            text = ax1.text(j, i, f'{success_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Heatmap 2: Median final_best_f
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(final_f_matrix, cmap='RdYlGn', aspect='auto', origin='lower')
    ax2.set_xticks(range(n_param1))
    ax2.set_yticks(range(n_param2))
    ax2.set_xticklabels(param1_values)
    ax2.set_yticklabels(param2_values)
    ax2.set_xlabel(param1_name, fontweight='bold')
    ax2.set_ylabel(param2_name, fontweight='bold')
    ax2.set_title("Median Final Best f", fontweight='bold')

    # Add text annotations
    for i in range(n_param2):
        for j in range(n_param1):
            text = ax2.text(j, i, f'{final_f_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Heatmap 3: Median evals_to_target (successful runs)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(evals_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower')
    ax3.set_xticks(range(n_param1))
    ax3.set_yticks(range(n_param2))
    ax3.set_xticklabels(param1_values)
    ax3.set_yticklabels(param2_values)
    ax3.set_xlabel(param1_name, fontweight='bold')
    ax3.set_ylabel(param2_name, fontweight='bold')
    ax3.set_title("Median Evals to Target (Success)", fontweight='bold')

    # Add text annotations
    for i in range(n_param2):
        for j in range(n_param1):
            if not np.isnan(evals_matrix[i, j]):
                text = ax3.text(j, i, f'{int(evals_matrix[i, j])}',
                              ha="center", va="center", color="black", fontsize=8)
            else:
                text = ax3.text(j, i, 'N/A',
                              ha="center", va="center", color="gray", fontsize=8)

    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(f"{algo} Hyperparameter Performance", fontsize=16, fontweight='bold', y=1.02)

    output_path = FIGURES_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {algo} heatmaps to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 4: BASIN COVERAGE BAR CHART
# ============================================================================

def plot_basin_coverage(df: pd.DataFrame, best_configs: Dict[str, List[str]],
                       output_file: str = "basin_coverage.png"):
    """
    Bar chart showing basin coverage for the best configuration of each algorithm.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all basin IDs
    all_basins = sorted(df['basin_id'].unique())
    n_basins = len(all_basins)

    # Width of bars
    width = 0.35
    x = np.arange(len(best_configs))

    # For each basin, collect counts
    basin_counts = {basin: [] for basin in all_basins}
    algo_labels = []

    for algo, configs in best_configs.items():
        best_config = configs[0]
        algo_labels.append(algo)

        # Filter data
        data = df[(df['algo'] == algo) & (df['config'] == best_config)]

        # Count runs in each basin
        for basin in all_basins:
            count = len(data[data['basin_id'] == basin])
            basin_counts[basin].append(count)

    # Plot stacked bars
    bottom = np.zeros(len(algo_labels))

    for basin in all_basins:
        counts = basin_counts[basin]
        ax.bar(x, counts, width=0.6, label=f'Basin {basin}', bottom=bottom, alpha=0.9)
        bottom += counts

    ax.set_xlabel("Algorithm", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Runs", fontsize=12, fontweight='bold')
    ax.set_title("Basin Coverage (Best Config per Algorithm)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_labels)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    output_path = FIGURES_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Basin coverage plot to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 5: TRAJECTORY PLOTS ON CONTOUR MAP
# ============================================================================

def create_contour_background(ax, bounds=(-15, 15), resolution=200):
    """Create a contour plot background of the Cross-in-Tray function."""
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = cross_in_tray(np.array([X[i, j], Y[i, j]]))

    # Plot contours
    levels = np.linspace(Z.min(), 0, 30)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.2, linewidths=0.5)

    # Mark global minima
    for minimum in KNOWN_MINIMA:
        ax.plot(minimum[0], minimum[1], 'r*', markersize=15,
               markeredgecolor='white', markeredgewidth=1)

    return contour


def select_representative_runs(raw_results: Dict[str, Dict[str, List[Dict]]],
                              algo: str, config_id: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Select 3 representative runs:
    1) Success fast: successful run with lowest evals_to_target
    2) Stuck-then-success: successful run with highest evals_to_target
    3) Fail: failed run (evals_to_target is None)

    Returns:
        (fast_run, slow_run, failed_run) - any can be None if not available
    """
    runs = raw_results[algo][config_id]

    # Separate successful and failed runs
    successful = [r for r in runs if r['evals_to_target'] is not None]
    failed = [r for r in runs if r['evals_to_target'] is None]

    fast_run = None
    slow_run = None
    failed_run = None

    if len(successful) > 0:
        # Fast: minimum evals_to_target
        fast_run = min(successful, key=lambda r: r['evals_to_target'])

        # Slow: maximum evals_to_target
        if len(successful) > 1:
            slow_run = max(successful, key=lambda r: r['evals_to_target'])

    if len(failed) > 0:
        # Pick a random failed run
        failed_run = failed[0]

    return fast_run, slow_run, failed_run


def plot_trajectories(raw_results: Dict[str, Dict[str, List[Dict]]],
                     best_configs: Dict[str, List[str]],
                     output_file: str = "trajectories.png"):
    """
    Plot trajectory plots on contour map for 3 representative runs per algorithm.
    """
    n_algos = len(best_configs)
    fig = plt.figure(figsize=(18, 6 * n_algos))
    gs = GridSpec(n_algos, 3, figure=fig, hspace=0.3, wspace=0.3)

    algo_idx = 0
    for algo, configs in best_configs.items():
        best_config = configs[0]

        if best_config not in raw_results[algo]:
            continue

        # Select representative runs
        fast_run, slow_run, failed_run = select_representative_runs(
            raw_results, algo, best_config
        )

        runs = [
            (fast_run, "Success (Fast)", 'green'),
            (slow_run, "Success (Slow)", 'orange'),
            (failed_run, "Failed", 'red')
        ]

        for col_idx, (run, title_suffix, color) in enumerate(runs):
            ax = fig.add_subplot(gs[algo_idx, col_idx])

            if run is None:
                ax.text(0.5, 0.5, 'No data available',
                       ha='center', va='center', fontsize=12)
                ax.set_xlim([-15, 15])
                ax.set_ylim([-15, 15])
                ax.set_title(f"{algo} - {title_suffix}", fontweight='bold')
                continue

            # Create contour background
            create_contour_background(ax)

            # Plot trajectory from best_x_by_iter
            trajectory = np.array([np.array(x) for x in run['best_x_by_iter']])

            if len(trajectory) > 0:
                # Plot trajectory line
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=color, linewidth=2, alpha=0.8, marker='o',
                       markersize=3, label='Best solution path')

                # Mark start
                ax.plot(trajectory[0, 0], trajectory[0, 1],
                       'go', markersize=10, markeredgecolor='white',
                       markeredgewidth=2, label='Start')

                # Mark end
                ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                       'bs', markersize=10, markeredgecolor='white',
                       markeredgewidth=2, label='End')

            ax.set_xlim([-15, 15])
            ax.set_ylim([-15, 15])
            ax.set_xlabel('x₁', fontweight='bold')
            ax.set_ylabel('x₂', fontweight='bold')

            # Add title with run info
            evals_info = f"evals={run['evals_to_target']}" if run['evals_to_target'] else f"f={run['final_best_f']:.4f}"
            ax.set_title(f"{algo} - {title_suffix}\n{evals_info} (seed={run['seed']})",
                        fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
            ax.grid(True, alpha=0.3, linestyle='--')

        algo_idx += 1

    plt.suptitle("Representative Trajectory Plots", fontsize=18, fontweight='bold', y=0.995)

    output_path = FIGURES_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Trajectory plots to {output_path}")
    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to generate all figures."""
    print("=" * 80)
    print("ANALYSIS: GENERATING FIGURES")
    print("=" * 80)
    print()

    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    raw_results = load_all_raw_results()
    df = load_summary_csv()

    if not raw_results or df.empty:
        print("Error: No results found! Run experiments first.")
        return

    print(f"Loaded results for {len(raw_results)} algorithms")
    print(f"Summary CSV contains {len(df)} runs")
    print()

    # Identify best configurations
    print("Identifying best configurations...")
    best_configs = get_best_configs(df, n_configs=2)
    for algo, configs in best_configs.items():
        print(f"  {algo}: {configs}")
    print()

    # Define grid parameters for heatmaps
    PSO_GRID = {
        "swarm_size": [10, 20, 40],
        "w": [0.4, 0.7, 0.9]
    }

    GA_GRID = {
        "pop_size": [20, 50, 100],
        "mut_sigma": [0.1, 0.5, 1.0]
    }

    # Generate figures
    print("Generating figures...")
    print()

    print("1. Convergence plot...")
    plot_convergence(raw_results, best_configs)

    print("2. ECDF plot...")
    plot_ecdf(df, best_configs)

    print("3. Heatmaps...")
    if 'PSO' in best_configs:
        plot_heatmaps(df, 'PSO', PSO_GRID, "heatmap_pso.png")
    if 'GA' in best_configs:
        plot_heatmaps(df, 'GA', GA_GRID, "heatmap_ga.png")

    print("4. Basin coverage...")
    plot_basin_coverage(df, best_configs)

    print("5. Trajectory plots...")
    plot_trajectories(raw_results, best_configs)

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
