"""
Logging utilities for optimization experiments.

Provides RunLog dataclass for tracking optimization runs and JSON serialization.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class RunLog:
    """
    Data structure to store a single optimization run.

    Attributes:
        seed: Random seed used for this run
        algo_name: Name of the optimization algorithm
        config: Dictionary of algorithm configuration parameters
        best_f_by_eval: List of best function values seen at each evaluation
        best_x_by_iter: List of best solution vectors at each iteration/generation
        pop_positions_by_iter: Optional list of population positions per iteration
        evals_to_target: Number of evaluations to reach target (None if not reached)
        final_best_x: Final best solution found
        final_best_f: Final best function value
        basin_id: ID of the basin of attraction (nearest known minimum)
    """
    seed: int
    algo_name: str
    config: Dict[str, Any]
    best_f_by_eval: List[float]
    best_x_by_iter: List[np.ndarray]
    pop_positions_by_iter: Optional[List[np.ndarray]]
    evals_to_target: Optional[int]
    final_best_x: np.ndarray
    final_best_f: float
    basin_id: int


def _numpy_to_list(obj: Any) -> Any:
    """
    Recursively convert numpy arrays to lists for JSON serialization.

    Args:
        obj: Object to convert (can be nested structures)

    Returns:
        Object with numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_numpy_to_list(item) for item in obj)
    else:
        return obj


def save_runlog_json(path: str, runlog: RunLog) -> None:
    """
    Save a RunLog to a JSON file with proper numpy array conversion.

    Args:
        path: File path to save the JSON log
        runlog: RunLog instance to save

    Example:
        >>> log = RunLog(seed=42, algo_name="GA", ...)
        >>> save_runlog_json("results/run_001.json", log)
    """
    # Convert dataclass to dictionary
    log_dict = asdict(runlog)

    # Convert all numpy arrays to lists recursively
    log_dict = _numpy_to_list(log_dict)

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Write to JSON with pretty formatting
    with open(path, 'w') as f:
        json.dump(log_dict, f, indent=2)


def load_runlog_json(path: str) -> Dict[str, Any]:
    """
    Load a RunLog from a JSON file.

    Note: Returns a dictionary rather than RunLog instance since numpy arrays
    are stored as lists in JSON. Convert back to numpy arrays as needed.

    Args:
        path: File path to load the JSON log from

    Returns:
        Dictionary containing the run log data

    Example:
        >>> log_dict = load_runlog_json("results/run_001.json")
        >>> final_x = np.array(log_dict["final_best_x"])
    """
    with open(path, 'r') as f:
        return json.load(f)
