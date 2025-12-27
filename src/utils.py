"""
Utility functions for optimization experiments.

Provides random seeding, bounds clipping, distance computation, and basin assignment.
"""

import numpy as np
from typing import List


# Global random generator instance
_rng: np.random.Generator = np.random.default_rng()


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for reproducibility.

    Args:
        seed: Integer seed value for numpy's random generator
    """
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """
    Get the global random number generator.

    Returns:
        The global numpy Generator instance
    """
    return _rng


def clip_bounds(x: np.ndarray, low: float = -15.0, high: float = 15.0) -> np.ndarray:
    """
    Clip array elements to specified bounds.

    Args:
        x: Input array to clip
        low: Lower bound (default: -15.0)
        high: Upper bound (default: 15.0)

    Returns:
        Array with values clipped to [low, high]
    """
    return np.clip(x, low, high)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        a: First point (numpy array)
        b: Second point (numpy array)

    Returns:
        Euclidean distance as a float
    """
    return float(np.linalg.norm(a - b))


def assign_basin(x: np.ndarray, minima: List[np.ndarray]) -> int:
    """
    Assign a point to the nearest basin of attraction.

    Args:
        x: Point to assign, shape (d,)
        minima: List of known minimum locations, each shape (d,)

    Returns:
        Index of the nearest minimum (basin ID)

    Raises:
        ValueError: If minima list is empty
    """
    if len(minima) == 0:
        raise ValueError("Minima list cannot be empty")

    distances = [distance(x, m) for m in minima]
    return int(np.argmin(distances))
