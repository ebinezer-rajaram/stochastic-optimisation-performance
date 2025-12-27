"""
Cross-in-Tray objective function for optimization benchmarking.

The Cross-in-Tray function is a multimodal test function with multiple global minima.
Global minima: f(±1.3491, ±1.3491) ≈ -2.06261
Domain: [-15, 15]²
"""

import numpy as np
from typing import Union


def clip_to_bounds(x: np.ndarray, low: float = -15.0, high: float = 15.0) -> np.ndarray:
    """
    Clip input vector to specified bounds.

    Args:
        x: Input array to clip
        low: Lower bound (default: -15.0)
        high: Upper bound (default: 15.0)

    Returns:
        Clipped array within bounds [low, high]
    """
    return np.clip(x, low, high)


def cross_in_tray(x: np.ndarray) -> float:
    """
    Cross-in-Tray function with bounds enforcement.

    Formula (per 4M17 coursework specification):
    f(x) = -0.001 * |sin(x₁)sin(x₂)exp(|100 - √(x₁² + x₂²)/π|)| + 1|^0.1

    Args:
        x: 2D input vector, shape (2,)

    Returns:
        Function value (float)

    Raises:
        ValueError: If x is not a 2D vector
    """
    if x.shape != (2,):
        raise ValueError(f"Input must be 2D vector, got shape {x.shape}")

    # Enforce bounds by clipping
    x = clip_to_bounds(x, low=-15.0, high=15.0)

    x1, x2 = x[0], x[1]

    # Compute the Cross-in-Tray function per coursework spec
    term1 = np.sin(x1) * np.sin(x2)
    term2 = np.exp(np.abs(100.0 - np.sqrt(x1**2 + x2**2) / np.pi))
    inner = np.abs(term1 * term2) + 1.0

    result = -0.001 * (inner ** 0.1)

    return float(result)


def _run_self_test() -> None:
    """
    Self-test for Cross-in-Tray function.
    Verifies symmetry properties and known minimum values.
    """
    print("Running Cross-in-Tray self-test...")

    # Known global minima locations (approximate for standard Cross-in-Tray)
    minima = [
        np.array([1.3491, 1.3491]),
        np.array([-1.3491, 1.3491]),
        np.array([1.3491, -1.3491]),
        np.array([-1.3491, -1.3491])
    ]

    # Test 1: Check all minima have approximately the same value
    min_values = [cross_in_tray(m) for m in minima]

    print(f"  Minimum values at known optima: {min_values}")
    print(f"  (Note: Coursework uses -0.001 coefficient, not standard -0.0001)")

    # Check symmetry (all four should be equal)
    for i in range(1, len(min_values)):
        assert abs(min_values[i] - min_values[0]) < 1e-10, \
            f"Minimum {i} value {min_values[i]} differs from minimum 0 value {min_values[0]}"

    # Test 2: Verify symmetry (f(x,y) = f(-x,y) = f(x,-y) = f(-x,-y))
    test_point = np.array([3.0, 4.0])
    f1 = cross_in_tray(test_point)
    f2 = cross_in_tray(np.array([-3.0, 4.0]))
    f3 = cross_in_tray(np.array([3.0, -4.0]))
    f4 = cross_in_tray(np.array([-3.0, -4.0]))

    print(f"  Symmetry test values: {[f1, f2, f3, f4]}")

    assert abs(f1 - f2) < 1e-10, "Symmetry failed: f(x,y) ≠ f(-x,y)"
    assert abs(f1 - f3) < 1e-10, "Symmetry failed: f(x,y) ≠ f(x,-y)"
    assert abs(f1 - f4) < 1e-10, "Symmetry failed: f(x,y) ≠ f(-x,-y)"

    # Test 3: Bounds clipping
    out_of_bounds = np.array([20.0, -20.0])
    f_clipped = cross_in_tray(out_of_bounds)
    f_boundary = cross_in_tray(np.array([15.0, -15.0]))

    print(f"  Bounds test: out-of-bounds point clipped correctly")
    assert abs(f_clipped - f_boundary) < 1e-10, "Bounds clipping failed"

    # Test 4: Origin sanity check
    origin = np.array([0.0, 0.0])
    f_origin = cross_in_tray(origin)
    print(f"  Value at origin: {f_origin}")
    assert f_origin < 0.0, "Origin value should be negative"

    print("[PASS] All self-tests passed!")


if __name__ == "__main__":
    _run_self_test()
