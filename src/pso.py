"""
Particle Swarm Optimization (PSO) implementation.

Implements the global-best (gbest) PSO algorithm for continuous optimization.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from src.logging_utils import RunLog
from src.utils import clip_bounds, assign_basin


def pso_optimize(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    seed: int,
    budget: int,
    swarm_size: int,
    w: float,
    c1: float,
    c2: float,
    max_iters: Optional[int] = None,
    target_f: Optional[float] = None,
    minima: Optional[List[np.ndarray]] = None
) -> RunLog:
    """
    Particle Swarm Optimization (gbest variant).

    Args:
        objective_fn: Function to minimize, takes np.ndarray(2,) returns float
        bounds: Tuple of (low, high) bounds for the search space
        seed: Random seed for reproducibility
        budget: Maximum number of objective function evaluations
        swarm_size: Number of particles in the swarm
        w: Inertia weight
        c1: Cognitive coefficient (personal best attraction)
        c2: Social coefficient (global best attraction)
        max_iters: Optional maximum iterations (defaults to budget//swarm_size)
        target_f: Optional target function value for early stopping
        minima: Optional list of known minima for basin assignment

    Returns:
        RunLog containing optimization results and tracking data
    """
    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Problem dimension (assumed 2D for Cross-in-Tray)
    dim = 2
    low, high = bounds

    # Calculate max iterations if not provided
    if max_iters is None:
        max_iters = budget // swarm_size

    # Initialize particles uniformly in bounds
    positions = rng.uniform(low, high, size=(swarm_size, dim))

    # Initialize velocities with small random values
    # Use 10% of search space range as velocity initialization scale
    v_init_scale = 0.1 * (high - low)
    velocities = rng.uniform(-v_init_scale, v_init_scale, size=(swarm_size, dim))

    # Initialize personal bests
    personal_best_positions = positions.copy()
    personal_best_values = np.full(swarm_size, np.inf)

    # Initialize global best
    global_best_position = None
    global_best_value = np.inf

    # Tracking variables
    best_f_by_eval = []
    best_x_by_iter = []
    pop_positions_by_iter = []
    evals_to_target = None
    eval_count = 0

    # Main PSO loop
    for iteration in range(max_iters):
        # Evaluate all particles in the swarm
        for i in range(swarm_size):
            # Check budget
            if eval_count >= budget:
                break

            # Evaluate objective function
            f_value = objective_fn(positions[i])
            eval_count += 1

            # Update personal best
            if f_value < personal_best_values[i]:
                personal_best_values[i] = f_value
                personal_best_positions[i] = positions[i].copy()

            # Update global best
            if f_value < global_best_value:
                global_best_value = f_value
                global_best_position = positions[i].copy()

            # Track best-so-far after each evaluation
            best_f_by_eval.append(global_best_value)

            # Check if target reached (only record first time)
            if target_f is not None and evals_to_target is None:
                if global_best_value <= target_f:
                    evals_to_target = eval_count

        # Break if budget exhausted
        if eval_count >= budget:
            # Track best position even if we hit budget mid-iteration
            if global_best_position is not None:
                best_x_by_iter.append(global_best_position.copy())
            break

        # Track best position for this iteration (always append)
        if global_best_position is not None:
            best_x_by_iter.append(global_best_position.copy())

        # Store population positions for plotting
        pop_positions_by_iter.append(positions.copy())

        # Update velocities and positions for all particles
        for i in range(swarm_size):
            # Generate random coefficients
            r1 = rng.uniform(0, 1, size=dim)
            r2 = rng.uniform(0, 1, size=dim)

            # Velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
            social = c2 * r2 * (global_best_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Position update: x = x + v
            positions[i] = positions[i] + velocities[i]

            # Clip positions to bounds
            positions[i] = clip_bounds(positions[i], low, high)

    # Note: best_x_by_iter is now tracked every iteration, so no need for final fallback

    # Assign basin if minima provided
    basin_id = -1
    if minima is not None and global_best_position is not None:
        basin_id = assign_basin(global_best_position, minima)

    # Handle case where no evaluations were performed
    if global_best_position is None:
        global_best_position = np.zeros(dim)
        global_best_value = np.inf

    # Create configuration dictionary
    config = {
        "swarm_size": swarm_size,
        "w": w,
        "c1": c1,
        "c2": c2,
        "budget": budget,
        "max_iters": max_iters,
        "bounds": bounds
    }

    # Return RunLog
    return RunLog(
        seed=seed,
        algo_name="PSO",
        config=config,
        best_f_by_eval=best_f_by_eval,
        best_x_by_iter=best_x_by_iter,
        pop_positions_by_iter=pop_positions_by_iter,
        evals_to_target=evals_to_target,
        final_best_x=global_best_position,
        final_best_f=global_best_value,
        basin_id=basin_id
    )


if __name__ == "__main__":
    """Simple test of PSO implementation."""
    from src.objective import cross_in_tray

    print("Testing PSO implementation...")
    print("-" * 50)

    # Test parameters
    seed = 42
    budget = 1000
    swarm_size = 20
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    bounds = (-15.0, 15.0)

    # Known minima for Cross-in-Tray
    minima = [
        np.array([1.3491, 1.3491]),
        np.array([-1.3491, 1.3491]),
        np.array([1.3491, -1.3491]),
        np.array([-1.3491, -1.3491])
    ]

    # Target value: f_star + eps where f_star is the known optimum
    # With -0.001 coefficient, f_star ≈ -20.6261 (10x standard value)
    f_star = -20.6261
    eps = 0.01
    target_f = f_star + eps  # -20.6161

    # Run PSO
    result = pso_optimize(
        objective_fn=cross_in_tray,
        bounds=bounds,
        seed=seed,
        budget=budget,
        swarm_size=swarm_size,
        w=w,
        c1=c1,
        c2=c2,
        target_f=target_f,
        minima=minima
    )

    # Print results
    print(f"Algorithm: {result.algo_name}")
    print(f"Seed: {result.seed}")
    print(f"Config: {result.config}")
    print(f"\nFinal Results:")
    print(f"  Best f: {result.final_best_f:.6f}")
    print(f"  Best x: {result.final_best_x}")
    print(f"  Basin ID: {result.basin_id}")
    print(f"  Total evaluations: {len(result.best_f_by_eval)}")
    print(f"  Iterations completed: {len(result.best_x_by_iter)}")

    if result.evals_to_target is not None:
        print(f"  Evals to target ({target_f}): {result.evals_to_target}")
    else:
        print(f"  Target ({target_f}) not reached")

    print(f"\nConvergence (first 10 evals):")
    for i in range(min(10, len(result.best_f_by_eval))):
        print(f"  Eval {i+1}: {result.best_f_by_eval[i]:.6f}")

    print(f"\nLast 5 evaluations:")
    for i in range(max(0, len(result.best_f_by_eval) - 5), len(result.best_f_by_eval)):
        print(f"  Eval {i+1}: {result.best_f_by_eval[i]:.6f}")

    print("-" * 50)
    print("Test completed successfully!")
