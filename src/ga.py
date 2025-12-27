"""
Genetic Algorithm (GA) implementation.

Implements a real-valued Genetic Algorithm for continuous optimization with:
- Tournament selection
- Blend/arithmetic crossover
- Gaussian mutation
- Elitism
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from src.logging_utils import RunLog
from src.utils import clip_bounds, assign_basin


def ga_optimize(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    seed: int,
    budget: int,
    pop_size: int,
    cx_prob: float,
    mut_prob: float,
    mut_sigma: float,
    tournament_k: int = 2,
    elitism: int = 1,
    max_gens: Optional[int] = None,
    target_f: Optional[float] = None,
    minima: Optional[List[np.ndarray]] = None
) -> RunLog:
    """
    Real-valued Genetic Algorithm with tournament selection.

    Args:
        objective_fn: Function to minimize, takes np.ndarray(2,) returns float
        bounds: Tuple of (low, high) bounds for the search space
        seed: Random seed for reproducibility
        budget: Maximum number of objective function evaluations
        pop_size: Population size
        cx_prob: Crossover probability (probability of crossover between parents)
        mut_prob: Mutation probability (per individual or per gene)
        mut_sigma: Standard deviation for Gaussian mutation
        tournament_k: Tournament size for selection (default: 2)
        elitism: Number of elite individuals to preserve (default: 1)
        max_gens: Optional maximum generations (defaults to budget//pop_size)
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

    # Calculate max generations if not provided
    if max_gens is None:
        max_gens = budget // pop_size

    # Initialize population uniformly in bounds
    population = rng.uniform(low, high, size=(pop_size, dim))

    # Tracking variables
    best_f_by_eval = []
    best_x_by_iter = []
    pop_positions_by_iter = []
    evals_to_target = None
    eval_count = 0

    # Global best tracking
    global_best_x = None
    global_best_f = np.inf

    # Evaluate initial population
    fitness = np.full(pop_size, np.inf)
    for i in range(pop_size):
        if eval_count >= budget:
            break
        fitness[i] = objective_fn(population[i])
        eval_count += 1

        # Update global best
        if fitness[i] < global_best_f:
            global_best_f = fitness[i]
            global_best_x = population[i].copy()

        # Track best-so-far after each evaluation
        best_f_by_eval.append(global_best_f)

        # Check if target reached
        if target_f is not None and evals_to_target is None:
            if global_best_f <= target_f:
                evals_to_target = eval_count

    # Track initial best
    if global_best_x is not None:
        best_x_by_iter.append(global_best_x.copy())

    # Store initial population positions
    pop_positions_by_iter.append(population.copy())

    # Main GA loop
    for generation in range(1, max_gens):
        if eval_count >= budget:
            break

        # Sort population by fitness for elitism
        sorted_indices = np.argsort(fitness)
        elite_indices = sorted_indices[:elitism]
        elites = population[elite_indices].copy()

        # Create new population
        new_population = []

        # Preserve elites
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate offspring to fill remaining population slots
        while len(new_population) < pop_size:
            # Tournament selection for parent 1
            parent1 = tournament_selection(population, fitness, tournament_k, rng)

            # Tournament selection for parent 2
            parent2 = tournament_selection(population, fitness, tournament_k, rng)

            # Crossover
            if rng.random() < cx_prob:
                # Blend crossover (arithmetic crossover)
                alpha = rng.random()
                offspring1 = alpha * parent1 + (1 - alpha) * parent2
                offspring2 = (1 - alpha) * parent1 + alpha * parent2
            else:
                # No crossover, copy parents
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()

            # Mutation - apply to each offspring independently
            if rng.random() < mut_prob:
                offspring1 = mutate_gaussian(offspring1, mut_sigma, low, high, rng)

            if rng.random() < mut_prob:
                offspring2 = mutate_gaussian(offspring2, mut_sigma, low, high, rng)

            # Add offspring to new population
            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)

        # Convert to array (trim if we generated one extra)
        population = np.array(new_population[:pop_size])

        # Evaluate new population
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if eval_count >= budget:
                break

            fitness[i] = objective_fn(population[i])
            eval_count += 1

            # Update global best
            if fitness[i] < global_best_f:
                global_best_f = fitness[i]
                global_best_x = population[i].copy()

            # Track best-so-far after each evaluation
            best_f_by_eval.append(global_best_f)

            # Check if target reached
            if target_f is not None and evals_to_target is None:
                if global_best_f <= target_f:
                    evals_to_target = eval_count

        # Break if budget exhausted
        if eval_count >= budget:
            break

        # Track best position for this generation
        if global_best_x is not None:
            best_x_by_iter.append(global_best_x.copy())

        # Store population positions for plotting
        pop_positions_by_iter.append(population.copy())

    # Assign basin if minima provided
    basin_id = -1
    if minima is not None and global_best_x is not None:
        basin_id = assign_basin(global_best_x, minima)

    # Handle case where no evaluations were performed
    if global_best_x is None:
        global_best_x = np.zeros(dim)
        global_best_f = np.inf

    # Create configuration dictionary
    config = {
        "pop_size": pop_size,
        "cx_prob": cx_prob,
        "mut_prob": mut_prob,
        "mut_sigma": mut_sigma,
        "tournament_k": tournament_k,
        "elitism": elitism,
        "budget": budget,
        "max_gens": max_gens,
        "bounds": bounds
    }

    # Return RunLog
    return RunLog(
        seed=seed,
        algo_name="GA",
        config=config,
        best_f_by_eval=best_f_by_eval,
        best_x_by_iter=best_x_by_iter,
        pop_positions_by_iter=pop_positions_by_iter,
        evals_to_target=evals_to_target,
        final_best_x=global_best_x,
        final_best_f=global_best_f,
        basin_id=basin_id
    )


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_k: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Select an individual using tournament selection.

    Args:
        population: Population array of shape (pop_size, dim)
        fitness: Fitness values of shape (pop_size,)
        tournament_k: Tournament size
        rng: Random number generator

    Returns:
        Selected individual (copy)
    """
    # Randomly select tournament_k individuals
    tournament_indices = rng.choice(len(population), size=tournament_k, replace=False)

    # Find the best (minimum fitness) in the tournament
    tournament_fitness = fitness[tournament_indices]
    winner_idx = tournament_indices[np.argmin(tournament_fitness)]

    return population[winner_idx].copy()


def mutate_gaussian(
    individual: np.ndarray,
    sigma: float,
    low: float,
    high: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply Gaussian mutation to an individual and clip to bounds.

    Args:
        individual: Individual to mutate, shape (dim,)
        sigma: Standard deviation of Gaussian noise
        low: Lower bound
        high: Upper bound
        rng: Random number generator

    Returns:
        Mutated individual (clipped to bounds)
    """
    # Add Gaussian noise to each gene
    noise = rng.normal(0, sigma, size=individual.shape)
    mutated = individual + noise

    # Clip to bounds
    mutated = clip_bounds(mutated, low, high)

    return mutated


if __name__ == "__main__":
    """Simple test of GA implementation."""
    from src.objective import cross_in_tray

    print("Testing GA implementation...")
    print("-" * 50)

    # Test parameters
    seed = 42
    budget = 1000
    pop_size = 50
    cx_prob = 0.8
    mut_prob = 0.2
    mut_sigma = 1.0
    tournament_k = 3
    elitism = 2
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

    # Run GA
    result = ga_optimize(
        objective_fn=cross_in_tray,
        bounds=bounds,
        seed=seed,
        budget=budget,
        pop_size=pop_size,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        mut_sigma=mut_sigma,
        tournament_k=tournament_k,
        elitism=elitism,
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
    print(f"  Generations completed: {len(result.best_x_by_iter)}")

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

    # Test determinism
    print("\n" + "-" * 50)
    print("Testing determinism (running with same seed)...")
    result2 = ga_optimize(
        objective_fn=cross_in_tray,
        bounds=bounds,
        seed=seed,
        budget=budget,
        pop_size=pop_size,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        mut_sigma=mut_sigma,
        tournament_k=tournament_k,
        elitism=elitism,
        target_f=target_f,
        minima=minima
    )

    # Check if results are identical
    same_final_f = np.abs(result.final_best_f - result2.final_best_f) < 1e-10
    same_final_x = np.allclose(result.final_best_x, result2.final_best_x, atol=1e-10)
    same_convergence = all(
        np.abs(a - b) < 1e-10
        for a, b in zip(result.best_f_by_eval, result2.best_f_by_eval)
    )

    print(f"  Same final f: {same_final_f}")
    print(f"  Same final x: {same_final_x}")
    print(f"  Same convergence history: {same_convergence}")

    if same_final_f and same_final_x and same_convergence:
        print("  [PASS] Determinism verified!")
    else:
        print("  [WARN] Results differ between runs with same seed")

    print("-" * 50)
    print("Test completed successfully!")
