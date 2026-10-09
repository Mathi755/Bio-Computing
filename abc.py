"""
abc.py
------
Artificial Bee Colony (ABC) — Nature-Inspired Metaheuristic
Source equations from: bio 1.pdf (canonical) & bio 2.pdf (1D workout)

Key equations:
  v_ij   = x_ij + phi * (x_ij - x_kj)        (candidate solution)
  prob_i = fitness_i / sum_k(fitness_k)        (selection probability)

Three bee types:
  - Employed Bees  : exploit current food sources
  - Onlooker Bees  : select sources probabilistically
  - Scout Bees     : abandon exhausted sources, explore randomly

Run:
  python abc.py
"""

import numpy as np
from common import sphere, clip_position


def artificial_bee_colony(
    objective=sphere,
    n_food_sources=20,
    dimensions=1,
    n_iterations=200,
    lower_bound=-10.0,
    upper_bound=10.0,
    limit=50,
    seed=42,
):
    """
    Artificial Bee Colony using PDF 1 terminology.

    Parameters:
        n_food_sources : number of food sources (= number of employed bees)
        limit          : abandonment threshold for scout bees

    Returns:
        best_solution, best_objective, convergence_curve
    """
    rng = np.random.default_rng(seed)

    x_ij = rng.uniform(lower_bound, upper_bound, (n_food_sources, dimensions))
    objective_values = np.asarray([objective(x) for x in x_ij])
    trials = np.zeros(n_food_sources, dtype=int)

    best_idx = np.argmin(objective_values)
    best_solution = x_ij[best_idx].copy()
    best_objective = objective_values[best_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(n_iterations):
        # ---------------- Employed Bees ----------------
        for i in range(n_food_sources):
            candidates = np.arange(n_food_sources)
            candidates = candidates[candidates != i]
            k = int(rng.choice(candidates))
            j = int(rng.integers(dimensions))

            phi = rng.uniform(-1.0, 1.0)

            v_ij = x_ij[i].copy()
            v_ij[j] = x_ij[i, j] + phi * (x_ij[i, j] - x_ij[k, j])
            v_ij = clip_position(v_ij, lower_bound, upper_bound)

            v_objective = objective(v_ij)

            if v_objective < objective_values[i]:
                x_ij[i] = v_ij
                objective_values[i] = v_objective
                trials[i] = 0
            else:
                trials[i] += 1

        # ---------------- Onlooker Bees ----------------
        quality = 1.0 / (1.0 + objective_values)
        prob_i = quality / np.sum(quality)

        for _ in range(n_food_sources):
            i = int(rng.choice(n_food_sources, p=prob_i))

            candidates = np.arange(n_food_sources)
            candidates = candidates[candidates != i]
            k = int(rng.choice(candidates))
            j = int(rng.integers(dimensions))

            phi = rng.uniform(-1.0, 1.0)

            v_ij = x_ij[i].copy()
            v_ij[j] = x_ij[i, j] + phi * (x_ij[i, j] - x_ij[k, j])
            v_ij = clip_position(v_ij, lower_bound, upper_bound)

            v_objective = objective(v_ij)

            if v_objective < objective_values[i]:
                x_ij[i] = v_ij
                objective_values[i] = v_objective
                trials[i] = 0
            else:
                trials[i] += 1

        # ---------------- Scout Bees ----------------
        exhausted = np.flatnonzero(trials >= limit)
        for i in exhausted:
            x_ij[i] = rng.uniform(lower_bound, upper_bound, dimensions)
            objective_values[i] = objective(x_ij[i])
            trials[i] = 0

        # ---------------- Track Best ----------------
        best_idx = np.argmin(objective_values)
        if objective_values[best_idx] < best_objective:
            best_solution = x_ij[best_idx].copy()
            best_objective = objective_values[best_idx]

        convergence_curve[t] = best_objective

    return best_solution, float(best_objective), convergence_curve


if __name__ == "__main__":
    print("=" * 55)
    print("ARTIFICIAL BEE COLONY (ABC)")
    print("Objective : Minimize f(x) = sum(x^2)")
    print("Bounds    : [-10, 10]  |  Dimensions: 1")
    print("=" * 55)

    best_x, best_f, curve = artificial_bee_colony()
    print(f"Best position : {best_x}")
    print(f"Best fitness  : {best_f:.10f}")
    print(f"(Global optimum: x=0, f=0)")
