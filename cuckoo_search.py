"""
cuckoo_search.py
----------------
Cuckoo Search (CS) — Nature-Inspired Metaheuristic
Source equations from: bio 1.pdf (canonical) & bio 2.pdf (1D workout)

Key equations:
  x_i^t = x_i^(t-1) + alpha * Levy(lambda)     (Levy flight search)
  x_new  = x_i^(t-1) + r * (x_j - x_k)         (nest abandonment)

Levy flight implemented via Mantegna's algorithm.

Run:
  python cuckoo_search.py
"""

import math
import numpy as np
from common import sphere, clip_position


def levy_flight(rng, size, lambda_=1.5):
    """
    Mantegna-style Levy(lambda) random step.
    Used to realize the PDF's heavy-tailed Levy(lambda) term.
    """
    sigma_u = (
        math.gamma(1 + lambda_)
        * np.sin(np.pi * lambda_ / 2)
        / (
            math.gamma((1 + lambda_) / 2)
            * lambda_
            * 2 ** ((lambda_ - 1) / 2)
        )
    ) ** (1 / lambda_)

    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1, size=size)

    return u / (np.abs(v) ** (1 / lambda_) + 1e-12)


def cuckoo_search(
    objective=sphere,
    n_nests=25,
    dimensions=1,
    n_iterations=200,
    lower_bound=-10.0,
    upper_bound=10.0,
    alpha=0.01,
    p_a=0.25,
    lambda_=1.5,
    seed=42,
):
    """
    Cuckoo Search using PDF 1 equations.

    Parameters:
        n_nests     : number of host nests (population size)
        alpha       : step size scaling factor
        p_a         : fraction of nests abandoned per iteration
        lambda_     : Levy exponent (1 < lambda_ <= 3)

    Returns:
        best_solution, best_fitness, convergence_curve
    """
    rng = np.random.default_rng(seed)

    x_i = rng.uniform(lower_bound, upper_bound, (n_nests, dimensions))
    fitness = np.asarray([objective(x) for x in x_i])

    best_idx = np.argmin(fitness)
    best_solution = x_i[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(n_iterations):
        # ---------------- Levy Flight Search ----------------
        i = int(rng.integers(n_nests))
        step = levy_flight(rng, dimensions, lambda_)
        x_new = x_i[i] + alpha * step * (x_i[i] - best_solution)
        x_new = clip_position(x_new, lower_bound, upper_bound)

        new_fitness = objective(x_new)

        j = int(rng.integers(n_nests))
        if new_fitness < fitness[j]:
            x_i[j] = x_new
            fitness[j] = new_fitness

        # ---------------- Nest Abandonment ----------------
        abandon = rng.random(n_nests) < p_a

        for i in np.flatnonzero(abandon):
            j, k = rng.choice(n_nests, size=2, replace=False)
            r = rng.random()

            x_new = x_i[i] + r * (x_i[j] - x_i[k])
            x_new = clip_position(x_new, lower_bound, upper_bound)
            new_fitness = objective(x_new)

            if new_fitness < fitness[i]:
                x_i[i] = x_new
                fitness[i] = new_fitness

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = x_i[best_idx].copy()
            best_fitness = fitness[best_idx]

        convergence_curve[t] = best_fitness

    return best_solution, float(best_fitness), convergence_curve


if __name__ == "__main__":
    print("=" * 55)
    print("CUCKOO SEARCH (CS)")
    print("Objective : Minimize f(x) = sum(x^2)")
    print("Bounds    : [-10, 10]  |  Dimensions: 1")
    print("=" * 55)

    best_x, best_f, curve = cuckoo_search()
    print(f"Best position : {best_x}")
    print(f"Best fitness  : {best_f:.10f}")
    print(f"(Global optimum: x=0, f=0)")
