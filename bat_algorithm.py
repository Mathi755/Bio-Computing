"""
bat_algorithm.py
----------------
Bat Algorithm (BA) — Nature-Inspired Metaheuristic
Source equations from: bio 1.pdf (canonical) & bio 2.pdf (1D workout)

Key equations:
  f_i  = f_min + (f_max - f_min) * beta
  v_i^t = v_i^(t-1) + (x_i^(t-1) - x_*) * f_i
  x_i^t = x_i^(t-1) + v_i^t
  x_new = x_* + epsilon * A_avg^t
  A_i^t = alpha * A_i^(t-1)
  r_i^t = r_i^(t-1) * [1 - exp(-gamma*t)]

Run:
  python bat_algorithm.py
"""

import numpy as np
from common import sphere, clip_position


def bat_algorithm(
    objective=sphere,
    n_bats=30,
    dimensions=1,
    n_iterations=200,
    lower_bound=-10.0,
    upper_bound=10.0,
    f_min=0.0,
    f_max=2.0,
    A_initial=0.5,
    r_initial=0.5,
    alpha=0.9,
    gamma=0.9,
    seed=42,
):
    """
    Bat Algorithm using the terminology/equations in PDF 1.

    Returns:
        best_position, best_fitness, convergence_curve
    """
    rng = np.random.default_rng(seed)

    x_i = rng.uniform(lower_bound, upper_bound, size=(n_bats, dimensions))
    v_i = np.zeros((n_bats, dimensions))
    A_i = np.full(n_bats, A_initial)
    r_i = np.full(n_bats, r_initial)

    fitness = np.asarray([objective(x) for x in x_i], dtype=float)
    best_idx = np.argmin(fitness)
    x_star = x_i[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(1, n_iterations + 1):
        # Frequency update
        beta = rng.random(n_bats)
        f_i = f_min + (f_max - f_min) * beta

        # Velocity and position updates
        v_new = v_i + (x_i - x_star) * f_i[:, None]
        x_new = clip_position(x_i + v_new, lower_bound, upper_bound)

        # Local search / random walk around x_star
        local_mask = rng.random(n_bats) > r_i
        if np.any(local_mask):
            A_avg = np.mean(A_i)
            epsilon = rng.uniform(-1.0, 1.0, size=(np.sum(local_mask), dimensions))
            x_new[local_mask] = x_star + epsilon * A_avg
            x_new[local_mask] = clip_position(
                x_new[local_mask], lower_bound, upper_bound
            )

        new_fitness = np.asarray([objective(x) for x in x_new], dtype=float)

        # Acceptance rule: improvement + random loudness check
        accept = (new_fitness < fitness) & (rng.random(n_bats) < A_i)

        x_i[accept] = x_new[accept]
        v_i[accept] = v_new[accept]
        fitness[accept] = new_fitness[accept]

        # Global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            x_star = x_i[best_idx].copy()
            best_fitness = fitness[best_idx]

        # Loudness decay and pulse-rate increase
        A_i *= alpha
        r_i = r_i * (1.0 - np.exp(-gamma * t))

        convergence_curve[t - 1] = best_fitness

    return x_star, float(best_fitness), convergence_curve


if __name__ == "__main__":
    print("=" * 55)
    print("BAT ALGORITHM (BA)")
    print("Objective : Minimize f(x) = sum(x^2)")
    print("Bounds    : [-10, 10]  |  Dimensions: 1")
    print("=" * 55)

    best_x, best_f, curve = bat_algorithm()
    print(f"Best position : {best_x}")
    print(f"Best fitness  : {best_f:.10f}")
    print(f"(Global optimum: x=0, f=0)")
