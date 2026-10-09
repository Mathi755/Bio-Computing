"""
flower_pollination.py
---------------------
Flower Pollination Algorithm (FPA) — Nature-Inspired Metaheuristic
Source equations from: bio 1.pdf (canonical) & bio 2.pdf (1D workout)

Key equations:
  Global (biotic) pollination:
    x_i^t = x_i^(t-1) + L * (g_* - x_i^(t-1))

  Local (abiotic) pollination:
    x_i^t = x_i^(t-1) + epsilon * (x_j^(t-1) - x_k^(t-1))

  Switch probability p: if rand < p → global, else → local

Run:
  python flower_pollination.py
"""

import math
import numpy as np
from common import sphere, clip_position


def levy_flight(rng, size, lambda_=1.5):
    """
    Mantegna-style Levy(lambda) random step — used as 'L' in FPA's
    global pollination step.
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


def flower_pollination_algorithm(
    objective=sphere,
    n_flowers=30,
    dimensions=1,
    n_iterations=200,
    lower_bound=-10.0,
    upper_bound=10.0,
    p=0.80,
    levy_lambda=1.5,
    seed=42,
):
    """
    Flower Pollination Algorithm using PDF 1 canonical equations.

    Parameters:
        n_flowers    : population size (number of flowers)
        p            : switch probability (global vs local pollination)
        levy_lambda  : Levy exponent for global pollination step

    Returns:
        best_position (g_star), best_fitness, convergence_curve
    """
    rng = np.random.default_rng(seed)

    x_i = rng.uniform(lower_bound, upper_bound, (n_flowers, dimensions))
    fitness = np.asarray([objective(x) for x in x_i])

    best_idx = np.argmin(fitness)
    g_star = x_i[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(n_iterations):
        for i in range(n_flowers):
            if rng.random() < p:
                # Global biotic pollination (Levy flight)
                L = levy_flight(rng, dimensions, levy_lambda)
                x_new = x_i[i] + L * (g_star - x_i[i])
            else:
                # Local abiotic / self-pollination
                j, k = rng.choice(
                    [idx for idx in range(n_flowers) if idx != i],
                    size=2,
                    replace=False,
                )
                epsilon = rng.uniform(0.0, 1.0)
                x_new = x_i[i] + epsilon * (x_i[j] - x_i[k])

            x_new = clip_position(x_new, lower_bound, upper_bound)
            new_fitness = objective(x_new)

            if new_fitness < fitness[i]:
                x_i[i] = x_new
                fitness[i] = new_fitness

                if new_fitness < best_fitness:
                    g_star = x_new.copy()
                    best_fitness = new_fitness

        convergence_curve[t] = best_fitness

    return g_star, float(best_fitness), convergence_curve


if __name__ == "__main__":
    print("=" * 55)
    print("FLOWER POLLINATION ALGORITHM (FPA)")
    print("Objective : Minimize f(x) = sum(x^2)")
    print("Bounds    : [-10, 10]  |  Dimensions: 1")
    print("=" * 55)

    best_x, best_f, curve = flower_pollination_algorithm()
    print(f"Best position : {best_x}")
    print(f"Best fitness  : {best_f:.10f}")
    print(f"(Global optimum: x=0, f=0)")
