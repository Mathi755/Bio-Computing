"""
pso.py
------
Particle Swarm Optimization (PSO) — Nature-Inspired Metaheuristic
Source equations from: bio 1.pdf (canonical) & bio 2.pdf (1D workout)

Key equations:
  v_i^t = w*v_i^(t-1)
          + c_1*r_1*(pbest_i - x_i^(t-1))
          + c_2*r_2*(gbest  - x_i^(t-1))
  x_i^t = x_i^(t-1) + v_i^t

Run:
  python pso.py
"""

import numpy as np
from common import sphere, clip_position


def pso(
    objective=sphere,
    n_particles=30,
    dimensions=1,
    n_iterations=200,
    lower_bound=-10.0,
    upper_bound=10.0,
    w=0.8,
    c_1=2.0,
    c_2=2.0,
    seed=42,
):
    """
    Particle Swarm Optimization using PDF 1 terminology.

    Returns:
        gbest_position, gbest_fitness, convergence_curve
    """
    rng = np.random.default_rng(seed)

    x_i = rng.uniform(lower_bound, upper_bound, (n_particles, dimensions))

    velocity_limit = 0.2 * (upper_bound - lower_bound)
    v_i = rng.uniform(-velocity_limit, velocity_limit, (n_particles, dimensions))

    fitness = np.asarray([objective(x) for x in x_i])
    pbest_i = x_i.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest_i[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(n_iterations):
        r_1 = rng.random((n_particles, dimensions))
        r_2 = rng.random((n_particles, dimensions))

        v_i = (
            w * v_i
            + c_1 * r_1 * (pbest_i - x_i)
            + c_2 * r_2 * (gbest - x_i)
        )

        v_i = np.clip(v_i, -velocity_limit, velocity_limit)
        x_i = clip_position(x_i + v_i, lower_bound, upper_bound)

        fitness = np.asarray([objective(x) for x in x_i])

        improved = fitness < pbest_fitness
        pbest_i[improved] = x_i[improved]
        pbest_fitness[improved] = fitness[improved]

        gbest_idx = np.argmin(pbest_fitness)
        if pbest_fitness[gbest_idx] < gbest_fitness:
            gbest = pbest_i[gbest_idx].copy()
            gbest_fitness = pbest_fitness[gbest_idx]

        convergence_curve[t] = gbest_fitness

    return gbest, float(gbest_fitness), convergence_curve


if __name__ == "__main__":
    print("=" * 55)
    print("PARTICLE SWARM OPTIMIZATION (PSO)")
    print("Objective : Minimize f(x) = sum(x^2)")
    print("Bounds    : [-10, 10]  |  Dimensions: 1")
    print("=" * 55)

    best_x, best_f, curve = pso()
    print(f"Best position : {best_x}")
    print(f"Best fitness  : {best_f:.10f}")
    print(f"(Global optimum: x=0, f=0)")
