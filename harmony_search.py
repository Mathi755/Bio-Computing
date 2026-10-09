"""
harmony_search.py
-----------------
Harmony Search (HS) — Nature-Inspired Metaheuristic
Source equations from: bio 1.pdf (canonical) & bio 2.pdf (1D workout)

Key parameters:
  HM   = Harmony Memory matrix  (HMS × dimensions)
  HMS  = Harmony Memory Size
  HMCR = Harmony Memory Consideration Rate
  PAR  = Pitch Adjusting Rate
  bw   = bandwidth

Pitch modification:
  x_new = x_old + rand(-1, 1) * bw

Run:
  python harmony_search.py
"""

import numpy as np
from common import sphere, clip_position


def harmony_search(
    objective=sphere,
    HMS=20,
    dimensions=1,
    n_iterations=200,
    lower_bound=-10.0,
    upper_bound=10.0,
    HMCR=0.90,
    PAR=0.40,
    bw=0.10,
    seed=42,
):
    """
    Harmony Search using PDF 1 terminology.

    Parameters:
        HMS  : Harmony Memory Size (number of solution vectors stored)
        HMCR : probability of choosing from Harmony Memory
        PAR  : probability of pitch adjustment after HM selection
        bw   : pitch adjustment bandwidth

    Returns:
        best_harmony, best_fitness, convergence_curve, HM (final memory)
    """
    rng = np.random.default_rng(seed)

    # Initialize Harmony Memory
    HM = rng.uniform(lower_bound, upper_bound, (HMS, dimensions))
    HM_fitness = np.asarray([objective(h) for h in HM])

    best_idx = np.argmin(HM_fitness)
    best_harmony = HM[best_idx].copy()
    best_fitness = HM_fitness[best_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(n_iterations):
        x_new = np.empty(dimensions)

        for j in range(dimensions):
            if rng.random() < HMCR:
                # Memory consideration
                memory_index = int(rng.integers(HMS))
                x_new[j] = HM[memory_index, j]

                # Pitch adjustment
                if rng.random() < PAR:
                    x_new[j] += rng.uniform(-1.0, 1.0) * bw
            else:
                # Random selection
                x_new[j] = rng.uniform(lower_bound, upper_bound)

        x_new = clip_position(x_new, lower_bound, upper_bound)
        new_fitness = objective(x_new)

        # Replace the worst harmony in HM
        worst_idx = np.argmax(HM_fitness)
        if new_fitness < HM_fitness[worst_idx]:
            HM[worst_idx] = x_new
            HM_fitness[worst_idx] = new_fitness

        best_idx = np.argmin(HM_fitness)
        if HM_fitness[best_idx] < best_fitness:
            best_harmony = HM[best_idx].copy()
            best_fitness = HM_fitness[best_idx]

        convergence_curve[t] = best_fitness

    return best_harmony, float(best_fitness), convergence_curve, HM


if __name__ == "__main__":
    print("=" * 55)
    print("HARMONY SEARCH (HS)")
    print("Objective : Minimize f(x) = sum(x^2)")
    print("Bounds    : [-10, 10]  |  Dimensions: 1")
    print("=" * 55)

    best_x, best_f, curve, HM = harmony_search()
    print(f"Best harmony  : {best_x}")
    print(f"Best fitness  : {best_f:.10f}")
    print(f"(Global optimum: x=0, f=0)")
