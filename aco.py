"""
aco.py
------
Ant Colony Optimization (ACO) — Discrete TSP variant
Source equations from: bio 1.pdf (canonical) & bio 2.pdf (1D workout)

Key equations:
  eta_ij = 1 / d_ij                           (heuristic visibility)
  p_ij   = (tau_ij^alpha * eta_ij^beta) /
            sum_k(tau_ik^alpha * eta_ik^beta)  (transition probability)
  tau_ij = (1 - rho) * tau_ij + delta_tau_ij  (pheromone update)

Run:
  python aco.py
"""

import numpy as np


def aco_tsp(
    distance_matrix,
    n_ants=None,
    n_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    Q=1.0,
    seed=42,
):
    """
    Ant Colony Optimization for a discrete TSP.

    Parameters:
        distance_matrix : 2-D square array of city-to-city distances
        n_ants          : number of ants  (default = n_nodes)
        n_iterations    : number of iterations
        alpha           : pheromone importance
        beta            : heuristic importance
        rho             : evaporation rate
        Q               : pheromone deposit constant

    Returns:
        best_tour, best_distance, convergence_curve, tau_ij
    """
    rng = np.random.default_rng(seed)

    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n_nodes = distance_matrix.shape[0]

    if distance_matrix.shape != (n_nodes, n_nodes):
        raise ValueError("distance_matrix must be square.")
    if np.any(distance_matrix < 0):
        raise ValueError("Distances must be non-negative.")

    if n_ants is None:
        n_ants = n_nodes

    # eta_ij: heuristic visibility = 1 / d_ij
    eta_ij = np.zeros_like(distance_matrix)
    valid = distance_matrix > 0
    eta_ij[valid] = 1.0 / distance_matrix[valid]

    # tau_ij: pheromone concentration (initialized to 1)
    tau_ij = np.ones((n_nodes, n_nodes), dtype=float)

    best_tour = None
    best_distance = np.inf
    convergence_curve = np.empty(n_iterations)

    for iteration in range(n_iterations):
        tours = []
        tour_lengths = np.empty(n_ants)

        for ant in range(n_ants):
            start = rng.integers(n_nodes)
            tour = [int(start)]
            unvisited = np.ones(n_nodes, dtype=bool)
            unvisited[start] = False

            for _step in range(n_nodes - 1):
                current = tour[-1]
                candidates = np.flatnonzero(unvisited)

                # Transition probability p_ij
                weights = (
                    np.power(tau_ij[current, candidates], alpha)
                    * np.power(eta_ij[current, candidates], beta)
                )

                weight_sum = np.sum(weights)
                if weight_sum <= 0 or not np.isfinite(weight_sum):
                    p_ij = np.full(len(candidates), 1.0 / len(candidates))
                else:
                    p_ij = weights / weight_sum

                next_node = int(rng.choice(candidates, p=p_ij))
                tour.append(next_node)
                unvisited[next_node] = False

            tour = np.asarray(tour, dtype=int)

            # Closed tour length for TSP
            length = np.sum(distance_matrix[tour, np.roll(tour, -1)])
            tours.append(tour)
            tour_lengths[ant] = length

        # Track global best
        iteration_best = np.argmin(tour_lengths)
        if tour_lengths[iteration_best] < best_distance:
            best_distance = float(tour_lengths[iteration_best])
            best_tour = tours[iteration_best].copy()

        # Pheromone deposit
        delta_tau_ij = np.zeros_like(tau_ij)
        for tour, length in zip(tours, tour_lengths):
            deposit = Q / max(length, np.finfo(float).eps)
            for i, j in zip(tour, np.roll(tour, -1)):
                delta_tau_ij[i, j] += deposit
                delta_tau_ij[j, i] += deposit

        # Global evaporation & deposition
        tau_ij = (1.0 - rho) * tau_ij + delta_tau_ij

        convergence_curve[iteration] = best_distance

    return best_tour, best_distance, convergence_curve, tau_ij


if __name__ == "__main__":
    print("=" * 55)
    print("ANT COLONY OPTIMIZATION (ACO) — TSP Demo")
    print("5-city symmetric distance matrix")
    print("=" * 55)

    # Small 5-city symmetric TSP from the original demo
    distance_matrix = np.array([
        [0, 2, 9, 10, 7],
        [2, 0, 6,  4, 3],
        [9, 6, 0,  8, 5],
        [10, 4, 8,  0, 6],
        [7, 3, 5,  6, 0],
    ], dtype=float)

    tour, dist, curve, tau = aco_tsp(
        distance_matrix=distance_matrix,
        n_ants=10,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        Q=1.0,
        seed=42,
    )

    print(f"Best tour     : {tour}")
    print(f"Best distance : {dist:.10f}")
