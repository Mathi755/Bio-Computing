
# ============================================================
# Nature-Inspired / Bio-Inspired Metaheuristic Algorithms
# Kaggle-ready NumPy implementations
#
# Source terminology/formulas:
#   bio 1.pdf = canonical theoretical equations
#   bio 2.pdf = 1D numerical-workout reference
#
# Algorithms:
#   1. Bat Algorithm (BA)
#   2. Particle Swarm Optimization (PSO)
#   3. Ant Colony Optimization (ACO) - discrete TSP/path problem
#   4. Artificial Bee Colony (ABC)
#   5. Cuckoo Search (CS)
#   6. Harmony Search (HS)
#   7. Flower Pollination Algorithm (FPA)
#
# Default continuous benchmark:
#   Minimize f(x) = sum(x^2), bounds [-10, 10]
#   Global optimum: x = 0, f(x) = 0
# ============================================================

import numpy as np
import math


# ------------------------- COMMON ---------------------------

def sphere(x):
    """Baseline objective from the PDFs: f(x) = x^2 in 1D,
    generalized to sum(x^2) for a D-dimensional vector."""
    x = np.asarray(x)
    return np.sum(x * x, axis=-1)


def clip_position(x, lower_bound, upper_bound):
    return np.clip(x, lower_bound, upper_bound)


# ============================================================
# 1. BAT ALGORITHM (BA)
# ============================================================

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

    f_i = f_min + (f_max - f_min) * beta
    v_i^t = v_i^(t-1) + (x_i^(t-1) - x_*) * f_i
    x_i^t = x_i^(t-1) + v_i^t
    x_new = x_* + epsilon * A_avg^t
    A_i^t = alpha * A_i^(t-1)
    r_i^t = r_i^(t-1) * [1 - exp(-gamma*t)]

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


# ============================================================
# 2. PARTICLE SWARM OPTIMIZATION (PSO)
# ============================================================

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

    v_i^t = w*v_i^(t-1)
            + c_1*r_1*(pbest_i - x_i^(t-1))
            + c_2*r_2*(gbest - x_i^(t-1))

    x_i^t = x_i^(t-1) + v_i^t
    """
    rng = np.random.default_rng(seed)

    x_i = rng.uniform(lower_bound, upper_bound, (n_particles, dimensions))

    velocity_limit = 0.2 * (upper_bound - lower_bound)
    v_i = rng.uniform(
        -velocity_limit, velocity_limit, (n_particles, dimensions)
    )

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


# ============================================================
# 3. ANT COLONY OPTIMIZATION (ACO)
# ============================================================

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

    eta_ij = 1 / d_ij

    p_ij = (tau_ij^alpha * eta_ij^beta) /
           sum(tau_ik^alpha * eta_ik^beta)

    tau_ij = (1-rho)*tau_ij + delta_tau_ij

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

    # tau_ij: pheromone concentration
    tau_ij = np.ones((n_nodes, n_nodes), dtype=float)

    best_tour = None
    best_distance = np.inf
    convergence_curve = np.empty(n_iterations)

    for _ in range(n_iterations):
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

            # Closed tour for TSP
            length = np.sum(
                distance_matrix[tour, np.roll(tour, -1)]
            )

            tours.append(tour)
            tour_lengths[ant] = length

        # Global best
        iteration_best = np.argmin(tour_lengths)
        if tour_lengths[iteration_best] < best_distance:
            best_distance = float(tour_lengths[iteration_best])
            best_tour = tours[iteration_best].copy()

        # delta_tau_ij
        delta_tau_ij = np.zeros_like(tau_ij)
        for tour, length in zip(tours, tour_lengths):
            deposit = Q / max(length, np.finfo(float).eps)
            for i, j in zip(tour, np.roll(tour, -1)):
                delta_tau_ij[i, j] += deposit
                delta_tau_ij[j, i] += deposit

        # Global pheromone evaporation & deposition
        tau_ij = (1.0 - rho) * tau_ij + delta_tau_ij

        convergence_curve[_] = best_distance

    return best_tour, best_distance, convergence_curve, tau_ij


# ============================================================
# 4. ARTIFICIAL BEE COLONY (ABC)
# ============================================================

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

    v_ij = x_ij + phi * (x_ij - x_kj)
    prob_i = fitness_i / sum(fitness_k)

    For minimization, a positive quality fitness is constructed from
    objective values before applying the PDF's probability formula.
    """
    rng = np.random.default_rng(seed)

    x_ij = rng.uniform(
        lower_bound, upper_bound, (n_food_sources, dimensions)
    )

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
        # For minimization, lower objective = better source.
        # quality_i is used so that prob_i = fitness_i / sum(fitness_k)
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
            x_ij[i] = rng.uniform(
                lower_bound, upper_bound, dimensions
            )
            objective_values[i] = objective(x_ij[i])
            trials[i] = 0

        # ---------------- Best food source ----------------
        best_idx = np.argmin(objective_values)
        if objective_values[best_idx] < best_objective:
            best_solution = x_ij[best_idx].copy()
            best_objective = objective_values[best_idx]

        convergence_curve[t] = best_objective

    return best_solution, float(best_objective), convergence_curve


# ============================================================
# 5. CUCKOO SEARCH (CS)
# ============================================================

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

    x_i^t = x_i^(t-1) + alpha * Levy(lambda)

    x_new = x_i^(t-1) + r * (x_j^(t-1) - x_k^(t-1))
    when nest abandonment is triggered by p_a.
    """
    rng = np.random.default_rng(seed)

    x_i = rng.uniform(
        lower_bound, upper_bound, (n_nests, dimensions)
    )
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

        # Replace if better
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


# ============================================================
# 6. HARMONY SEARCH (HS)
# ============================================================

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

    HM  = Harmony Memory matrix
    HMS = Harmony Memory Size
    HMCR = Harmony Memory Consideration Rate
    PAR  = Pitch Adjusting Rate
    bw   = bandwidth

    Pitch modification:
        x_new = x_old + rand(-1,1) * bw
    """
    rng = np.random.default_rng(seed)

    # HM: Harmony Memory matrix
    HM = rng.uniform(
        lower_bound, upper_bound, (HMS, dimensions)
    )
    HM_fitness = np.asarray([objective(h) for h in HM])

    best_idx = np.argmin(HM_fitness)
    best_harmony = HM[best_idx].copy()
    best_fitness = HM_fitness[best_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(n_iterations):
        x_new = np.empty(dimensions)

        for j in range(dimensions):
            # Memory consideration
            if rng.random() < HMCR:
                memory_index = int(rng.integers(HMS))
                x_new[j] = HM[memory_index, j]

                # Pitch adjustment
                if rng.random() < PAR:
                    x_new[j] += rng.uniform(-1.0, 1.0) * bw
            else:
                x_new[j] = rng.uniform(lower_bound, upper_bound)

        x_new = clip_position(x_new, lower_bound, upper_bound)
        new_fitness = objective(x_new)

        # Replace the worst harmony
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


# ============================================================
# 7. FLOWER POLLINATION ALGORITHM (FPA)
# ============================================================

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
    FPA using the canonical theoretical equation in PDF 1.

    Global biotic pollination:
        x_i^t = x_i^(t-1) + L * (g_* - x_i^(t-1))

    Local abiotic pollination:
        x_i^t = x_i^(t-1)
                + epsilon * (x_j^(t-1) - x_k^(t-1))

    Switch:
        Global if rand < p, otherwise Local.
    """
    rng = np.random.default_rng(seed)

    x_i = rng.uniform(
        lower_bound, upper_bound, (n_flowers, dimensions)
    )
    fitness = np.asarray([objective(x) for x in x_i])

    best_idx = np.argmin(fitness)
    g_star = x_i[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.empty(n_iterations)

    for t in range(n_iterations):
        for i in range(n_flowers):
            if rng.random() < p:
                # Global biotic pollination
                L = levy_flight(rng, dimensions, levy_lambda)
                x_new = x_i[i] + L * (g_star - x_i[i])
            else:
                # Local abiotic pollination
                j, k = rng.choice(
                    [idx for idx in range(n_flowers) if idx != i],
                    size=2,
                    replace=False,
                )
                epsilon = rng.uniform(0.0, 1.0)
                x_new = (
                    x_i[i]
                    + epsilon * (x_i[j] - x_i[k])
                )

            x_new = clip_position(
                x_new, lower_bound, upper_bound
            )
            new_fitness = objective(x_new)

            if new_fitness < fitness[i]:
                x_i[i] = x_new
                fitness[i] = new_fitness

                if new_fitness < best_fitness:
                    g_star = x_new.copy()
                    best_fitness = new_fitness

        convergence_curve[t] = best_fitness

    return g_star, float(best_fitness), convergence_curve


# ============================================================
# DEMO: RUN ALL 7 ALGORITHMS ON THE PDF BASELINE
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("BIO-INSPIRED METAHEURISTIC ALGORITHMS")
    print("Objective: Minimize f(x) = sum(x^2)")
    print("Bounds: [-10, 10]")
    print("Expected global optimum: x = 0, f(x) = 0")
    print("=" * 65)

    results = {}

    x, f, curve = bat_algorithm()
    results["BA"] = (x, f)
    print(f"BA  -> best x = {x}, best f = {f:.10f}")

    x, f, curve = pso()
    results["PSO"] = (x, f)
    print(f"PSO -> best x = {x}, best f = {f:.10f}")

    x, f, curve = artificial_bee_colony()
    results["ABC"] = (x, f)
    print(f"ABC -> best x = {x}, best f = {f:.10f}")

    x, f, curve = cuckoo_search()
    results["CS"] = (x, f)
    print(f"CS  -> best x = {x}, best f = {f:.10f}")

    x, f, curve, HM = harmony_search()
    results["HS"] = (x, f)
    print(f"HS  -> best x = {x}, best f = {f:.10f}")

    x, f, curve = flower_pollination_algorithm()
    results["FPA"] = (x, f)
    print(f"FPA -> best x = {x}, best f = {f:.10f}")

    # ---------------- ACO DEMO ----------------
    # Small symmetric TSP distance matrix.
    # ACO is intentionally demonstrated on a DISCRETE path problem,
    # because the PDF explicitly specifies ACO for routing/path problems.
    distance_matrix = np.array([
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0],
    ], dtype=float)

    tour, distance, curve, tau_ij = aco_tsp(
        distance_matrix=distance_matrix,
        n_ants=10,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        Q=1.0,
        seed=42,
    )

    print(f"ACO -> best tour = {tour}")
    print(f"ACO -> best distance = {distance:.10f}")

    print("\nCompleted successfully.")
