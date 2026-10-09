"""
common.py
---------
Shared utilities used by all metaheuristic algorithm modules.
  - sphere()        : benchmark objective function  f(x) = sum(x^2)
  - clip_position() : bounds-enforcement helper
"""

import numpy as np


def sphere(x):
    """Baseline objective: f(x) = x^2 in 1D,
    generalized to sum(x^2) for a D-dimensional vector.
    Global optimum: x = 0, f(x) = 0."""
    x = np.asarray(x)
    return np.sum(x * x, axis=-1)


def clip_position(x, lower_bound, upper_bound):
    return np.clip(x, lower_bound, upper_bound)
