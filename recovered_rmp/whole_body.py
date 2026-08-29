"""Acceleration-QP composition used by the whole-body planner variant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .core import QPProblem, solve_box_qp


@dataclass(frozen=True)
class WeightedTask:
    problem: QPProblem
    weight: float = 1.0


def combine_qps(tasks: Iterable[WeightedTask]) -> QPProblem:
    """Add task metrics/forces and intersect their box constraints.

    This implements the global metric/force accumulation architecture confirmed
    in WBRMP.  Each task must already be expressed in the same global
    acceleration coordinates.  Proprietary balance-task weighting is outside
    this helper and must be supplied explicitly by the caller.
    """

    items = list(tasks)
    if not items:
        raise ValueError("at least one task is required")
    first = items[0].problem
    H = np.zeros_like(first.H, dtype=float)
    g = np.zeros_like(first.g, dtype=float)
    lower = np.full_like(first.lower, -np.inf, dtype=float)
    upper = np.full_like(first.upper, np.inf, dtype=float)
    for item in items:
        problem = item.problem
        if problem.H.shape != H.shape or problem.g.shape != g.shape:
            raise ValueError("all QPs must use the same global variables")
        if item.weight < 0.0:
            raise ValueError("task weight cannot be negative")
        H += item.weight * problem.H
        g += item.weight * problem.g
        lower = np.maximum(lower, problem.lower)
        upper = np.minimum(upper, problem.upper)
    if np.any(lower > upper):
        raise ValueError("combined task bounds are infeasible")
    return QPProblem(H, g, lower, upper)


def solve_weighted_tasks(tasks: Iterable[WeightedTask]) -> np.ndarray:
    return solve_box_qp(combine_qps(tasks))
