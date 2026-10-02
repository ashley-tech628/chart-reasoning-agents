"""Solver registry for cost-aware branch activation."""

from __future__ import annotations

from chart_agents.solvers.arithmetic_solver import ArithmeticSolver
from chart_agents.solvers.boolean_solver import BooleanSolver
from chart_agents.solvers.comparison_solver import ComparisonSolver
from chart_agents.solvers.counting_solver import CountingSolver
from chart_agents.solvers.lookup_solver import LookupSolver
from chart_agents.solvers.label_matching_solver import LabelMatchingSolver

SOLVER_REGISTRY = {
    "comparison_solver": ComparisonSolver(),
    "arithmetic_solver": ArithmeticSolver(),
    "boolean_solver": BooleanSolver(),
    "counting_solver": CountingSolver(),
    "lookup_solver": LookupSolver(),
    "label_matching_solver": LabelMatchingSolver(),
}