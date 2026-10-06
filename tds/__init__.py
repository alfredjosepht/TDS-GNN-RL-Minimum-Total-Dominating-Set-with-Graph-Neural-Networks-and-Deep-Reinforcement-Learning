"""Minimum Total Dominating Set with GNN + Double DQN."""
from .checker import assert_tds, is_total_dominating_set, verify
from .graph import Graph, NoTotalDominatingSetError
from .result import SolveResult
# `tds.solve` is a callable module (see tds/solve.py), so `from tds import solve; solve(G)`
# works and `python -m tds.solve` runs the CLI without importing it twice.

__version__ = "0.1.0"


__all__ = ["Graph", "NoTotalDominatingSetError", "SolveResult", "solve", "verify",
           "assert_tds", "is_total_dominating_set"]
