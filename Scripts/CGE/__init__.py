"""
CGE Package
-----------
Multi-Agent Computable General Equilibrium (MA-CGE) simulation modules.

This package contains the MA-CGE implementation for comparison with the
Cybernetic In-Kind Planning (CIKP) system.
"""

from .ma_calibration import MACGEState, calibrate_ma
from .agents import Firm, Household
from .ma_simulation import run_quarter_ma, run_simulation_ma, run_montecarlo_ma
from .compare import run_comparison, run_montecarlo_comparison, summarize_comparison

__all__ = [
    'MACGEState', 'calibrate_ma',
    'Firm', 'Household',
    'run_quarter_ma', 'run_simulation_ma', 'run_montecarlo_ma',
    'run_comparison', 'run_montecarlo_comparison', 'summarize_comparison'
]
