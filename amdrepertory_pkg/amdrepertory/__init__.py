"""
amdrepertory
============

Generalized AMD-stability criterion for multi-planet (three-planet) systems.

Quick start
-----------
>>> from amdrepertory import optimize_amd_crit
>>> out = optimize_amd_crit(periods=[1.0, 1.7, 2.9], masses=[3e-6, 3e-6, 3e-6])
>>> out["c"], out["AMD_crit"]

From a rebound simulation
-------------------------
>>> from amdrepertory import amd_crit_from_sim
>>> out = amd_crit_from_sim(sim)
>>> out["beta"], out["stable"]
"""

from . import functions
from .functions import (
    optdepthminus1,
    optdepthminus1_2bodyMMR,
    optdepthminus1_2bodyMMR_tau12,
    jac_opticaldepth,
    jac_opticaldepth_2bodyMMR_tau12,
    AMD_f_opti,
    AMD_J,
    const1,
    const2,
    const3,
    AMD_tot,
    AMD_tot_inc,
    AMD_crit,
    AMD_crit_eq,
    opticdepthtot,
    get_pool_params,
    get_centered_grid,
    relatively_prime,
)
from .optimize import (
    optimize_amd_crit,
    amd_crit_from_sim,
    AMDOptimizationError,
)

__version__ = "0.1.0"

__all__ = [
    "functions",
    "optimize_amd_crit",
    "amd_crit_from_sim",
    "AMDOptimizationError",
    "optdepthminus1",
    "optdepthminus1_2bodyMMR",
    "optdepthminus1_2bodyMMR_tau12",
    "jac_opticaldepth",
    "jac_opticaldepth_2bodyMMR_tau12",
    "AMD_f_opti",
    "AMD_J",
    "const1",
    "const2",
    "const3",
    "AMD_tot",
    "AMD_tot_inc",
    "AMD_crit",
    "AMD_crit_eq",
    "opticdepthtot",
    "get_pool_params",
    "get_centered_grid",
    "relatively_prime",
]
