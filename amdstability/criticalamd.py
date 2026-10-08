"""
amdstability.criticalamd
========================

Computation of the *critical* AMD of the General AMD-stability criterion.

Given a set of planets we look for the eccentricity-like vector ``c`` that
**minimizes** the AMD while **reaching the MMR-overlap condition**
``tau(c) = 1``.  The minimum is ``AMD_crit``; comparing it to the actual AMD of
the system gives ``beta = AMD / AMD_crit`` (``beta < 1`` stable).

- :func:`critical_amd_pair`    -- N = 2 planets.  Fully analytic.
- :func:`critical_amd_triplet` -- N = 3 planets.  Constrained optimization
  (``"SLSQP"`` or ``"trust-constr"``) from several initial conditions.

Objective
---------
In both cases the AMD is approximated at low eccentricity by
``0.5 * sum_i Lambda_i c_i**2`` (:func:`amdstability.amd.amd_quadratic`).

Two planets (Hadden & Lithwick 2018; Tamayo et al. 2021, appendix)
------------------------------------------------------------------
With ``delta = 1 - a1/a2`` (orbit-crossing eccentricity),
``S = delta / epsilon**(1/4)`` and ``epsilon = (m1 + m2) / m0``,

    AMD_crit = 0.5 * Lambda_r * delta**2 * exp(-(3/S)**(4/3)),
    Lambda_r = Lambda_1 Lambda_2 / (Lambda_1 + Lambda_2).

This is the minimum of ``0.5 (Lambda_1 c1**2 + Lambda_2 c2**2)`` under
``c1 + c2 = e_crit`` (anti-aligned orbits), with the equivalent critical
eccentricity ``e_crit = delta * exp(-0.5 * (3/S)**(4/3))``:

    c1 = Lambda_2 / (Lambda_1 + Lambda_2) * e_crit,
    c2 = Lambda_1 / (Lambda_1 + Lambda_2) * e_crit.

(Same as ``e_crit = delta * exp(-(1.8/S)**(4/3))`` of Hadden & Lithwick, since
``2 * 1.8**(4/3) = 3.03**(4/3)``.)

Three planets: initial conditions
---------------------------------
SLSQP and trust-constr can stop on a feasible point that is *not* the minimum
while reporting success, which overestimates ``AMD_crit``.  The optimization
is therefore started from many initial conditions and the lowest feasible
AMD is kept:

- the fixed guesses ``_IC_SETS``;
- a grid spread over the constraint surface: each pair is placed at
  ``u_ij = (c_i + c_j) / delta_ij`` in ``_GRID_U``, with ``c2 = 0`` or ``c2``
  shared between the two pairs (:func:`_grid_ics`).
"""

import numpy as np
from scipy.optimize import Bounds, minimize

from . import amd as _amd
from .opticaldepth import PairGeometry, TripletGeometry

__all__ = [
    "AMDOptimizationError",
    "critical_amd_pair",
    "critical_amd_triplet",
]


class AMDOptimizationError(RuntimeError):
    """Raised when the critical-AMD computation fails."""


#: Initial conditions.
_IC_SETS = (
    np.array([0.01, 0.01, 0.0]),
    np.array([0.001, 0.001, 0.001]),
    np.array([0.0, 0.01, 0.01]),
)

#: Values of ``u = (c_i + c_j) / delta`` used to build the grid of initial
#: conditions.  ``tau`` is singular at ``u = 1``, so the grid stays below.
_GRID_U = (0.05, 0.2, 0.5, 0.8)

#: Constant of the two-planet critical AMD, ``exp(-(PAIR_K / S)**(4/3))``.
PAIR_K = 3.0

_METHODS = ("SLSQP", "trust-constr")
_CRITERIA = ("full", "2body")

#: Tolerances tighter than the scipy defaults: the default ``ftol`` of SLSQP and
#: the default ``xtol`` of trust-constr stop the solvers before the minimum.
_SOLVER_OPTIONS = {
    "SLSQP": {"ftol": 1e-12, "maxiter": 1000},
    "trust-constr": {"xtol": 1e-10, "gtol": 1e-10, "maxiter": 1000},
}


def _as_arrays(periods, masses, n):
    periods = np.asarray(periods, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if periods.shape != (n,) or masses.shape != (n,):
        raise ValueError(f"Expected exactly {n} periods and {n} masses.")
    if np.any(masses <= 0):
        raise ValueError("Planet masses must be positive.")
    return periods, masses


#  ---------- N = 2 : Hadden & Lithwick (2018) / Tamayo et al. (2021) ----------

def critical_amd_pair(periods, masses, m0=1.0, G=1.0):
    """Critical AMD of a **two-planet** system (analytic).

    Parameters
    ----------
    periods, masses : sequence of 2 floats
        Orbital periods (inner first) and planet masses.
    m0, G : float
        Central mass and gravitational constant.

    Returns
    -------
    dict with keys ``c`` (critical eccentricities), ``AMD_crit``, ``ecrit``,
    ``tau_residual`` (``None``: no optimization), ``method``, ``degenerate``,
    ``n_ic_agree``.
    """
    periods, masses = _as_arrays(periods, masses, 2)
    geom = PairGeometry.from_pair(m0, masses[0], masses[1], periods[0], periods[1])
    L = _amd.lambdas_from_periods(periods, masses, m0=m0, G=G)

    Lr = L[0] * L[1] / (L[0] + L[1])
    AMD_crit = 0.5 * Lr * geom.delta ** 2 * np.exp(-(PAIR_K / geom.S) ** (4 / 3))
    ecrit = np.sqrt(2.0 * AMD_crit / Lr)     # = delta * exp(-0.5 (3/S)^(4/3))
    c = ecrit * np.array([L[1], L[0]]) / (L[0] + L[1])
    return {
        "c": c,
        "AMD_crit": float(AMD_crit),
        "ecrit": float(ecrit),
        "tau_residual": None,
        "method": "analytic",
        "degenerate": False,
        "n_ic_agree": None,
    }


#  ---------- N = 3 : general AMD criterion ----------

def _grid_ics(geom, grid_u=_GRID_U):
    """Initial conditions spread over the constraint surface.

    For every ``(u12, u23)`` in ``grid_u x grid_u`` the pair sums are
    ``s12 = u12 * delta_12`` and ``s23 = u23 * delta_23``, and ``c2`` is either
    0 or half of ``min(s12, s23)``.
    """
    ics = []
    for u12 in grid_u:
        for u23 in grid_u:
            s12, s23 = u12 * geom.inner.delta, u23 * geom.outer.delta
            ics.append(np.array([s12, 0.0, s23]))
            c2 = 0.5 * min(s12, s23)
            ics.append(np.array([s12 - c2, c2, s23 - c2]))
    return ics


def _solve_once(c0, weights, geom, criterion, method):
    """Run a single constrained minimization from one initial guess."""
    constraint = {
        "type": "eq",
        "fun": lambda c: geom.tau_minus_one(c, criterion),
        "jac": geom.jac_tau,          # tau_123 does not depend on c
    }
    return minimize(
        _amd.amd_quadratic,
        c0,
        args=(weights,),
        jac=_amd.amd_quadratic_jac,
        method=method,
        constraints=[constraint],
        bounds=Bounds(np.zeros(3), np.ones(3)),
        options=_SOLVER_OPTIONS[method],
    )


def critical_amd_triplet(
    periods,
    masses,
    m0=1.0,
    G=1.0,
    criterion="full",
    method="SLSQP",
    constraint_tol=1e-4,
    agreement_rtol=1e-3,
    ic_sets=None,
    verbose=False,
):
    """Critical AMD of a **three-planet** system.

    Parameters
    ----------
    periods, masses : sequence of 3 floats
        Orbital periods ``P1 < P2 < P3`` and planet masses.
    m0, G : float
        Central mass and gravitational constant (normalized units by default).
    criterion : {"full", "2body"}
        ``"full"`` includes the three-body overlap term ``tau_123``,
        ``"2body"`` keeps only the two-body terms.
    method : {"SLSQP", "trust-constr"}
        Constrained optimizer.
    constraint_tol : float
        Max allowed ``|tau - 1|`` for a solution to count as feasible.
    agreement_rtol : float
        Relative tolerance used to count the initial conditions that reached
        the minimum (``n_ic_agree``).
    ic_sets : sequence of arrays of shape (3,), optional
        Fixed initial guesses for ``c`` (default ``_IC_SETS``).  The grid of
        :func:`_grid_ics` is always added.
    verbose : bool
        Print the outcome of every initial guess.

    Returns
    -------
    dict with keys ``c``, ``AMD_crit``, ``tau_residual``, ``method``,
    ``degenerate``, ``n_ic_agree`` (number of initial conditions that reached
    the minimum within ``agreement_rtol``).

    Notes
    -----
    If the three-body term alone already reaches 1 the criterion is saturated by
    circular orbits and ``AMD_crit = 0``.  This is detected *before* any
    optimization and returned as ``degenerate=True`` instead of raising.

    The solvers minimize the AMD divided by ``sum_i Lambda_i * delta_min**2``
    (an O(1) quantity) so that their tolerances are meaningful whatever the
    units; ``AMD_crit`` is returned in physical units.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}.")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}.")
    periods, masses = _as_arrays(periods, masses, 3)

    geom = TripletGeometry.from_triplet(m0, masses, periods)
    L = _amd.lambdas_from_periods(periods, masses, m0=m0, G=G)

    def output(c, n_agree, degenerate=False):
        c = np.clip(np.asarray(c, dtype=float), 0.0, None)
        return {
            "c": c,
            "AMD_crit": 0.0 if degenerate else _amd.amd_quadratic(c, L),
            "tau_residual": float(abs(geom.tau_minus_one(c, criterion))),
            "method": method,
            "degenerate": degenerate,
            "n_ic_agree": n_agree,
        }

    # Three-body overlap alone is enough: circular orbits are already critical.
    if criterion == "full" and geom.tau_budget <= 0.0:
        if verbose:
            print(f"Degenerate triplet: tau_123 = {geom.tau_3body:.3f} >= 1.")
        return output(np.zeros(3), 0, degenerate=True)

    if ic_sets is None:
        ic_sets = _IC_SETS
    starts = [np.asarray(c0, dtype=float) for c0 in ic_sets] + _grid_ics(geom)

    # Objective scaled to O(1): AMD / (sum_i Lambda_i * delta_min**2).
    dmin = min(geom.inner.delta, geom.outer.delta)
    weights = L / (L.sum() * dmin ** 2)

    amds = np.full(len(starts), np.inf)
    xs = [None] * len(starts)
    for k, c0 in enumerate(starts):
        res = _solve_once(c0, weights, geom, criterion, method)
        tau_res = abs(geom.tau_minus_one(res.x, criterion))
        ok = bool(res.success) and tau_res <= constraint_tol and np.all(res.x >= -1e-9)
        if verbose:
            print(f"IC#{k}: success={res.success} c={res.x} |tau-1|={tau_res:.2e} "
                  f"feasible={ok} msg={res.message!r}")
        if ok:
            amds[k] = _amd.amd_quadratic(res.x, weights)
            xs[k] = res.x

    if not np.isfinite(amds).any():
        raise AMDOptimizationError(
            f"{method}: no initial condition produced a feasible solution "
            f"(|tau - 1| > {constraint_tol}). Check the inputs or try another method."
        )

    # We are minimizing: keep the lowest feasible AMD.
    k_best = int(np.argmin(amds))
    n_agree = int(np.sum(amds <= amds[k_best] * (1.0 + agreement_rtol)))
    return output(xs[k_best], n_agree)
