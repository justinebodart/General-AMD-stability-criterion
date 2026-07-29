"""
amdrepertory.optimize
======================

Solve the constrained optimization that defines the generalized AMD-stability
criterion: find the eccentricity-like vector c = (c1, c2, c3) that *minimizes*
the AMD while *saturating* the MMR overlap optical depth (tau = 1).

The minimum AMD on that surface is AMD_crit; comparing it to the system's
actual AMD gives beta = AMD / AMD_crit  (beta < 1 stable, beta > 1 unstable).

Two entry points:
- ``optimize_amd_crit(periods, masses, ...)``  : raw inputs.
- ``amd_crit_from_sim(sim, ...)``              : from a rebound Simulation.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from . import functions as rep


class AMDOptimizationError(RuntimeError):
    """Raised when the constrained optimization fails or is inconsistent."""


# Two different initial guesses (we are unsure of the "right" basin / scaling).
_IC_SETS = (
    np.array([0.01, 0.01, 0.00]),
    np.array([0.05, 0.05, 0.05]),
)


def _solve_once(c0, args, method, eq_fun, eq_jac, ineq_funcs):
    """Run a single constrained minimization from one initial guess."""
    constraints = [{"type": "eq", "fun": eq_fun, "jac": eq_jac, "args": args}]
    for f in ineq_funcs:
        constraints.append({"type": "ineq", "fun": f, "args": args})

    bounds = Bounds(np.zeros(3), np.ones(3))

    return minimize(
        rep.AMD_f_opti,
        c0,
        method=method,
        jac=rep.AMD_J,
        constraints=constraints,
        bounds=bounds,
        args=args,
    )


def optimize_amd_crit(
    periods,
    masses,
    m0=1.0,
    G=1.0,
    method="trust-constr",
    eq_fun=rep.optdepthminus1,
    eq_jac=rep.jac_opticaldepth,
    ineq_funcs=(),
    constraint_tol=1e-4,
    agreement_rtol=5e-2,
    verbose=False,
):
    """Compute the AMD-critical configuration.

    Parameters
    ----------
    periods, masses : sequence of 3 floats
        Orbital periods P1,P2,P3 and planet masses m1,m2,m3.
    m0, G : float
        Central mass and gravitational constant (defaults to normalized units).
    method : {"trust-constr", "SLSQP"}
        Default trust-constr: robustly satisfies the single nonlinear equality
        constraint tau=1 together with the bounds. The tau surface has a
        singularity at c_sum = delta which makes SLSQP unreliable here; SLSQP
        remains available but is not recommended for this problem.
    eq_fun, eq_jac : callables
        Equality constraint tau-1 and its jacobian (default: full criterion).
    ineq_funcs : tuple of callables
        Optional inequality constraints (e.g. rep.const1, rep.const2).
    constraint_tol : float
        Max allowed |tau-1| for a solution to count as feasible.
    agreement_rtol : float
        Max relative difference between the two IC solutions' AMD.

    Returns
    -------
    dict with keys: c, AMD_crit, tau_residual, method, n_ic_agree, raw.

    Raises
    ------
    AMDOptimizationError
        If neither IC converges feasibly, or the two solutions disagree.
    """
    periods = np.asarray(periods, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if periods.shape != (3,) or masses.shape != (3,):
        raise ValueError("periods and masses must each have length 3.")

    # args layout expected everywhere: (m0,m1,m2,m3,P1,P2,P3,G,AMD)
    args = (m0, *masses, *periods, G, 0.0)

    results = []
    for k, c0 in enumerate(_IC_SETS):
        res = _solve_once(c0, args, method, eq_fun, eq_jac, ineq_funcs)
        tau_res = abs(eq_fun(res.x, *args))  # |tau - 1|
        feasible = (
            res.success
            and tau_res <= constraint_tol
            and np.all(res.x >= -1e-9)
        )
        if verbose:
            print(f"IC#{k}: success={res.success} c={res.x} "
                  f"|tau-1|={tau_res:.2e} feasible={feasible} msg={res.message!r}")
        if feasible:
            results.append((res, tau_res))

    if not results:
        raise AMDOptimizationError(
            "Optimization failed: no initial condition produced a feasible "
            "solution (tau=1 not satisfied within tolerance). Check the inputs "
            "or try method='trust-constr'."
        )

    # Pick the solution that best satisfies the constraint.
    best_res, best_tau = min(results, key=lambda t: t[1])
    AMD_crit = rep.AMD_f_opti(best_res.x, *args)

    # Cross-check: if BOTH ICs converged, their AMD should agree.
    n_agree = len(results)
    if n_agree == 2:
        amds = [rep.AMD_f_opti(r.x, *args) for r, _ in results]
        rel = abs(amds[0] - amds[1]) / max(abs(amds[0]), abs(amds[1]), 1e-30)
        if rel > agreement_rtol:
            raise AMDOptimizationError(
                f"Optimization inconsistent: the two initial conditions gave "
                f"AMD values differing by {rel:.1%} (> {agreement_rtol:.0%}). "
                f"Solutions did not converge to the same optimum: "
                f"{amds[0]:.3e} vs {amds[1]:.3e}."
            )

    return {
        "c": best_res.x,
        "AMD_crit": AMD_crit,
        "tau_residual": best_tau,
        "method": method,
        "n_ic_agree": n_agree,
        "raw": best_res,
    }


def amd_crit_from_sim(sim, with_inclination=False, **kwargs):
    """Compute AMD_crit, the actual AMD and beta directly from a rebound sim.

    Returns a dict including 'AMD' (actual), 'AMD_crit', and
    'beta' = AMD / AMD_crit  (<1 stable, >1 unstable).
    """
    ps = sim.particles
    if len(ps) < 4:
        raise ValueError("Simulation must have a central body + 3 planets.")

    masses = [ps[i].m for i in (1, 2, 3)]
    periods = [ps[i].P for i in (1, 2, 3)]
    m0 = ps[0].m
    G = sim.G

    out = optimize_amd_crit(periods, masses, m0=m0, G=G, **kwargs)

    AMD_actual = rep.AMD_tot_inc(sim) if with_inclination else rep.AMD_tot(sim)
    out["AMD"] = AMD_actual
    out["beta"] = AMD_actual / out["AMD_crit"] if out["AMD_crit"] else np.inf
    out["stable"] = out["beta"] < 1.0
    return out
