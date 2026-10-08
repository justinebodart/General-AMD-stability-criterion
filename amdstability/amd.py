"""
amdstability.amd
================

Angular Momentum Deficit (AMD) utilities functions : 
- circular angular momenta ``Lambda_i``
- the actual AMD of a system (from orbital elements or from a ``rebound`` simulation)
- the quadratic AMD used as the objective of the constrained minimization.

Convention for ``Lambda``
------------------------- 
``Lambda_i = m_i * sqrt(G * m0 * a_i)`` and ``a_i = (G * m0 * P_i**2 / (4 pi**2))**(1/3)`.
"""

import numpy as np

__all__ = [
    "semi_major_axis",
    "lambdas_from_periods",
    "amd_from_elements",
    "amd_from_sim",
    "amd_quadratic",
    "amd_quadratic_jac",
]


def semi_major_axis(period, m0=1.0, G=1.0):
    """Semi-major axis from the orbital period."""
    mu = G * m0
    return (mu * np.asarray(period, dtype=float) ** 2 / (4.0 * np.pi ** 2)) ** (1 / 3)


def lambdas_from_periods(periods, masses, m0=1.0, G=1.0):
    """Circular angular momenta ``Lambda_i`` of every planet."""
    periods = np.asarray(periods, dtype=float)
    masses = np.asarray(masses, dtype=float)
    mu = G * m0
    a = semi_major_axis(periods, m0=m0, G=G)
    return masses * np.sqrt(mu * a)


def amd_from_elements(periods, masses, eccs, incs=None, m0=1.0, G=1.0):
    """Actual AMD of a system from its orbital elements.

    ``AMD = sum_i Lambda_i * (1 - sqrt(1 - e_i**2) * cos(i_i))``.
    Inclinations are optional (default to a coplanar system).
    """
    lam = lambdas_from_periods(periods, masses, m0=m0, G=G)
    eccs = np.asarray(eccs, dtype=float)
    if incs is None:
        cosi = np.ones_like(eccs)
    else:
        cosi = np.cos(np.asarray(incs, dtype=float))
    return float(np.sum(lam * (1.0 - np.sqrt(1.0 - eccs ** 2) * cosi)))


def amd_from_sim(sim, coplanar=False):
    """Actual AMD from a ``rebound`` simulation (star = particle 0)."""
    mu = sim.G * sim.particles[0].m
    total = 0.0
    for pl in sim.particles[1:]:
        cosi = 1.0 if coplanar else np.cos(pl.inc)
        total += pl.m * np.sqrt(mu * pl.a) * (1.0 - np.sqrt(1.0 - pl.e**2) * cosi)
    return float(total)


def amd_quadratic(c, lambdas):
    """Objective function of the minimization problem =``0.5 * sum_i Lambda_i * c_i**2``.

    This is the low-eccentricity expansion of the AMD, with ``c_i`` playing the
    role of an eccentricity.
    """
    c = np.asarray(c, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    return float(0.5 * np.sum(lambdas * c ** 2))


def amd_quadratic_jac(c, lambdas):
    """Gradient of :func:`amd_quadratic` = ``Lambda_i * c_i``."""
    return np.asarray(lambdas, dtype=float) * np.asarray(c, dtype=float)


def ecc_from_amd(C_i, Lambda_i):
    """Eccentricity e_i carried by planet i holding an AMD C_i (with circular
    angular momentum Lambda_i)."""
    return np.sqrt(1.0 - (1.0 - np.asarray(C_i) / np.asarray(Lambda_i)) ** 2)
