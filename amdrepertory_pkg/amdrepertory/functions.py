"""
amdrepertory.functions
=======================

Physics functions for the generalized AMD-stability criterion of multi-planet
systems (Generalized AMD-stability criterion for multi-planet systems).

Conventions
-----------
- Three planets, indices 1, 2, 3 (innermost to outermost).
- ``args`` tuple is always: (m0, m1, m2, m3, P1, P2, P3, G, AMD)
  unless stated otherwise.
- ``c`` is the optimization vector of eccentricity-like variables [c1, c2, c3].
- alpha_ij = (Pi/Pj)**(2/3)  (semi-major axis ratio)
- delta_ij = 1 - alpha_ij    (normalized orbital separation)
- S_ij     = delta_ij / epsilon_ij**(1/4)  (spacing)
- tau      : MMR overlap "optical depth" (Hadden & Lithwick style + 3-body term)
"""

import numpy as np

__all__ = [
    "get_pool_params",
    "get_centered_grid",
    "relatively_prime",
    "optdepthminus1",
    "optdepthminus1_2bodyMMR_tau12",
    "optdepthminus1_2bodyMMR",
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
]


# --------------------------------------------------------------------------- #
# Helpers for parallel grids / plotting
# --------------------------------------------------------------------------- #
def get_pool_params(xlist, ylist, P3, mu, tmax=None):
    """Build a flat list of parameter tuples for a 2D grid (for use with Pool)."""
    params = []
    ctr = 0
    for y in ylist:
        for x in xlist:
            params.append((ctr, x, P3, y, mu, tmax))
            ctr += 1
    return params


def get_centered_grid(xlist, ylist, poolresults):
    """Return cell-edge meshgrid (X, Y) and reshaped Z for pcolormesh.

    Assumes uniformly spaced xlist and ylist (lengths may differ).
    """
    xlist = np.asarray(xlist, dtype=float)
    ylist = np.asarray(ylist, dtype=float)

    dx = xlist[1] - xlist[0]
    dy = ylist[1] - ylist[0]

    xgrid = np.concatenate([xlist - dx / 2, [xlist[-1] + dx / 2]])
    ygrid = np.concatenate([ylist - dy / 2, [ylist[-1] + dy / 2]])

    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.asarray(poolresults).reshape(len(ylist), len(xlist))
    return X, Y, Z


def relatively_prime(num1, num2):
    """Return True if gcd(num1, num2) == 1. Robust to ordering and small ints."""
    return m_gcd(int(num1), int(num2)) == 1


def m_gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


# --------------------------------------------------------------------------- #
# Internal: derive all geometric quantities once (DRY)
# --------------------------------------------------------------------------- #
def _geometry(m0, m1, m2, m3, P1, P2, P3):
    """Compute the common geometric/mass quantities shared by tau functions."""
    epsilon12 = (m1 + m2) / m0
    epsilon23 = (m2 + m3) / m0

    P12 = P1 / P2
    P23 = P2 / P3

    alpha12 = P12 ** (2 / 3)
    alpha23 = P23 ** (2 / 3)

    delta12 = 1 - alpha12
    delta23 = 1 - alpha23

    S_12 = delta12 / epsilon12 ** (1 / 4)
    S_23 = delta23 / epsilon23 ** (1 / 4)

    eta = P12 * (1 - P23) / (1 - P12 * P23)          # resonance locator
    delta = delta12 * delta23 / (delta12 + delta23)  # generalized spacing

    # epsilon_M (three-body strength factor)
    Mfac = (1 / m0) * (
        m1 * m3
        + m2 * m3 * eta ** 2 * alpha12 ** (-2)
        + m1 * m2 * (1 - eta) ** 2 * alpha23 ** 2
    ) ** 0.5

    return dict(
        epsilon12=epsilon12, epsilon23=epsilon23,
        alpha12=alpha12, alpha23=alpha23,
        delta12=delta12, delta23=delta23,
        S_12=S_12, S_23=S_23,
        eta=eta, delta=delta, Mfac=Mfac,
    )


def _tau_pair(c_sum, S, delta_pair):
    """Two-body MMR optical depth for one pair (Hadden form)."""
    return (
        (1.8 / S) ** 2
        * (c_sum / delta_pair) ** 0.5
        * abs(1 - c_sum / delta_pair) ** (-1.5)
    )


def _tau_three_body(g):
    """Three-body MMR optical depth term."""
    return 6.55 * g["Mfac"] * (g["eta"] * (1 - g["eta"])) ** 1.5 / g["delta"] ** 4


# --------------------------------------------------------------------------- #
# Optical-depth constraint functions: f(c) = tau - 1
# --------------------------------------------------------------------------- #
def optdepthminus1(c, *args):
    """Full criterion: tau_12 + tau_23 + tau_123 - 1.  (article default)"""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    g = _geometry(m0, m1, m2, m3, P1, P2, P3)
    tau_12 = _tau_pair(c[0] + c[1], g["S_12"], g["delta12"])
    tau_23 = _tau_pair(c[1] + c[2], g["S_23"], g["delta23"])
    tau_123 = _tau_three_body(g)
    return (tau_12 + tau_23 + tau_123) - 1


def optdepthminus1_2bodyMMR(c, *args):
    """Two-body only: tau_12 + tau_23 - 1."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    g = _geometry(m0, m1, m2, m3, P1, P2, P3)
    tau_12 = _tau_pair(c[0] + c[1], g["S_12"], g["delta12"])
    tau_23 = _tau_pair(c[1] + c[2], g["S_23"], g["delta23"])
    return (tau_12 + tau_23) - 1


def optdepthminus1_2bodyMMR_tau12(c, *args):
    """Single pair (1-2) only: tau_12 - 1."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    g = _geometry(m0, m1, m2, m3, P1, P2, P3)
    tau_12 = _tau_pair(c[0] + c[1], g["S_12"], g["delta12"])
    return tau_12 - 1


# --------------------------------------------------------------------------- #
# Jacobians of the optical depth
# --------------------------------------------------------------------------- #
def _dtau_pair_dcsum(c_sum, S, delta_pair):
    """d(tau_pair)/d(c_sum). Sum of the two chain-rule terms.

    tau = (1.8/S)^2 * u^0.5 * |1-u|^-1.5  with u = c_sum/delta_pair
    d tau/d c_sum = (1.8/S)^2 * (1/delta) *
        [ 0.5 u^-0.5 |1-u|^-1.5  +  1.5 u^0.5 |1-u|^-2.5 ]
    """
    pref = (1.8 / S) ** 2 / delta_pair
    u = c_sum / delta_pair
    term1 = 0.5 * u ** (-0.5) * abs(1 - u) ** (-1.5)
    term2 = 1.5 * u ** 0.5 * abs(1 - u) ** (-2.5)
    return pref * (term1 + term2)


def jac_opticaldepth(c, *args):
    """Gradient of optdepthminus1 w.r.t. (c1, c2, c3).

    tau_12 depends on c1+c2, tau_23 on c2+c3, tau_123 is c-independent.
    -> d/dc1 = dtau_12 ; d/dc2 = dtau_12 + dtau_23 ; d/dc3 = dtau_23
    """
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    g = _geometry(m0, m1, m2, m3, P1, P2, P3)
    dtau_12 = _dtau_pair_dcsum(c[0] + c[1], g["S_12"], g["delta12"])
    dtau_23 = _dtau_pair_dcsum(c[1] + c[2], g["S_23"], g["delta23"])
    return np.array([dtau_12, dtau_12 + dtau_23, dtau_23])


def jac_opticaldepth_2bodyMMR_tau12(c, *args):
    """Gradient of optdepthminus1_2bodyMMR_tau12 (only tau_12)."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    g = _geometry(m0, m1, m2, m3, P1, P2, P3)
    dtau_12 = _dtau_pair_dcsum(c[0] + c[1], g["S_12"], g["delta12"])
    return np.array([dtau_12, dtau_12, 0.0])


# --------------------------------------------------------------------------- #
# Objective: AMD as a function of c (to minimize)
# --------------------------------------------------------------------------- #
def _lambdas(m0, m1, m2, m3, P1, P2, P3, G):
    mu = G * m0
    a1 = (mu * P1 ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    a2 = (mu * P2 ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    a3 = (mu * P3 ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    return (
        m1 * (mu * a1) ** 0.5,
        m2 * (mu * a2) ** 0.5,
        m3 * (mu * a3) ** 0.5,
    )


def AMD_f_opti(c, *args):
    """AMD objective: 0.5 * sum_i Lambda_i c_i^2."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    L1, L2, L3 = _lambdas(m0, m1, m2, m3, P1, P2, P3, G)
    return 0.5 * (L1 * c[0] ** 2 + L2 * c[1] ** 2 + L3 * c[2] ** 2)


def AMD_J(c, *args):
    """Gradient of AMD_f_opti: [Lambda_i * c_i]."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    L1, L2, L3 = _lambdas(m0, m1, m2, m3, P1, P2, P3, G)
    return np.array([L1 * c[0], L2 * c[1], L3 * c[2]])


# --------------------------------------------------------------------------- #
# Inequality constraints: critical eccentricities (Hadden)
# --------------------------------------------------------------------------- #
def const1(c, *args):
    """ecrit_12 - (c1 + c2) >= 0."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    epsilon12 = (m1 + m2) / m0
    alpha12 = (P1 / P2) ** (2 / 3)
    delta12 = 1 - alpha12
    S_12 = delta12 / epsilon12 ** (1 / 4)
    ecrit12 = delta12 * np.exp(-(1.8 / S_12) ** (4 / 3))
    return ecrit12 - c[0] - c[1]


def const2(c, *args):
    """ecrit_23 - (c2 + c3) >= 0."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    epsilon23 = (m2 + m3) / m0
    alpha23 = (P2 / P3) ** (2 / 3)
    delta23 = 1 - alpha23
    S_23 = delta23 / epsilon23 ** (1 / 4)
    ecrit23 = delta23 * np.exp(-(1.8 / S_23) ** (4 / 3))
    return ecrit23 - c[1] - c[2]


def const3(c, *args):
    """ecrit_13 - (c1 + c3) >= 0.   (planets 1 & 3)"""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    epsilon13 = (m1 + m3) / m0
    alpha13 = (P1 / P3) ** (2 / 3)
    delta13 = 1 - alpha13
    S_13 = delta13 / epsilon13 ** (1 / 4)
    ecrit13 = delta13 * np.exp(-(1.8 / S_13) ** (4 / 3))
    return ecrit13 - c[0] - c[2]   # FIXED: was c[0] - c[1]


# --------------------------------------------------------------------------- #
# AMD diagnostics from a rebound simulation
# --------------------------------------------------------------------------- #
def AMD_tot(sim):
    """Total AMD without inclination."""
    res = 0.0
    mu = sim.G * sim.particles[0].m
    for pl in sim.particles[1:]:
        res += pl.m * (mu * pl.a) ** 0.5 * (1 - (1 - pl.e ** 2) ** 0.5)
    return res


def AMD_tot_inc(sim):
    """Total AMD with inclination."""
    res = 0.0
    mu = sim.G * sim.particles[0].m
    for pl in sim.particles[1:]:
        res += (
            pl.m * (mu * pl.a) ** 0.5
            * (1 - (1 - pl.e ** 2) ** 0.5 * np.cos(pl.inc))
        )
    return res


def AMD_crit(sim, i):
    """Pairwise AMD-critical value between planets i and i+1 (Laskar & Petit)."""
    ps = sim.particles
    mu = sim.G * ps[0].m
    Lambda_i = ps[i].m * (mu * ps[i].a) ** 0.5
    Lambda_j = ps[i + 1].m * (mu * ps[i + 1].a) ** 0.5
    epsilon = (ps[i].m + ps[i + 1].m) / ps[0].m
    alpha = (ps[i].P / ps[i + 1].P) ** (2 / 3)
    delta = 1 - alpha
    S = delta / epsilon ** (1 / 4)
    return (
        0.5 * (Lambda_i * Lambda_j) / (Lambda_i + Lambda_j)
        * delta ** 2 * np.exp(-(3 / S) ** (4 / 3))
    )


def AMD_crit_eq(sim):
    """AMD-critical for equal masses and equal spacing (special case)."""
    ps = sim.particles
    alpha = (ps[1].P / ps[2].P) ** (2 / 3)
    mu = sim.G * ps[0].m
    epsilon = (2 * ps[2].m) / ps[0].m
    Lambda = ps[2].m * (mu * ps[2].a) ** 0.5
    tau_123 = (16 * (ps[2].m / ps[0].m)) / (1 - alpha) ** 4
    A = ((2.9) ** 2 * epsilon ** 0.5) / (1 - alpha) ** 3
    return (Lambda * (1 - tau_123) ** 2) / (A ** 2 * (2 * alpha ** 0.5 + 2 * (1 / alpha) ** 0.5 + 8))


def opticdepthtot(c, *args):
    """Three-body optical depth tau_123 alone.

    NOTE: this signature differs (no AMD field): (m0,m1,m2,m3,P1,P2,P3,G).
    """
    m0, m1, m2, m3, P1, P2, P3, G = args
    g = _geometry(m0, m1, m2, m3, P1, P2, P3)
    return _tau_three_body(g)
