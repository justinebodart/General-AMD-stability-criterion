"""
amdrepertory.functions
=======================

Functions for the generalized AMD-stability criterion of multi-planet
systems.

Conventions
-----------
- Three planets, indices 1, 2, 3 (innermost to outermost).
- ``args`` tuple: (m0, m1, m2, m3, P1, P2, P3, G, AMD)
- ``c`` is the optimization vector of eccentricity-like variables [c1, c2, c3].
- alpha_ij = (Pi/Pj)**(2/3)  (semi-major axis ratio)
- delta_ij = 1 - alpha_ij    (normalized orbital separation)
- S_ij     = delta_ij / epsilon_ij**(1/4)  (spacing)
- tau      : MMR overlap "optical depth" from Hadden & Lithwick + 3-body term from Petit et al. 2020
"""

import numpy as np

__all__ = [
    "optdepthminus1",
    "optdepthminus1_2bodyMMR",
    "jac_opticaldepth",
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

def m_gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


# Derive all geometric quantities

def _geometry(m0, m1, m2, m3, P1, P2, P3):
    """Compute the geometric/mass quantities."""
    epsilon12 = (m1 + m2) / m0
    epsilon23 = (m2 + m3) / m0

    P12 = P1 / P2
    P23 = P2 / P3

    alpha12 = P12 **(2/3)
    alpha23 = P23 **(2/3)

    delta12 = 1 - alpha12
    delta23 = 1 - alpha23

    S_12 = delta12 / (epsilon12 **(1/4))
    S_23 = delta23 / (epsilon23 **(1/4))

    eta = P12 * (1 - P23) / (1 - P12*P23)            # resonance locator
    delta = delta12 * delta23 / (delta12 + delta23)  # generalized spacing

    # epsilon_M (from Petit et al. 2020)
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


def _tau_pair(c_sum, S, delta_pair, eps=1e-8):
    """Two-body MMR optical depth for one pair (Hadden form)."""
    u = c_sum / delta_pair
    u = np.clip(u, eps, 1 - eps)
    return (1.8 / S) ** 2 * u ** 0.5 * abs(1 - u) ** (-1.5)


def _tau_three_body(g):
    """Three-body MMR optical depth term."""
    return 6.55 * g["Mfac"] * (g["eta"] * (1 - g["eta"])) ** 1.5 / g["delta"] ** 4


# Optical-depth constraint functions: f(c) = tau - 1

def optdepthminus1(c, *args):
    """Full criterion: tau_12 + tau_23 + tau_123 - 1. """
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

# Jacobians of the optical depth for the optimization

def _dtau_pair_dcsum(c_sum, S, delta_pair, eps=1e-8):
    """d(tau_pair)/d(c_sum). Sum of the two chain-rule terms.

    tau = (1.8/S)^2 * u^0.5 * |1-u|^-1.5  with u = c_sum/delta_pair
    d tau/d c_sum = (1.8/S)^2 * (1/delta) *
        [ 0.5 u^-0.5 |1-u|^-1.5  +  1.5 u^0.5 |1-u|^-2.5 ]
    """
    pref = (1.8 / S) ** 2 / delta_pair
    u = c_sum / delta_pair
    u = np.clip(u, eps, 1 - eps)
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


# Objective: AMD as a function of c (to minimize)

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
    """Gradient of AMD_f_opti: Lambda_i * c_i."""
    m0, m1, m2, m3, P1, P2, P3, G, AMD = args
    L1, L2, L3 = _lambdas(m0, m1, m2, m3, P1, P2, P3, G)
    return np.array([L1 * c[0], L2 * c[1], L3 * c[2]])


# Inequality constraints: critical eccentricities (Hadden)

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

# AMD diagnostics from a rebound simulation

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
    """AMD-critical for equal masses and equal spacing in the low-eccentricities case."""
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
