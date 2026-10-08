"""
amdstability.twoplanetcrit
==========================

Classical **two-planet** AMD-stability criteria.

All the ``Cc_*`` functions return the *relative* critical AMD, i.e.
``C_c / Lambda'`` where ``Lambda'`` is the circular angular momentum of the
**outer** planet of the pair, as in Laskar & Petit (2017).  They take:

- ``alpha = a / a'``            semi-major axis ratio (inner / outer),
- ``gamma = m / m'``            mass ratio (inner / outer),
- ``epsilon = (m + m') / m0``   pair mass parameter.

Originally written by Antoine Petit.
"""

import numpy as np
import scipy.optimize as optimize

__all__ = [
    "Cc_col",
    "ecrit_collision",
    "Cc_MMR",
    "mmr_crit",
    "Cc_MMR_Hadden",
    "mmr_Hadden",
    "Cc_hill",
    "pa_hill",
    "hill_circular_spacing",
]

#  Collision criterion (Laskar & Petit 2017)

def Fe(alpha, gamma, e):
    return alpha * e + gamma * e / np.sqrt(
        np.abs(alpha * (1.0 - e ** 2) + gamma ** 2 * e ** 2)
    ) - 1.0 + alpha


def deF(alpha, gamma, e):
    return alpha + gamma * alpha / np.abs(
        alpha * (1.0 - e ** 2) + gamma ** 2 * e ** 2
    ) ** 1.5


def new_step(a,g,e):
    ns=Fe(a,g,e)/deF(a,g,e)
    #We add these lines to avoid the case of alpha=0 perturbing the computation
    return np.where(np.isnan(ns), 0.0, ns)

def ecrit_collision(alpha, gamma, tol=1e-14):
    """Critical eccentricity for a collision, computed by Newton's method."""
    
    ecc=np.zeros_like(alpha)
    necc=ecc-new_step(alpha,gamma,ecc)
    nit=0
    while (np.max(abs(ecc-necc))>tol) and (nit<20):
        nit+=1
        ecc=necc
        necc =np.minimum(1,necc-new_step(alpha,gamma,ecc))

    #Remove extra dimensions
    ecc=necc
    return ecc


def Cc_col(alpha, gamma, tol=1e-14):
    """Relative critical AMD for a collision in the secular system. (Laskar & Petit 2017)"""
    ec = ecrit_collision(alpha, gamma, tol=tol)
    return (gamma * np.sqrt(alpha) * (1.0 - np.sqrt(1.0 - ec ** 2))
        + (1.0 - np.sqrt(1.0 - ep(alpha, ec) ** 2)))

#  MMR overlap criterion (Petit, Laskar & Boue 2017)

def mmr_crit(alpha, epsilon):
    r = 0.8019857395
    return (3 ** 4 * (1.0 - alpha) ** 5) / (2 ** 9 * r * epsilon) - 32 * r * epsilon / (9 * (1.0 - alpha) ** 2)


def Cc_MMR(alpha, epsilon, gamma):
    """Relative critical AMD based on MMR overlap. (Petit, Laskar & Boue 2017)"""
    crit = mmr_crit(alpha, epsilon)
    return (crit > 0) * crit ** 2 / 2 * gamma * np.sqrt(alpha) / (
        1.0 + gamma * np.sqrt(alpha)
    )


def mmr_Hadden(alpha, epsilon):
    """MMR-overlap criterion proposed by Hadden & Lithwick (2018)."""
    return (1.0 - alpha) * np.exp(-2.2 * (epsilon / (1.0 - alpha) ** 4) ** (1 / 3))


def Cc_MMR_Hadden(alpha, epsilon, gamma):
    """Relative critical AMD based on the Hadden & Lithwick overlap criterion."""
    return (gamma * np.sqrt(alpha) / (1.0 + gamma * np.sqrt(alpha))
        * mmr_Hadden(alpha, epsilon) ** 2 / 2)


#  Hill-stability criterion (Petit, Laskar & Boue 2018)

def Cc_hill(alpha, epsilon, gamma):
    """Relative critical AMD based on two-planet Hill stability."""
    zeta = gamma / (1.0 + gamma)
    crit = (gamma * np.sqrt(alpha)+ 1.0 - 1.0 / (1.0 - zeta) * np.sqrt(
            alpha * (1.0 + 3 ** (4 / 3) * epsilon ** (2 / 3) * zeta * (1.0 - zeta))
            / (zeta + (1.0 - zeta) * alpha)))
    return crit * (crit > 0)


def pa_hill(alpha, epsilon, gamma, rC):
    """Classic Hill criterion as a function of ``p/a`` (Marchal & Bozis 1982)."""
    zeta = gamma / (1.0 + gamma)
    return ((1.0 - zeta) ** 2 * (zeta + (1.0 - zeta) * alpha) / alpha
        * (gamma * np.sqrt(alpha) + 1.0 - rC) ** 2
        / (1.0 + 3 ** (4 / 3) * epsilon ** (2 / 3) * zeta * (1.0 - zeta)))


def hill_circular_spacing(epsilon, gamma=1.0):
    """Semi-major axis ratio of circular Hill-stable planets (exact)."""
    guess = 1.0 - 2.4 * epsilon ** (1 / 3)
    return optimize.fsolve(Cc_hill, guess, args=(epsilon, gamma))

#  Small utilities

def ep(alpha, e):
    """Eccentricity of the outer orbit at the collision condition."""
    return 1.0 - alpha * (1.0 + e)

def e0(alpha):
    return np.minimum(1.0, 1.0 / alpha - 1.0)