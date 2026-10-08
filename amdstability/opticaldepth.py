"""
amdstability.opticaldepth
=========================

Geometry of a planet pair / triplet and the MMR-overlap "optical depth" tau.

Conventions
-----------
- Planets are ordered from innermost to outermost.
- ``alpha_ij = (P_i / P_j)**(2/3)``       : semi-major axis ratio 
- ``delta_ij = 1 - alpha_ij``             : normalized orbital separation
- ``epsilon_ij = (m_i + m_j) / m0``       : pair mass parameter
- ``S_ij = delta_ij / epsilon_ij**(1/4)`` : spacing
- ``c`` is the vector of eccentricity-like variables.

The two-body term follows Hadden & Lithwick (2018). Two forms are possible :
 
- :meth:`PairGeometry.tau_hadden` -- the two-body MMR-overlap optical depth of
  Hadden & Lithwick (2018) as revised by Tamayo et al. (2021),
 
      tau(u) = (1.8/S)**2 * |ln u|**(-3/2) with ``u = (c_i + c_j) / delta``.
 
  This is the reference expression; use it whenever you want the two-planet
  criterion itself.
 
- :meth:`PairGeometry.tau` -- the form actually used in the three-planet optimization,
 
      tau(u) = (1.8/S)**2 * u**0.5 * (1-u)**(-1.5) with ``u = (c_i + c_j) / delta``.

The three-body term follows Petit et al. (2020).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "HADDEN_K",
    "THREE_BODY_K",
    "PairGeometry",
    "TripletGeometry",
]

#: Prefactor of the two-body MMR overlap width (Hadden & Lithwick 2018).
HADDEN_K = 1.8

#: Prefactor of the three-body MMR overlap term (Petit et al. 2020).
THREE_BODY_K = 6.55


@dataclass(frozen=True)
class PairGeometry:
    """Geometric and mass quantities of a single (inner, outer) planet pair."""

    epsilon: float
    alpha: float
    delta: float
    S: float

    @classmethod
    def from_pair(cls, m0, m_in, m_out, P_in, P_out):
        """Values from the central mass, the two planet masses and periods."""
        if not (P_in > 0 and P_out > 0):
            raise ValueError("Periods must be positive.")
        if P_in >= P_out:
            raise ValueError(
                f"Planets must be ordered from inner to outer "
                f"(got P_in={P_in!r} >= P_out={P_out!r})."
            )
        if m0 <= 0:
            raise ValueError("The central mass must be positive.")

        epsilon = (m_in + m_out) / m0
        alpha = (P_in / P_out) ** (2 / 3)
        delta = 1.0 - alpha          # (a_out - a_in)/a_out
        S = delta / epsilon ** 0.25  # Eq (6) Tamayo et al. (2021)
        return cls(epsilon=epsilon, alpha=alpha, delta=delta, S=S)

    # properties
    @property
    def k(self):
        """Prefactor ``(1.8 / S)**2`` of the two-body optical depth. See Eq(7) Tamayo et al. (2021). """
        return (HADDEN_K / self.S) ** 2

    @property
    def ecrit(self):
        """Hadden & Lithwick critical eccentricity of the pair. See Eq(8) Tamayo et al. (2021)."""
        return self.delta * np.exp(-(HADDEN_K / self.S) ** (4 / 3))
    
    # reference tau (Hadden / Tamayo)
    def tau_hadden(self, c_sum, eps=1e-300):
        """Reference two-body MMR optical depth (Hadden & Lithwick, rev. Tamayo).
        """
        u = np.clip(np.asarray(c_sum, dtype=float) / self.delta, eps, 1.0 - 1e-15)
        return self.k * np.abs(np.log(u)) ** (-1.5)
    
    # tau
    def tau(self, c_sum, eps=1e-12):
        """Two-body MMR optical depth ``tau_ij`` as a function of ``(c_i+c_j)``.

        ``tau(u) = (1.8/S)**2 * u**0.5 * (1-u)**(-1.5)`` with ``u = c_sum/delta``.
        Monotonically increasing on ``u in (0, 1)``, singular at ``u = 1``.
        """
        u = np.clip(np.asarray(c_sum, dtype=float) / self.delta, 0.0, 1.0 - eps)
        return self.k * u ** 0.5 * (1.0 - u) ** (-1.5)

    def dtau(self, c_sum, eps=1e-12):
        """``d tau / d c_sum``."""
        u = np.clip(np.asarray(c_sum, dtype=float) / self.delta, eps, 1.0 - eps)
        term1 = 0.5 * u ** (-0.5) * (1.0 - u) ** (-1.5)
        term2 = 1.5 * u ** 0.5 * (1.0 - u) ** (-2.5)
        return self.k * (term1 + term2) / self.delta

    def inv_tau(self, target):
        """Invert ``tau``: return the ``c_sum >= 0`` such that ``tau(c_sum) = target``.

        ``tau`` is strictly increasing from 0 (at ``u=0``) to ``+inf``
        (at ``u=1``), so the root always exists and is unique.  We solve the
        equivalent regular equation ``k*sqrt(u) - target*(1-u)**1.5 = 0``,
        which is well behaved on the closed interval ``[0, 1]``.
        """
        target = float(target)
        if target <= 0.0:
            return 0.0

        def residual(u):
            return self.k * np.sqrt(u) - target * (1.0 - u) ** 1.5

        u = brentq(residual, 0.0, 1.0, xtol=1e-15, rtol=8.9e-16, maxiter=200)
        return u * self.delta


@dataclass(frozen=True)
class TripletGeometry:
    """Geometry of three consecutive planets, i.e. two pairs + the 3-body term."""

    inner: PairGeometry   # pair (1, 2)
    outer: PairGeometry   # pair (2, 3)
    eta: float            # resonance locator
    delta: float          # generalized spacing
    Mfac: float           # epsilon_M of Petit et al. (2020)

    # values
    @classmethod
    def from_triplet(cls, m0, masses, periods):
        """Values from the central mass and the three planet masses/periods."""
        m1, m2, m3 = (float(m) for m in masses)
        P1, P2, P3 = (float(P) for P in periods)

        inner = PairGeometry.from_pair(m0, m1, m2, P1, P2)
        outer = PairGeometry.from_pair(m0, m2, m3, P2, P3)

        P12 = P1 / P2
        P23 = P2 / P3

        eta = P12 * (1.0 - P23) / (1.0 - P12 * P23)
        delta = (inner.delta * outer.delta) / (inner.delta + outer.delta)

        Mfac = (1.0 / m0) * (
            m1 * m3
            + m2 * m3 * eta ** 2 * inner.alpha ** (-2)
            + m1 * m2 * (1.0 - eta) ** 2 * outer.alpha ** 2
        ) ** 0.5

        return cls(inner=inner, outer=outer, eta=eta, delta=delta, Mfac=Mfac)

    # tau
    @property
    def tau_3body(self):
        """Three-body MMR optical depth ``tau_123`` (independent of ``c``)."""
        return (
            THREE_BODY_K
            * self.Mfac
            * (self.eta * (1.0 - self.eta)) ** 1.5
            / self.delta ** 4
        )

    def tau(self, c, criterion="full"):
        """Total optical depth.

        ``criterion="full"``   -> tau_12 + tau_23 + tau_123
        ``criterion="2body"``  -> tau_12 + tau_23
        """
        c = np.asarray(c, dtype=float)
        total = self.inner.tau(c[0] + c[1]) + self.outer.tau(c[1] + c[2])
        if criterion == "full":
            total = total + self.tau_3body
        elif criterion != "2body":
            raise ValueError("criterion must be 'full' or '2body'.")
        return float(total)

    def tau_minus_one(self, c, criterion="full"):
        """Equality-constraint function ``tau(c) - 1``."""
        return self.tau(c, criterion=criterion) - 1.0

    def jac_tau(self, c):
        """Gradient of ``tau`` w.r.t. ``(c1, c2, c3)``."""
        c = np.asarray(c, dtype=float)
        d12 = self.inner.dtau(c[0] + c[1])
        d23 = self.outer.dtau(c[1] + c[2])
        return np.array([d12, d12 + d23, d23])

    @property
    def tau_budget(self):
        """Budget left for the two-body terms: ``1 - tau_123``.

        If this is ``<= 0`` the three-body overlap alone already saturates the
        criterion and the critical AMD is zero (see
        :func:`amdstability.criticalamd.critical_amd_triplet`).
        """
        return 1.0 - self.tau_3body
