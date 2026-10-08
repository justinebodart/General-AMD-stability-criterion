"""
amdstability.results
====================

Containers returned by :class:`amdstability.AMDClassifier`.

- :class:`SubsystemResult`   -- critical AMD of one pair or one triplet.
- :class:`AMDStabilityResult` -- verdict for the whole system.

Convention
----------
The AMD is conserved for the **whole** system and can be exchanged between all
the planets, so every subsystem is compared with the **total** AMD of the
system (as in Laskar & Petit 2017):

    beta_k = AMD_total / AMD_crit_k .

The system verdict is that of the most constraining subsystem, i.e. the one
with the smallest ``AMD_crit`` (largest ``beta``).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["SubsystemResult", "AMDStabilityResult"]


@dataclass
class SubsystemResult:
    """Critical AMD of one subsystem (a pair or a triplet of planets).

    Attributes
    ----------
    indices : tuple of int
        Indices of the planets of the subsystem (0 = innermost planet).
    AMD_crit : float
        Critical AMD of the subsystem.
    beta : float or None
        ``AMD_total / AMD_crit`` (``None`` if no eccentricities were given).
    c : ndarray or None
        Critical eccentricities ``c_i`` at the minimum (``None`` for criteria
        that do not provide them).
    tau_residual : float or None
        ``|tau(c) - 1|`` at the solution.
    method : str
        ``"analytic"``, ``"SLSQP"`` or ``"trust-constr"``.
    degenerate : bool
        ``True`` if the three-body term alone saturates the criterion
        (``AMD_crit = 0``: circular orbits are already critical).
    """

    indices: Tuple[int, ...]
    AMD_crit: float
    beta: Optional[float] = None
    c: Optional[np.ndarray] = None
    tau_residual: Optional[float] = None
    method: str = "analytic"
    degenerate: bool = False

    @property
    def stable(self):
        return None if self.beta is None else bool(self.beta < 1.0)


@dataclass
class AMDStabilityResult:
    """AMD-stability of a planetary system.

    Attributes
    ----------
    criterion : str
        Name of the criterion (e.g. ``"general (full)"``, ``"Hill"``).
    n_planets : int
    AMD : float or None
        Actual (total) AMD of the system, ``None`` if no eccentricities given.
    subsystems : list of SubsystemResult
        One entry per pair (N = 2 or pairwise criteria) or per adjacent triplet.

    Properties
    ----------
    AMD_crit, beta, stable, limiting
        Values of the most constraining subsystem.
    """

    criterion: str
    n_planets: int
    AMD: Optional[float] = None
    subsystems: List[SubsystemResult] = field(default_factory=list)

    @property
    def limiting(self):
        """The most constraining subsystem (smallest ``AMD_crit``)."""
        if not self.subsystems:
            return None
        return min(self.subsystems, key=lambda s: s.AMD_crit)

    @property
    def AMD_crit(self):
        lim = self.limiting
        return np.inf if lim is None else lim.AMD_crit

    @property
    def beta(self):
        if self.AMD is None:
            return None
        if self.limiting is None:          # single planet
            return 0.0
        return _beta(self.AMD, self.AMD_crit)

    @property
    def stable(self):
        b = self.beta
        return None if b is None else bool(b < 1.0)

    def to_dict(self):
        """Plain-dict export (e.g. for ``pandas.DataFrame`` or JSON)."""
        lim = self.limiting
        return {
            "criterion": self.criterion,
            "n_planets": self.n_planets,
            "AMD": self.AMD,
            "AMD_crit": self.AMD_crit,
            "beta": self.beta,
            "stable": self.stable,
            "limiting_indices": None if lim is None else lim.indices,
            "subsystems": [
                {
                    "indices": s.indices,
                    "AMD_crit": s.AMD_crit,
                    "beta": s.beta,
                    "c": None if s.c is None else s.c.tolist(),
                    "tau_residual": s.tau_residual,
                    "method": s.method,
                    "degenerate": s.degenerate,
                }
                for s in self.subsystems
            ],
        }

    def __str__(self):
        def fmt(x):
            return "-" if x is None else f"{x:.4e}"

        verdict = {True: "STABLE", False: "UNSTABLE", None: "n/a (no eccentricities)"}
        lines = [
            f"AMD-stability [{self.criterion}]  N = {self.n_planets}",
            f"  AMD      = {fmt(self.AMD)}",
            f"  AMD_crit = {fmt(self.AMD_crit)}",
            f"  beta     = {fmt(self.beta)}  ->  {verdict[self.stable]}",
        ]
        if len(self.subsystems) > 1:
            lines.append("  subsystems:")
            lim = self.limiting
            for s in self.subsystems:
                flag = "  <- limiting" if s is lim else ""
                deg = "  (degenerate)" if s.degenerate else ""
                lines.append(f"    {s.indices}: AMD_crit = {fmt(s.AMD_crit)}  "
                             f"beta = {fmt(s.beta)}{deg}{flag}")
        return "\n".join(lines)


def _beta(AMD, AMD_crit):
    """``AMD / AMD_crit`` with ``AMD_crit = 0`` mapped to ``+inf``."""
    if AMD is None:
        return None
    return float(AMD / AMD_crit) if AMD_crit > 0 else np.inf
