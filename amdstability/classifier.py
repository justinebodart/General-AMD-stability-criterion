"""
amdstability.classifier
=======================

The class :class:`AMDClassifier` gathers every AMD-stability criterion behind a
single object.

N denotes the number of planets:

- ``N = 1`` -- stable (``beta = 0``).
- ``N = 2`` -- the solution is analytic (Hadden & Lithwick 2018 overlap,
  Tamayo et al. 2021).
- ``N = 3`` -- the general criterion.
- ``N > 3`` -- the criterion is applied to every set of three **adjacent**
  planets ``(i, i+1, i+2)``.  All the per-triplet results are returned, and the
  system verdict is that of the most constraining triplet (smallest
  ``AMD_crit``, compared with the total AMD of the system).

The classical two-planet criteria (Hill, MMR overlap of Petit et al. 2017,
collision of Laskar & Petit 2017) are available through
:meth:`AMDClassifier.HillAMDcrit`, :meth:`AMDClassifier.MMRAMDcrit` and
:meth:`AMDClassifier.CollisionAMDcrit`; they are applied to every adjacent pair.
"""

import numpy as np

from . import amd as _amd
from . import twoplanetcrit as _tpc
from .criticalamd import critical_amd_pair, critical_amd_triplet
from .results import AMDStabilityResult, SubsystemResult, _beta

__all__ = ["AMDClassifier", "elements_from_simulation"]


def elements_from_simulation(sim):
    """Extract ``periods, masses, eccs, incs, m0, G`` from a ``rebound`` simulation.

    The star is particle 0; every other particle is a planet.  The returned
    dict can be passed directly to the classifier methods, e.g.
    ``clf.HillAMDcrit(**elements_from_simulation(sim))``.
    """
    ps = sim.particles
    if sim.N < 2:
        raise ValueError("The simulation must contain a star and at least one planet.")
    planets = [ps[i] for i in range(1, sim.N)]
    if any(p.e >= 1.0 for p in planets):
        raise ValueError("All planets must be on bound orbits (e < 1).")
    return {
        "periods": np.array([p.P for p in planets]),
        "masses": np.array([p.m for p in planets]),
        "eccs": np.array([p.e for p in planets]),
        "incs": np.array([p.inc for p in planets]),
        "m0": ps[0].m,
        "G": sim.G,
    }


def _check_elements(periods, masses, eccs, incs):
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    masses = np.atleast_1d(np.asarray(masses, dtype=float))
    if periods.ndim != 1 or periods.shape != masses.shape or periods.size == 0:
        raise ValueError("periods and masses must be 1D arrays of the same length.")
    if np.any(periods <= 0) or np.any(masses <= 0):
        raise ValueError("Periods and masses must be positive.")
    if np.any(np.diff(periods) <= 0):
        raise ValueError("Planets must be ordered from inner to outer "
                         "(strictly increasing periods).")
    for name, x in (("eccs", eccs), ("incs", incs)):
        if x is not None and np.shape(x) != periods.shape:
            raise ValueError(f"{name} must have the same length as periods.")
    return periods, masses


class AMDClassifier:
    """AMD-stability classifier for systems with any number of planets.

    Parameters
    ----------
    criterion : {"full", "2body"}
        Triplet criterion: with (``"full"``) or without (``"2body"``) the
        three-body overlap term.  Ignored for N = 2.
    method : {"SLSQP", "trust-constr"}
        Solver of the triplet problem (see :mod:`amdstability.criticalamd`).
    constraint_tol, agreement_rtol, verbose
        Passed to :func:`amdstability.criticalamd.critical_amd_triplet`.

    Examples
    --------
    >>> clf = AMDClassifier()
    >>> res = clf.AMDcritfromelements([1.0, 1.4, 1.8], [1e-5] * 3,
    ...                               eccs=[0.01, 0.005, 0.01])
    >>> res.beta, res.stable          # doctest: +SKIP
    """

    def __init__(self, criterion="full", method="SLSQP",
                 constraint_tol=1e-4, agreement_rtol=1e-3, verbose=False):
        self.criterion = criterion
        self.method = method
        self.solver_options = {
            "constraint_tol": constraint_tol,
            "agreement_rtol": agreement_rtol,
            "verbose": verbose,
        }
        # Fail early on a typo rather than at the first triplet.
        if criterion not in ("full", "2body"):
            raise ValueError("criterion must be 'full' or '2body'.")
        if method not in ("SLSQP", "trust-constr"):
            raise ValueError("method must be 'SLSQP' or 'trust-constr'.")

    def __repr__(self):
        return f"AMDClassifier(criterion={self.criterion!r}, method={self.method!r})"

    # ------------------------------------------------------------------
    # General AMD-stability criterion
    # ------------------------------------------------------------------
    def AMDcritfromelements(self, periods, masses, eccs=None, incs=None, m0=1.0, G=1.0):
        """General AMD-stability criterion from orbital elements.

        Parameters
        ----------
        periods, masses : sequence of N floats
            Orbital periods (strictly increasing) and planet masses.
        eccs, incs : sequence of N floats, optional
            Eccentricities and inclinations (radians), used only for the actual
            AMD.  Without ``eccs`` only ``AMD_crit`` is computed.
        m0, G : float
            Central mass and gravitational constant.

        Returns
        -------
        AMDStabilityResult
        """
        periods, masses = _check_elements(periods, masses, eccs, incs)
        AMD = None if eccs is None else _amd.amd_from_elements(
            periods, masses, eccs, incs=incs, m0=m0, G=G)
        return self._general(periods, masses, AMD, m0, G)

    def AMDcritfromsimulation(self, sim, with_inclination=False):
        """General AMD-stability criterion from a ``rebound`` simulation.

        The actual AMD is computed with :func:`amdstability.amd.amd_from_sim`
        (planar by default, ``with_inclination=True`` to include ``cos i``).
        """
        el = elements_from_simulation(sim)
        periods, masses = _check_elements(el["periods"], el["masses"], None, None)
        AMD = _amd.amd_from_sim(sim, coplanar=not with_inclination)
        return self._general(periods, masses, AMD, el["m0"], el["G"])

    def _general(self, periods, masses, AMD, m0, G):
        n = periods.size
        name = "general" if n == 2 else f"general ({self.criterion})"
        result = AMDStabilityResult(criterion=name, n_planets=n, AMD=AMD)

        if n == 2:
            out = critical_amd_pair(periods, masses, m0=m0, G=G)
            result.subsystems.append(self._subsystem((0, 1), out, AMD))
        elif n >= 3:
            for i in range(n - 2):
                idx = slice(i, i + 3)
                out = critical_amd_triplet(
                    periods[idx], masses[idx], m0=m0, G=G,
                    criterion=self.criterion, method=self.method,
                    **self.solver_options,
                )
                result.subsystems.append(self._subsystem((i, i + 1, i + 2), out, AMD))
        return result

    @staticmethod
    def _subsystem(indices, out, AMD):
        return SubsystemResult(
            indices=indices,
            AMD_crit=out["AMD_crit"],
            beta=_beta(AMD, out["AMD_crit"]),
            c=out["c"],
            tau_residual=out["tau_residual"],
            method=out["method"],
            degenerate=out["degenerate"],
        )

    # ------------------------------------------------------------------
    # Classical two-planet criteria, applied to adjacent pairs
    # ------------------------------------------------------------------
    def HillAMDcrit(self, periods, masses, eccs=None, incs=None, m0=1.0, G=1.0):
        """Hill-stability critical AMD (Petit, Laskar & Boue 2018), per adjacent pair."""
        return self._pairwise("Hill", _tpc.Cc_hill, periods, masses, eccs, incs, m0, G)

    def MMRAMDcrit(self, periods, masses, eccs=None, incs=None, m0=1.0, G=1.0):
        """MMR-overlap critical AMD (Petit, Laskar & Boue 2017), per adjacent pair."""
        return self._pairwise("MMR (Petit+2017)", _tpc.Cc_MMR,
                              periods, masses, eccs, incs, m0, G)

    def CollisionAMDcrit(self, periods, masses, eccs=None, incs=None, m0=1.0, G=1.0):
        """Collision critical AMD (Laskar & Petit 2017), per adjacent pair."""
        def Cc(alpha, epsilon, gamma):
            return _tpc.Cc_col(alpha, gamma)
        return self._pairwise("Collision (Laskar & Petit 2017)", Cc,
                              periods, masses, eccs, incs, m0, G)

    def _pairwise(self, name, Cc_func, periods, masses, eccs, incs, m0, G):
        """Apply a relative criterion ``Cc_func(alpha, epsilon, gamma)`` to each adjacent pair.

        ``Cc_func`` returns ``C_c / Lambda'`` (Laskar & Petit 2017); it is
        converted to an absolute AMD with the outer planet ``Lambda'``.
        """
        periods, masses = _check_elements(periods, masses, eccs, incs)
        AMD = None if eccs is None else _amd.amd_from_elements(
            periods, masses, eccs, incs=incs, m0=m0, G=G)
        L = _amd.lambdas_from_periods(periods, masses, m0=m0, G=G)

        result = AMDStabilityResult(criterion=name, n_planets=periods.size, AMD=AMD)
        for i in range(periods.size - 1):
            alpha = (periods[i] / periods[i + 1]) ** (2 / 3)
            gamma = masses[i] / masses[i + 1]
            epsilon = (masses[i] + masses[i + 1]) / m0
            AMD_crit = float(Cc_func(alpha, epsilon, gamma)) * L[i + 1]
            result.subsystems.append(SubsystemResult(
                indices=(i, i + 1),
                AMD_crit=AMD_crit,
                beta=_beta(AMD, AMD_crit),
            ))
        return result
