"""
amdstability
============

Generalized AMD-stability criterion for multi-planet systems.

Companion code to *"General AMD-stability criterion for multi-planet
systems"*.

Quick start
-----------
>>> from amdstability import AMDClassifier
>>> clf = AMDClassifier()
>>> res = clf.AMDcritfromelements(
...     periods=[1.0, 1.4, 1.8],
...     masses=[1e-5, 1e-5, 1e-5],
...     eccs=[0.01, 0.005, 0.01],
... )
>>> print(res)                      # doctest: +SKIP
>>> res.stable, res.beta            # doctest: +SKIP

The classifier works for any number of planets: ``N = 2`` is solved
analytically, ``N = 3`` is the general criterion, and ``N > 3`` is decomposed
into adjacent triplets (the verdict being that of the most constraining one).
"""

from .classifier import AMDClassifier, elements_from_simulation
from .criticalamd import AMDOptimizationError, critical_amd_pair, critical_amd_triplet
from .results import AMDStabilityResult, SubsystemResult

__version__ = "0.1.0"

__all__ = [
    "AMDClassifier",
    "AMDStabilityResult",
    "SubsystemResult",
    "AMDOptimizationError",
    "critical_amd_pair",
    "critical_amd_triplet",
    "elements_from_simulation",
]
