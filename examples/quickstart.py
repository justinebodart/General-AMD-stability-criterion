"""Quick tour of amdstability.  Run: python examples/quickstart.py"""

import numpy as np

from amdstability import AMDClassifier, elements_from_simulation

clf = AMDClassifier()            # criterion="full", method="SLSQP"

# --- 3 planets: the general criterion --------------------------------------
res = clf.AMDcritfromelements(
    periods=[1.0, 1.4, 1.8],
    masses=[1e-5, 1e-5, 1e-5],
    eccs=[0.01, 0.005, 0.01],
)
print(res, "\n")
print("critical eccentricities:", res.limiting.c, "\n")

# --- 2 planets: analytic (Hadden & Lithwick 2018 / Tamayo et al. 2021) ------
print(clf.AMDcritfromelements([1.0, 1.3], [3e-5, 1e-5], eccs=[0.02, 0.01]), "\n")

# --- 5 planets: adjacent triplets, most constraining one wins ---------------
P = [1.0, 1.35, 1.8, 2.5, 3.3]
m = [1e-5, 2e-5, 1e-5, 3e-5, 1e-5]
e = [0.01, 0.02, 0.01, 0.01, 0.02]
res5 = clf.AMDcritfromelements(P, m, eccs=e)
print(res5, "\n")

# --- classical two-planet criteria on the same system -----------------------
print(clf.HillAMDcrit(P, m, eccs=e), "\n")
print(clf.MMRAMDcrit(P, m, eccs=e), "\n")
print(clf.CollisionAMDcrit(P, m, eccs=e), "\n")

# --- results as a table ------------------------------------------------------
try:
    import pandas as pd
    print(pd.DataFrame(res5.to_dict()["subsystems"]), "\n")
except ImportError:
    pass

# --- from a REBOUND simulation ----------------------------------------------
try:
    import rebound
except ImportError:
    rebound = None

if rebound is not None:
    sim = rebound.Simulation()
    sim.add(m=1.0)
    for Pi, mi, ei in zip(P, m, e):
        sim.add(m=mi, P=Pi * 2 * np.pi, e=ei, pomega=np.random.uniform(0, 2 * np.pi))
    print(clf.AMDcritfromsimulation(sim), "\n")
    print(clf.HillAMDcrit(**elements_from_simulation(sim)))
