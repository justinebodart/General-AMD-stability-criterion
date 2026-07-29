# amdrepertory

Generalized AMD-stability criterion for multi-planet (three-planet) systems.

Companion code to the paper *"Generalized AMD-stability criterion for
multi-planet systems"*.

## Idea

For a three-planet system we look for the eccentricity-like configuration
`c = (c1, c2, c3)` that **minimizes the Angular Momentum Deficit (AMD)** while
**saturating the MMR-overlap optical depth** (`tau = 1`). The resulting minimum
AMD is the *critical* AMD. Comparing it to the system's actual AMD gives

```
beta = AMD / AMD_crit
```

with `beta < 1` → stable, `beta > 1` → unstable.

## Install

```bash
pip install -e .
# for the simulation-based helpers and the example notebook:
pip install -e ".[sim,examples]"
```

## Usage

Raw inputs:

```python
from amdrepertory import optimize_amd_crit

out = optimize_amd_crit(
    periods=[1.0, 1.7, 2.9],
    masses=[3e-6, 3e-6, 3e-6],   # in units of central mass if m0=1
)
print(out["c"], out["AMD_crit"], out["tau_residual"])
```

From a rebound simulation:

```python
from amdrepertory import amd_crit_from_sim

out = amd_crit_from_sim(sim, with_inclination=False)
print(out["beta"], out["stable"])
```

## Notes

- The optimizer runs from **two different initial conditions** and returns the
  solution that best satisfies `tau = 1`. If both converge but disagree on the
  AMD value, an `AMDOptimizationError` is raised.
- `SLSQP` is the default (fast, well suited to n=3 with one nonlinear equality
  constraint and analytic jacobians). `method="trust-constr"` is available as a
  more robust fallback.

## Structure

```
amdrepertory/
├── __init__.py
├── functions.py     # physics: tau, AMD, jacobians, constraints, diagnostics
└── optimize.py      # constrained optimization + rebound entry points
examples/
└── amd_stability_example.ipynb
```
