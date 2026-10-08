# General-AMD-stability-criterion

General AMD-Stability Criterion for multi-planet systems. This repository implements a unified analytical framework combining AMD-stability and resonance overlap to predict the onset of chaos.

The Angular Momentum Deficit (AMD) is a conserved quantity that measures the orbital excitation of a planetary system. The classical AMD-stability criterion (Laskar & Petit, 2017) determines whether the available AMD is sufficient for orbital crossing and guarantees long-term stability in the absence of strong resonant effects. However, this criterion does not account for the chaos that can be induced by resonance overlap. This project extends the AMD framework by incorporating the resonance overlap criterion developed by Hadden & Lithwick (2018) and generalized by Tamayo et al. (2021).

This repository contains the implementation of the criterion (the `amdstability` Python package), numerical tools, and examples.

## Installation

Requires Python ≥ 3.9, `numpy` and `scipy`.

```bash
git clone https://github.com/justinebodart/General-AMD-stability-criterion.git
cd General-AMD-stability-criterion
pip install -e .
```

Optional extras:

```bash
pip install -e ".[sim]"        # + rebound, for AMDcritfromsimulation
pip install -e ".[examples]"   # + jupyter, matplotlib, pandas, for the notebook
```

Or, without cloning:

```bash
pip install git+https://github.com/justinebodart/General-AMD-stability-criterion.git
```

> If `pip install -e .` complains about a missing `setup.py`, upgrade pip first:
> `pip install --upgrade pip`.

## Quick start

```python
from amdstability import AMDClassifier

clf = AMDClassifier()
res = clf.AMDcritfromelements(
    periods=[1.0, 1.4, 1.8],
    masses=[1e-5, 1e-5, 1e-5],      # in units of the central mass
    eccs=[0.01, 0.005, 0.01],
)
print(res)
res.beta, res.stable                # beta = AMD / AMD_crit, stable if beta < 1
```

From a `rebound` simulation:

```python
res = clf.AMDcritfromsimulation(sim)
```

## What is computed

| N planets | criterion |
|---|---|
| 1 | trivially stable (`beta = 0`) |
| 2 | analytic critical AMD (Hadden & Lithwick 2018; Tamayo et al. 2021) |
| 3 | general criterion: minimal AMD reaching the MMR-overlap condition `tau = 1` (two-body + three-body terms) |
| > 3 | applied to every set of three adjacent planets; the verdict is that of the most constraining triplet |

The total AMD of the system is compared with the critical AMD of each
subsystem. The classical two-planet criteria are also available:
`HillAMDcrit`, `CollisionAMDcrit` and `MMRAMDcrit`.

## Examples

- [`examples/amd_stability_example.ipynb`](examples/amd_stability_example.ipynb) — guided tour of the package.
- [`examples/quickstart.py`](examples/quickstart.py) — the same, as a script.

## Repository structure

```
General-AMD-stability-criterion/
├── amdstability/           # the Python package
│   ├── __init__.py
│   ├── amd.py              # AMD, circular angular momenta
│   ├── opticaldepth.py     # MMR-overlap optical depth tau
│   ├── twoplanetcrit.py    # classical two-planet criteria
│   ├── criticalamd.py      # critical AMD (N = 2 analytic, N = 3 optimization)
│   ├── classifier.py       # AMDClassifier
│   └── results.py          # result objects
├── examples/
├── pyproject.toml
└── README.md
```
