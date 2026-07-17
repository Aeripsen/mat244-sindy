# Recovering ODEs from data with SINDy

Code for my MAT244 (Summer 2026, University of Toronto) final project. The
question: given only measurements of a trajectory, can you recover the
differential equations that produced it? The method is Sparse Identification
of Nonlinear Dynamics (SINDy, Brunton, Proctor, Kutz 2016): estimate dX/dt
from the data, build a library of candidate polynomial terms, and solve for a
sparse coefficient matrix by sequentially thresholded least squares.

Everything is written from scratch on top of NumPy/SciPy; PySINDy is only
used as an independent check of my implementation.

## Files

- `systems.py` - the three test systems (damped oscillator, Lotka-Volterra,
  Lorenz), their true coefficients, simulation, noise.
- `sindy.py` - polynomial library, sequentially thresholded least squares,
  derivative estimators (finite differences, Savitzky-Golay), metrics.
- `experiments.py` - the six experiments in the report (clean recovery, noise
  sweep, step size and data length, threshold sweep, library size,
  single-orbit identifiability). Writes `results/*.npz`.
- `make_figures.py` - turns `results/` into the report figures in `figures/`.
- `check_pysindy.py` - optional cross-check against PySINDy (needs
  `pip install pysindy`).

## Run

```
pip install -r requirements.txt
python experiments.py     # a few minutes, writes results/
python make_figures.py    # writes figures/
```

Random seeds are fixed, so the numbers and figures in the report reproduce
exactly.
