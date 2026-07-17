"""The three test systems and a helper to simulate them.

Each system knows its right-hand side and its true coefficients written in the
polynomial-library coordinates that sindy.py uses, so recovered models can be
compared against the truth term by term.
"""

import numpy as np
from scipy.integrate import solve_ivp


def linear2d_rhs(t, s):
    # damped oscillator from Brunton et al. (2016): decays while rotating
    x, y = s
    return [-0.1 * x + 2.0 * y, -2.0 * x - 0.1 * y]


def lotka_rhs(t, s):
    # predator-prey, a=1, b=0.5, c=1, d=0.25 (center at x=4, y=2)
    x, y = s
    return [1.0 * x - 0.5 * x * y, -1.0 * y + 0.25 * x * y]


def lorenz_rhs(t, s):
    # classic parameters sigma=10, rho=28, beta=8/3
    x, y, z = s
    return [10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z]


# true coefficients as {state index: {monomial: value}}, where a monomial is a
# sorted tuple of variable indices, e.g. () = 1, (0,) = x, (0, 1) = x*y
SYSTEMS = {
    "linear2d": {
        "rhs": linear2d_rhs,
        "n": 2,
        "x0": [2.0, 0.0],
        "names": ["x", "y"],
        "true": {
            0: {(0,): -0.1, (1,): 2.0},
            1: {(0,): -2.0, (1,): -0.1},
        },
    },
    "lotka": {
        "rhs": lotka_rhs,
        "n": 2,
        "x0": [3.0, 1.0],
        "names": ["x", "y"],
        "true": {
            0: {(0,): 1.0, (0, 1): -0.5},
            1: {(1,): -1.0, (0, 1): 0.25},
        },
    },
    "lorenz": {
        "rhs": lorenz_rhs,
        "n": 3,
        "x0": [-8.0, 7.0, 27.0],
        "names": ["x", "y", "z"],
        "true": {
            0: {(0,): -10.0, (1,): 10.0},
            1: {(0,): 28.0, (1,): -1.0, (0, 2): -1.0},
            2: {(2,): -8.0 / 3.0, (0, 1): 1.0},
        },
    },
}


def simulate(name, T=25.0, dt=0.01, x0=None, rtol=1e-10, atol=1e-10):
    """Integrate a named system and return (t, X) with X of shape (N, n)."""
    sys = SYSTEMS[name]
    if x0 is None:
        x0 = sys["x0"]
    t = np.arange(0.0, T + dt / 2, dt)
    sol = solve_ivp(sys["rhs"], (t[0], t[-1]), x0, t_eval=t,
                    method="RK45", rtol=rtol, atol=atol)
    return sol.t, sol.y.T


def add_noise(X, level, rng):
    """Additive Gaussian noise, sigma = level * std of each state component."""
    if level == 0:
        return X.copy()
    scale = level * X.std(axis=0)
    return X + rng.normal(0.0, 1.0, X.shape) * scale
