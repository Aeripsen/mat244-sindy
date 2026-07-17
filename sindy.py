"""Core pieces of SINDy: candidate library, sparse regression, derivatives, metrics.

The whole method is: estimate dX/dt from data, build a library Theta(X) of
candidate terms, then solve Theta * Xi ~ dX/dt asking for a sparse Xi. Written
from the description in Brunton, Proctor, Kutz (2016).
"""

from itertools import combinations_with_replacement

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------- library

def poly_terms(n, degree):
    """All monomials in n variables up to the given degree.

    A monomial is a tuple of variable indices, e.g. () is the constant 1,
    (0,) is x, (0, 1) is x*y, (1, 1) is y^2. Ordered by degree.
    """
    terms = []
    for d in range(degree + 1):
        terms.extend(combinations_with_replacement(range(n), d))
    return terms


def build_library(X, degree):
    """Evaluate every monomial on the data. Returns (Theta, terms)."""
    terms = poly_terms(X.shape[1], degree)
    cols = []
    for term in terms:
        col = np.ones(X.shape[0])
        for idx in term:
            col = col * X[:, idx]
        cols.append(col)
    return np.column_stack(cols), terms


def term_name(term, names):
    """Readable name for a monomial, e.g. (0, 0, 1) -> 'x^2 y'."""
    if len(term) == 0:
        return "1"
    parts = []
    for idx in sorted(set(term)):
        p = term.count(idx)
        parts.append(names[idx] if p == 1 else f"{names[idx]}^{p}")
    return " ".join(parts)


def true_coef_matrix(system, terms):
    """Write a system's true coefficients in library coordinates."""
    Xi = np.zeros((len(terms), system["n"]))
    lookup = {term: j for j, term in enumerate(terms)}
    for i, coefs in system["true"].items():
        for term, value in coefs.items():
            Xi[lookup[term], i] = value
    return Xi


# ---------------------------------------------------------- sparse solve

def stlsq(Theta, dX, lam, max_iter=20):
    """Sequential thresholded least squares.

    Fit by least squares, zero every coefficient smaller than lam, refit on
    the surviving terms, repeat until the active set stops changing.
    """
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
    for _ in range(max_iter):
        active = np.abs(Xi) >= lam
        Xi[~active] = 0.0
        for i in range(dX.shape[1]):
            if active[:, i].any():
                Xi[active[:, i], i] = np.linalg.lstsq(
                    Theta[:, active[:, i]], dX[:, i], rcond=None)[0]
        # refitting can push a coefficient back under lam, so iterate
        if ((np.abs(Xi) >= lam) == active).all():
            break
    Xi[np.abs(Xi) < lam] = 0.0
    return Xi


# ------------------------------------------------------------ derivatives

def deriv_fd(X, dt):
    """Second order central differences (one sided at the ends)."""
    return np.gradient(X, dt, axis=0)


def deriv_savgol(X, dt, window=21, poly=3):
    """Savitzky-Golay derivative: fit a local polynomial, differentiate it."""
    return savgol_filter(X, window_length=window, polyorder=poly,
                         deriv=1, delta=dt, axis=0)


def trim(arrs, k):
    """Drop k samples at each end, where derivative estimates are worst."""
    return [a[k:-k] for a in arrs]


# ---------------------------------------------------------------- metrics

def coef_error(Xi, Xi_true):
    """Relative Frobenius error of the recovered coefficients."""
    return np.linalg.norm(Xi - Xi_true) / np.linalg.norm(Xi_true)


def support(Xi):
    return set(zip(*np.nonzero(Xi)))


def support_stats(Xi, Xi_true):
    """Exact support match, plus precision and recall of the active terms."""
    got, want = support(Xi), support(Xi_true)
    tp = len(got & want)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(want) if want else 1.0
    return {"exact": got == want, "precision": precision, "recall": recall}


def model_str(Xi, terms, names):
    """Print the recovered model as readable equations."""
    lines = []
    for i in range(Xi.shape[1]):
        parts = []
        for j in range(len(terms)):
            if Xi[j, i] != 0.0:
                coef = f"{Xi[j, i]:.4g}"
                parts.append(coef if terms[j] == ()
                             else f"{coef} {term_name(terms[j], names)}")
        lines.append(f"d{names[i]}/dt = {' + '.join(parts) if parts else '0'}")
    return "\n".join(lines)
